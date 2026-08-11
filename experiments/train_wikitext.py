#!/usr/bin/env python3
"""Train the native SDM WikiText-103 occupancy arm."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from copy_on_write_sdm.allocator import (
    PackedSDMCopyOnWriteState,
    dense_sdm_sparse_step,
    packed_sdm_cow_step,
)
from copy_on_write_sdm.model import LanguageModel
from copy_on_write_sdm.sdm import (
    SparseDeltaMemory,
    aggregate_sdm_copy_on_write_accounting,
    dense_sparse_routes,
    gated_delta_recurrence,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_IDENTITY = json.loads((ROOT / "PROJECT_IDENTITY.json").read_text())[
    "model_identity"
]
VALUE_ELEMENT_BYTES = 4
SLOT_ID_BYTES = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class WikiTextData:
    """Hash-verified deterministic GPT-2-token causal windows."""

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path.resolve()
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("format") != "wikitext103_gpt2_causal_windows_v1":
            raise ValueError("unexpected WikiText stream format")
        if self.manifest.get("storage_dtype") != "uint16":
            raise ValueError("WikiText stream must use uint16")
        self.arrays: dict[str, np.memmap] = {}
        self.hashes: dict[str, str] = {}
        for split, record in self.manifest["records"].items():
            path = self.root / record["path"]
            if path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"size mismatch for {path}")
            digest = sha256_file(path)
            if digest != record["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {path}")
            if record["dtype"] != "uint16":
                raise ValueError(f"unexpected dtype for {path}")
            self.hashes[split] = digest
            self.arrays[split] = np.memmap(
                path,
                mode="r",
                dtype=np.uint16,
                shape=tuple(int(value) for value in record["shape"]),
            )

    def train_batch(self, step: int) -> np.ndarray:
        return np.asarray(self.arrays["train"][step])

    def eval_batches(
        self,
        split: str,
        batch_size: int,
        maximum_examples: int | None = None,
    ) -> Iterator[np.ndarray]:
        values = self.arrays[split]
        stop = len(values) if maximum_examples is None else min(
            len(values), maximum_examples
        )
        for start in range(0, stop, batch_size):
            yield np.asarray(values[start : min(start + batch_size, stop)])


@dataclass(frozen=True)
class TrainConfig:
    arm: str
    layout: str
    steps: int
    schedule_steps: int
    batch_size: int
    micro_batch_size: int
    eval_batch_size: int
    sequence_length: int
    vocab_size: int
    width: int
    heads: int
    slots: int
    reads: int
    writes: int
    memory_heads: int
    mlp_expansion: float
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    seed: int
    activation_dtype: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.int64, copy=False)).to(
        device, non_blocking=True
    )


def learning_rate(step: int, config: TrainConfig) -> float:
    if step <= config.warmup_steps:
        return config.learning_rate * step / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(
        1, config.schedule_steps - config.warmup_steps
    )
    progress = min(max(progress, 0.0), 1.0)
    return config.learning_rate * (
        0.1 + 0.45 * (1.0 + math.cos(math.pi * progress))
    )


def checkpoint_steps(total: int) -> list[int]:
    candidates = (250, 500, 1_000, 2_000, 3_000, 4_000, total)
    return sorted({step for step in candidates if 0 < step <= total})


def curve_summary(
    rows: list[dict[str, Any]], *, step_key: str, value_key: str
) -> dict[str, Any]:
    steps = [int(row[step_key]) for row in rows]
    values = [float(row[value_key]) for row in rows]
    if not values:
        raise ValueError("curve cannot be empty")
    area = sum(
        0.5
        * (values[index - 1] + values[index])
        * (steps[index] - steps[index - 1])
        for index in range(1, len(values))
    )
    return {
        "points": len(values),
        "first_step": steps[0],
        "last_step": steps[-1],
        "mean": sum(values) / len(values),
        "normalized_auc": (
            values[0] if len(values) == 1 else area / (steps[-1] - steps[0])
        ),
    }


def parameter_accounting(model: LanguageModel, config: TrainConfig) -> dict[str, Any]:
    learned_ids = {
        id(parameter)
        for parameter in model.parameters()
        if getattr(parameter, "_sdm_memory_bank", False)
    }
    learned_names = [
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in learned_ids
    ]
    active = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in learned_ids
    )
    learned = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in learned_ids
    )
    layers = config.layout.count("B")
    elements = layers * config.slots * config.width
    map_elements = layers * config.memory_heads * config.slots
    return {
        "active_parameters": active,
        "learned_initial_state_parameters": learned,
        "learned_initial_state_names": learned_names,
        "total_trainable_parameters": active + learned,
        "logical_mutable_state_elements_per_sequence": elements,
        "logical_mutable_state_bytes_fp32_per_sequence": 4 * elements,
        "logical_mutable_state_bytes_bf16_per_sequence": 2 * elements,
        "allocator_map_bytes_per_sequence": 4 * map_elements,
        "sdm_layers": layers,
    }


def native_routing(rows: list[Any]) -> list[Any]:
    routed = [row.sdm for row in rows if row.sdm is not None]
    if len(routed) != sum(row.kind == "B" for row in rows):
        raise AssertionError("native SDM routing diagnostics are incomplete")
    return routed


def describe(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(torch.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


@torch.no_grad()
def evaluate(
    model: LanguageModel,
    data: WikiTextData,
    split: str,
    device: torch.device,
    batch_size: int,
    *,
    maximum_examples: int | None,
    include_examples: bool = False,
    include_occupancy: bool = False,
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    token_count = 0
    examples: list[dict[str, Any]] = []
    unique_rows: list[torch.Tensor] = []
    hard_fractions: list[torch.Tensor] = []
    soft_fractions: list[torch.Tensor] = []
    layer_rows: list[torch.Tensor] = []
    slot_counts: torch.Tensor | None = None
    first_touches = 0
    repeated_writes = 0
    private_read_weight = 0.0
    private_read_count = 0
    read_selections_per_position = 0
    route_entropy_sum = 0.0
    route_entropy_count = 0
    started = time.perf_counter()
    example_index = 0
    for batch in data.eval_batches(split, batch_size, maximum_examples):
        values = to_device(batch, device)
        inputs, targets = values[:, :-1], values[:, 1:]
        with torch.autocast("cuda", dtype=model.activation_dtype):
            forwarded = model(inputs, return_routing=include_occupancy)
            if include_occupancy:
                logits, routing = forwarded
                accounting = aggregate_sdm_copy_on_write_accounting(
                    native_routing(routing), slots=model.stack.sdm_layers[
                        next(iter(model.stack.sdm_layers))
                    ].slots
                )
            else:
                logits = forwarded
            token_losses = F.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="none"
            ).reshape_as(targets)
        loss_sum += float(token_losses.float().sum())
        correct += int((logits.argmax(dim=-1) == targets).sum())
        token_count += targets.numel()
        if include_examples:
            for row_loss in token_losses.float().sum(dim=1):
                nll = float(row_loss) / targets.shape[1]
                examples.append(
                    {
                        "example": example_index,
                        "tokens": targets.shape[1],
                        "nll": nll,
                        "perplexity": math.exp(min(nll, 20.0)),
                    }
                )
                example_index += 1
        if include_occupancy:
            unique_rows.append(accounting["unique_by_position"][:, -1].cpu())
            hard_fractions.append(accounting["hard_final_fraction"].cpu())
            soft_fractions.append(accounting["soft_final_fraction"].cpu())
            layer_rows.append(accounting["unique_final_by_layer"].cpu())
            current_counts = accounting["slot_write_counts"].sum(dim=0).cpu()
            slot_counts = (
                current_counts if slot_counts is None else slot_counts + current_counts
            )
            first_touches += int(accounting["first_touch_by_position"].sum())
            repeated_writes += int(accounting["repeated_write_by_position"].sum())
            private_read_count += int(
                accounting["private_read_count_by_position"].sum()
            )
            private_read_weight += float(
                accounting["private_read_weight_by_position"].sum()
            )
            read_selections_per_position = int(
                accounting["read_selections_per_position"]
            )
            entropy = accounting["route_entropy_by_position"]
            route_entropy_sum += float(entropy.sum())
            route_entropy_count += entropy.numel()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    nll = loss_sum / token_count
    result: dict[str, Any] = {
        "split": split,
        "loss": nll,
        "perplexity": math.exp(min(nll, 20.0)),
        "bits_per_token": nll / math.log(2.0),
        "token_accuracy": correct / token_count,
        "tokens": token_count,
        "seconds": elapsed,
        "tokens_per_second": token_count / elapsed,
    }
    if include_examples:
        result["source_level"] = {
            "source": "WikiText-103",
            "loss": nll,
            "perplexity": math.exp(min(nll, 20.0)),
            "examples": examples,
        }
    if include_occupancy:
        unique = torch.cat(unique_rows)
        hard = torch.cat(hard_fractions)
        soft = torch.cat(soft_fractions)
        by_layer = torch.cat(layer_rows)
        row_width = model.stack.width // next(
            iter(model.stack.sdm_layers.values())
        ).memory_heads
        total_writes = first_touches + repeated_writes
        if slot_counts is None:
            raise AssertionError("occupancy routing histogram is absent")
        routing_layers = []
        for layer in range(slot_counts.shape[0]):
            counts = slot_counts[layer].float()
            probability = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1)
            entropy = -(probability.clamp_min(1e-12).log() * probability).sum(-1)
            routing_layers.append(
                {
                    "layer": layer,
                    "effective_slots": float(entropy.exp().mean()),
                    "normalized_entropy": float(
                        (entropy / math.log(counts.shape[-1])).mean()
                    ),
                    "unused_fraction": float(counts.eq(0).float().mean()),
                    "maximum_to_mean_writes": float(
                        (counts.max(-1).values / counts.mean(-1).clamp_min(1)).mean()
                    ),
                }
            )
        value_bytes = unique * row_width * VALUE_ELEMENT_BYTES
        index_bytes = unique * SLOT_ID_BYTES
        result["occupancy"] = {
            "unique_private_rows": describe(unique),
            "hard_active_fraction": describe(hard),
            "soft_active_fraction": describe(soft),
            "unique_rows_by_layer_mean": by_layer.float().mean(0).tolist(),
            "private_value_bytes": describe(value_bytes),
            "private_row_id_bytes": describe(index_bytes),
            "first_touch_rate": first_touches / max(total_writes, 1),
            "repeated_write_rate": repeated_writes / max(total_writes, 1),
            "private_read_fraction": private_read_count
            / max(token_count * read_selections_per_position, 1),
            "private_read_weight_fraction": private_read_weight
            / max(token_count * config_read_weight_per_position(model), 1),
            "mean_write_route_entropy": route_entropy_sum
            / max(route_entropy_count, 1),
            "routing_layers": routing_layers,
        }
    return result


def config_read_weight_per_position(model: LanguageModel) -> int:
    return sum(
        module.memory_heads for module in model.stack.sdm_layers.values()
    )


def profiler_evidence(profile: torch.profiler.profile, layers: int) -> dict[str, Any]:
    events = {row.key: int(row.count) for row in profile.key_averages()}
    route_calls = int(events.get("sdm_position_parallel_route_generation", 0))
    occupancy_calls = int(events.get("sdm_copy_on_write_occupancy", 0))
    suspicious = sorted(
        key
        for key in events
        if "tokenwise" in key.lower() or "serial_copy_on_write" in key.lower()
    )
    if route_calls != layers or occupancy_calls != layers or suspicious:
        raise AssertionError(
            "native SDM execution path changed: "
            f"routes={route_calls}, occupancy={occupancy_calls}, suspicious={suspicious}"
        )
    return {
        "schema_version": 1,
        "evidence": "torch profiler over the first real optimizer microbatch",
        "position_parallel_route_generation_calls": route_calls,
        "vectorized_occupancy_accounting_calls": occupancy_calls,
        "expected_sdm_layers": layers,
        "tokenwise_controller_loop": False,
        "second_model_pass": False,
        "suspicious_events": suspicious,
    }


def verify_prefix_causality(
    model: LanguageModel, config: TrainConfig, device: torch.device
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(config.seed + 91_003)
    tokens = torch.randint(
        config.vocab_size, (2, 16), generator=generator, device=device
    )
    cut = 7
    changed = tokens.clone()
    changed[:, cut + 1 :] = torch.randint(
        config.vocab_size,
        changed[:, cut + 1 :].shape,
        generator=generator,
        device=device,
    )
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=model.activation_dtype):
        baseline = model(tokens)
        counterfactual = model(changed)
    difference = float(
        (baseline[:, : cut + 1].float() - counterfactual[:, : cut + 1].float())
        .abs()
        .max()
    )
    if difference != 0.0:
        raise AssertionError(f"prefix causality failed: {difference}")
    return {
        "schema_version": 1,
        "status": "causal",
        "maximum_prefix_difference": difference,
        "cut_position": cut,
    }


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    delta = actual.float() - expected.float()
    if float(delta.abs().max()) <= 1e-7:
        return 0.0
    return float(delta.norm() / expected.float().norm().clamp_min(1e-8))


def verify_native_sdm(device: torch.device, config: TrainConfig) -> dict[str, Any]:
    import copy

    width = min(config.width, 64)
    heads = config.memory_heads
    if width % heads:
        width = config.width
    fast = SparseDeltaMemory(
        width,
        slots=config.slots,
        reads=config.reads,
        writes=config.writes,
        memory_heads=heads,
    ).to(device)
    fast.reset_role_keyed(config.seed * 100_000 + 12_101)
    reference = copy.deepcopy(fast)
    source = torch.randn(1, 6, width, device=device, requires_grad=True)
    reference_source = source.detach().clone().requires_grad_(True)
    probe = torch.randn_like(source)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output, routing = fast(
            source, return_routing=True, include_final_memory=True
        )
        serial_output, serial_routing = reference(
            reference_source,
            return_routing=True,
            include_final_memory=True,
            serial_reference=True,
        )
        (output.float() * probe).sum().backward()
        (serial_output.float() * probe).sum().backward()
    parameter_gradient = 0.0
    for (name, parameter), (reference_name, reference_parameter) in zip(
        fast.named_parameters(), reference.named_parameters(), strict=True
    ):
        if name != reference_name:
            raise AssertionError("native SDM parameter order changed")
        if parameter.grad is not None:
            parameter_gradient = max(
                parameter_gradient,
                relative_l2(parameter.grad, reference_parameter.grad),
            )
    if routing.final_memory is None or serial_routing.final_memory is None:
        raise AssertionError("SDM validation final state is absent")

    batch, time, memory_heads, _ = routing.write_indices.shape
    write_indices = routing.write_indices.permute(0, 2, 1, 3).flatten(0, 1)
    write_weights = routing.write_weights.permute(0, 2, 1, 3).flatten(0, 1)
    read_indices = routing.read_indices.permute(0, 2, 1, 3).flatten(0, 1)
    read_weights = routing.read_weights.permute(0, 2, 1, 3).flatten(0, 1)
    write_routes = dense_sparse_routes(write_weights, write_indices, config.slots)
    read_routes = dense_sparse_routes(read_weights, read_indices, config.slots)
    values = routing.values.permute(0, 2, 1, 3).flatten(0, 1)
    input_gate = routing.input_gate.permute(0, 2, 1).flatten(0, 1)
    forget = routing.forget_log_gate.permute(0, 2, 1).flatten(0, 1)
    initial = (
        fast.initial_memory.unsqueeze(0)
        .expand(batch, -1, -1, -1)
        .flatten(0, 1)
        .to(values.dtype)
    )
    split = time // 2
    first, carried = gated_delta_recurrence(
        initial,
        write_routes[:, :split],
        values[:, :split],
        input_gate[:, :split],
        forget[:, :split],
        read_routes[:, :split],
    )
    second, split_final = gated_delta_recurrence(
        carried,
        write_routes[:, split:],
        values[:, split:],
        input_gate[:, split:],
        forget[:, split:],
        read_routes[:, split:],
    )
    full, full_final = gated_delta_recurrence(
        initial, write_routes, values, input_gate, forget, read_routes
    )
    checks = {
        "output_relative_l2": relative_l2(output, serial_output),
        "input_gradient_relative_l2": relative_l2(
            source.grad, reference_source.grad
        ),
        "parameter_gradient_max_relative_l2": parameter_gradient,
        "final_state_relative_l2": relative_l2(
            routing.final_memory, serial_routing.final_memory
        ),
        "chunk_output_relative_l2": relative_l2(
            torch.cat((first, second), dim=1), full
        ),
        "chunk_state_relative_l2": relative_l2(split_final, full_final),
    }
    if max(checks.values()) > 0.035:
        raise AssertionError(f"native SDM differs from serial reference: {checks}")
    return {
        "schema_version": 1,
        "status": "equivalent",
        "relative_l2_tolerance": 0.035,
        **checks,
        "model_identity": MODEL_IDENTITY,
    }


def verify_allocator(device: torch.device, config: TrainConfig) -> dict[str, Any]:
    banks = 3
    width = config.width // config.memory_heads
    initial = torch.randn(1, config.slots, width, device=device)
    dense = initial.expand(banks, -1, -1).clone().float()
    packed = PackedSDMCopyOnWriteState.allocate(
        initial,
        banks=banks,
        capacity_rows=banks * config.slots,
        state_dtype=torch.float32,
    )
    maximum_read_difference = 0.0
    maximum_state_difference = 0.0
    for step in range(3):
        write_indices = torch.stack(
            [
                (torch.arange(config.writes, device=device) + row + step)
                % config.slots
                for row in range(banks)
            ]
        )
        write_weights = torch.softmax(
            torch.randn(banks, config.writes, device=device), dim=-1
        )
        read_indices = torch.stack(
            [
                (torch.arange(config.reads, device=device) + row) % config.slots
                for row in range(banks)
            ]
        )
        read_weights = torch.softmax(
            torch.randn(banks, config.reads, device=device), dim=-1
        )
        values = torch.randn(banks, width, device=device)
        input_gate = torch.sigmoid(torch.randn(banks, device=device))
        forget = -torch.rand(banks, device=device)
        dense_read = dense_sdm_sparse_step(
            dense,
            write_indices,
            write_weights,
            values,
            input_gate,
            forget,
            read_indices,
            read_weights,
        )
        packed_read, _ = packed_sdm_cow_step(
            packed,
            write_indices,
            write_weights,
            values,
            input_gate,
            forget,
            read_indices,
            read_weights,
        )
        maximum_read_difference = max(
            maximum_read_difference,
            float((dense_read.float() - packed_read.float()).abs().max()),
        )
        maximum_state_difference = max(
            maximum_state_difference,
            float((dense - packed.materialize_dense()).abs().max()),
        )
    if maximum_read_difference != 0.0 or maximum_state_difference != 0.0:
        raise AssertionError("packed allocator changed native SDM state")
    return {
        "schema_version": 1,
        "status": "exact",
        "dense_packed_read_maximum_difference": maximum_read_difference,
        "dense_packed_state_maximum_difference": maximum_state_difference,
        "allocated_rows": int(packed.allocated_rows_tensor()),
        "invariants": packed.validate_invariants(),
    }


@torch.no_grad()
def export_trained_occupancy(
    model: LanguageModel,
    data: WikiTextData,
    device: torch.device,
    *,
    examples: int,
    prefix_lengths: tuple[int, ...],
    output: Path,
) -> dict[str, Any]:
    model.eval()
    all_routes: list[torch.Tensor] = []
    for batch in data.eval_batches("validation", 1, examples):
        values = to_device(batch, device)
        with torch.autocast("cuda", dtype=model.activation_dtype):
            _, routing = model(values[:, :-1], return_routing=True)
        layers = [row.write_indices.to(torch.int16).cpu() for row in native_routing(routing)]
        all_routes.append(torch.stack(layers, dim=1))
    routes = torch.cat(all_routes, dim=0)
    route_path = output / "serving_write_routes.pt"
    atomic_torch_save(
        route_path,
        {
            "schema": "native-sdm-serving-write-routes-v1",
            "write_indices": routes,
            "prefix_lengths": prefix_lengths,
        },
    )
    layers = routes.shape[1]
    capacity = layers * model.stack.sdm_layers[
        next(iter(model.stack.sdm_layers))
    ].memory_heads * model.stack.sdm_layers[
        next(iter(model.stack.sdm_layers))
    ].slots
    rows = []
    for prefix in prefix_lengths:
        per_example = []
        by_layer = []
        for example in range(routes.shape[0]):
            counts = []
            for layer in range(layers):
                counts.append(
                    int(torch.unique(routes[example, layer, :prefix].long()).numel())
                )
            per_example.append(sum(counts))
            by_layer.append(counts)
        values = torch.tensor(per_example, dtype=torch.float32)
        layer_values = torch.tensor(by_layer, dtype=torch.float32)
        rows.append(
            {
                "prefix_tokens": prefix,
                "unique_rows": describe(values),
                "mean_active_fraction": float(values.mean() / capacity),
                "mean_private_value_bytes_fp32": float(
                    values.mean()
                    * next(iter(model.stack.sdm_layers.values())).head_width
                    * VALUE_ELEMENT_BYTES
                ),
                "mean_rows_by_layer": layer_values.mean(0).tolist(),
            }
        )
    result = {
        "schema": "native-sdm-trained-occupancy-v1",
        "examples": routes.shape[0],
        "logical_rows_per_sequence": capacity,
        "prefixes": rows,
        "route_artifact": route_path.name,
        "route_artifact_sha256": sha256_file(route_path),
    }
    atomic_json(output / "TRAINED_OCCUPANCY.json", result)
    return result


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    try:
        import triton
    except ImportError as error:
        raise SystemExit("Triton is required") from error
    data = WikiTextData(args.manifest)
    manifest = data.manifest
    if args.steps > int(manifest["steps"]):
        raise ValueError("steps exceed deterministic stream length")
    if args.batch_size != int(manifest["batch_size"]):
        raise ValueError("batch size must match deterministic stream")
    if args.batch_size % args.micro_batch_size:
        raise ValueError("micro batch size must divide batch size")
    if len(args.layout) == 0 or any(kind not in "AB" for kind in args.layout):
        raise ValueError("layout must contain only A and B")
    if not args.layout.count("B"):
        raise ValueError("native SDM training requires at least one B layer")
    if args.schedule_steps < args.steps:
        raise ValueError("schedule cannot be shorter than training")
    config = TrainConfig(
        arm=args.arm,
        layout=args.layout,
        steps=args.steps,
        schedule_steps=args.schedule_steps,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        eval_batch_size=args.eval_batch_size,
        sequence_length=int(manifest["seq_len"]),
        vocab_size=int(manifest["tokenizer_vocab_size"]),
        width=args.width,
        heads=args.heads,
        slots=args.slots,
        reads=args.reads,
        writes=args.writes,
        memory_heads=args.memory_heads,
        mlp_expansion=args.mlp_expansion,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        activation_dtype=args.activation_dtype,
    )
    set_seed(config.seed)
    device = torch.device("cuda")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[
        config.activation_dtype
    ]
    model = LanguageModel(
        vocab_size=config.vocab_size,
        maximum_sequence_length=config.sequence_length,
        layout=config.layout,
        width=config.width,
        heads=config.heads,
        slots=config.slots,
        reads=config.reads,
        writes=config.writes,
        memory_heads=config.memory_heads,
        mlp_expansion=config.mlp_expansion,
        activation_dtype=dtype,
    )
    model.initialize_role_keyed(config.seed)
    model = model.to(device)
    accounting = parameter_accounting(model, config)
    if (
        args.expected_active_parameters is not None
        and accounting["active_parameters"] != args.expected_active_parameters
    ):
        raise ValueError("active parameter count differs from the experiment specification")
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (no_decay if getattr(parameter, "_no_weight_decay", False) else decay).append(
                parameter
            )
    groups: list[dict[str, Any]] = [
        {"params": decay, "weight_decay": config.weight_decay}
    ]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
        fused=True,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "config.json",
        {
            **asdict(config),
            "model_identity": MODEL_IDENTITY,
            "manifest_sha256": sha256_file(data.manifest_path),
            "stream_sha256": data.hashes,
            "parameter_accounting": accounting,
            "objective": "token_nll",
        },
    )
    atomic_json(output / "SDM_VALIDATION.json", verify_native_sdm(device, config))
    atomic_json(output / "ALLOCATOR_VALIDATION.json", verify_allocator(device, config))
    atomic_json(output / "CAUSALITY.json", verify_prefix_causality(model, config, device))

    metrics: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    pending: list[tuple[int, float, torch.Tensor]] = []
    recoveries: list[dict[str, Any]] = []
    checkpoints = checkpoint_steps(config.steps)
    recovery_steps = {
        *range(
            args.recovery_checkpoint_interval,
            config.steps + 1,
            args.recovery_checkpoint_interval,
        ),
        config.steps,
    }
    metric_path = output / "metrics.jsonl"
    curve_path = output / "training_curve.jsonl"
    started = time.perf_counter()
    interval_started = started
    interval_tokens = 0
    optimizer_seconds = 0.0
    evaluation_seconds = 0.0
    recovery_seconds = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    with metric_path.open("w") as metric_handle, curve_path.open("w") as curve_handle:
        for step_index in range(config.steps):
            step = step_index + 1
            lr = learning_rate(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = data.train_batch(step_index)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            task_total = torch.zeros((), device=device)
            first_profile: torch.profiler.profile | None = None
            for start in range(0, config.batch_size, config.micro_batch_size):
                values = to_device(batch[start : start + config.micro_batch_size], device)
                inputs, targets = values[:, :-1], values[:, 1:]
                profile = (
                    torch.profiler.profile(
                        activities=(
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        )
                    )
                    if step == 1 and start == 0
                    else None
                )
                profile_context = profile if profile is not None else nullcontext()
                with profile_context:
                    with torch.autocast("cuda", dtype=model.activation_dtype):
                        logits = model(inputs)
                        task_loss = F.cross_entropy(
                            logits.flatten(0, 1), targets.flatten()
                        )
                        scaled = task_loss * (
                            config.micro_batch_size / config.batch_size
                        )
                    scaled.backward()
                if profile is not None:
                    first_profile = profile
                scale = config.micro_batch_size / config.batch_size
                task_total += task_loss.detach() * scale
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            before = (
                model.token_embedding.weight.detach().clone() if step == 1 else None
            )
            optimizer.step()
            interval_tokens += config.batch_size * config.sequence_length
            pending.append((step, lr, task_total))

            if step == 1:
                torch.cuda.synchronize(device)
                update = float(
                    (model.token_embedding.weight.detach() - before).abs().max()
                )
                health = {
                    "schema_version": 1,
                    "status": "healthy",
                    "step": 1,
                    "task_loss": float(task_total),
                    "gradient_norm": float(gradient_norm),
                    "maximum_parameter_update": update,
                    "gpu": torch.cuda.get_device_name(device),
                }
                if not all(
                    math.isfinite(float(health[key]))
                    for key in (
                        "task_loss",
                        "gradient_norm",
                        "maximum_parameter_update",
                    )
                ) or update <= 0:
                    raise FloatingPointError(f"invalid first optimizer step: {health}")
                atomic_json(output / "HEALTHY.json", health)
                if first_profile is None:
                    raise AssertionError("first-step execution profile is absent")
                atomic_json(
                    output / "EXECUTION_PATH.json",
                    profiler_evidence(first_profile, config.layout.count("B")),
                )

            if step in checkpoints:
                torch.cuda.synchronize(device)
                now = time.perf_counter()
                interval_elapsed = now - interval_started
                optimizer_seconds += interval_elapsed
                stacked = torch.stack(
                    [row[2].float() for row in pending]
                ).cpu()
                for pending_row, values in zip(pending, stacked, strict=True):
                    curve_row = {
                        "step": pending_row[0],
                        "learning_rate": pending_row[1],
                        "train_task_loss": float(values),
                    }
                    curve.append(curve_row)
                    curve_handle.write(json.dumps(curve_row, sort_keys=True) + "\n")
                curve_handle.flush()
                pending.clear()
                evaluation_started = time.perf_counter()
                validation = evaluate(
                    model,
                    data,
                    "validation",
                    device,
                    config.eval_batch_size,
                    maximum_examples=args.checkpoint_eval_examples,
                )
                evaluation_seconds += time.perf_counter() - evaluation_started
                metric = {
                    "step": step,
                    "train_task_loss": float(task_total),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": lr,
                    "training_tokens_per_second": interval_tokens
                    / max(interval_elapsed, 1e-9),
                    "elapsed_seconds": now - started,
                    "validation": validation,
                }
                metrics.append(metric)
                metric_handle.write(json.dumps(metric, sort_keys=True) + "\n")
                metric_handle.flush()
                print(json.dumps(metric, sort_keys=True), flush=True)
                interval_tokens = 0
                interval_started = time.perf_counter()

            if step in recovery_steps:
                torch.cuda.synchronize(device)
                recovery_started = time.perf_counter()
                checkpoint = output / "recovery_checkpoint.pt"
                atomic_torch_save(
                    checkpoint,
                    {
                        "schema_version": 1,
                        "kind": "copy_on_write_sdm_wikitext_recovery",
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": asdict(config),
                        "step": step,
                        "manifest_sha256": sha256_file(data.manifest_path),
                        "metrics": metrics,
                        "training_curve": curve,
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_states": torch.cuda.get_rng_state_all(),
                    },
                )
                recovery_elapsed = time.perf_counter() - recovery_started
                recovery_seconds += recovery_elapsed
                recovery = {
                    "schema_version": 1,
                    "status": "recoverable",
                    "artifact": checkpoint.name,
                    "step": step,
                    "base_complete": step == config.steps,
                    "sha256": sha256_file(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                    "write_and_hash_seconds": recovery_elapsed,
                }
                recoveries.append(recovery)
                atomic_json(output / "RECOVERY.json", recovery)
                atomic_json(
                    output / "RECOVERY_AUDIT.json",
                    {
                        "schema_version": 1,
                        "interval_steps": args.recovery_checkpoint_interval,
                        "checkpoints": recoveries,
                        "latest": recovery,
                    },
                )
                interval_started = time.perf_counter()

    evaluation_started = time.perf_counter()
    final_validation = evaluate(
        model,
        data,
        "validation",
        device,
        config.eval_batch_size,
        maximum_examples=args.final_eval_examples,
        include_occupancy=True,
    )
    final_test = evaluate(
        model,
        data,
        "test",
        device,
        config.eval_batch_size,
        maximum_examples=args.final_eval_examples,
        include_examples=True,
        include_occupancy=True,
    )
    prefix_lengths = tuple(
        int(value) for value in args.serving_route_lengths.split(",") if value
    )
    trained_occupancy = export_trained_occupancy(
        model,
        data,
        device,
        examples=args.serving_route_examples,
        prefix_lengths=prefix_lengths,
        output=output,
    )
    evaluation_seconds += time.perf_counter() - evaluation_started
    if args.save_checkpoint:
        atomic_torch_save(
            output / "final_checkpoint.pt",
            {
                "model": model.state_dict(),
                "config": asdict(config),
                "step": config.steps,
                "manifest_sha256": sha256_file(data.manifest_path),
            },
        )
    total_seconds = time.perf_counter() - started
    result = {
        "schema": "copy-on-write-sdm-wikitext103-v1",
        "model_identity": MODEL_IDENTITY,
        "config": asdict(config),
        "manifest_sha256": sha256_file(data.manifest_path),
        "stream_sha256": data.hashes,
        "parameter_accounting": accounting,
        "optimizer_seconds": optimizer_seconds,
        "evaluation_seconds": evaluation_seconds,
        "recovery_checkpoint_seconds": recovery_seconds,
        "total_seconds": total_seconds,
        "training_tokens": config.steps * config.batch_size * config.sequence_length,
        "optimizer_tokens_per_second": config.steps
        * config.batch_size
        * config.sequence_length
        / optimizer_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "final_validation": final_validation,
        "final_test": final_test,
        "trained_occupancy": trained_occupancy,
        "curve_summary": {
            "complete_task_loss": curve_summary(
                curve, step_key="step", value_key="train_task_loss"
            ),
            "checkpoint_validation": curve_summary(
                [
                    {
                        "step": row["step"],
                        "validation_loss": row["validation"]["loss"],
                    }
                    for row in metrics
                ],
                step_key="step",
                value_key="validation_loss",
            ),
        },
        "training_curve": curve,
        "metrics": metrics,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
    }
    atomic_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--schedule-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-eval-examples", type=int, default=32)
    parser.add_argument("--final-eval-examples", type=int, default=128)
    parser.add_argument("--serving-route-examples", type=int, default=64)
    parser.add_argument("--serving-route-lengths", default="16,64,256,1024,2048")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--slots", type=int, default=256)
    parser.add_argument("--reads", type=int, default=16)
    parser.add_argument("--writes", type=int, default=4)
    parser.add_argument("--memory-heads", type=int, default=1)
    parser.add_argument("--mlp-expansion", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--activation-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--expected-active-parameters", type=int)
    parser.add_argument("--recovery-checkpoint-interval", type=int, default=1_000)
    parser.add_argument(
        "--save-checkpoint", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    if args.recovery_checkpoint_interval <= 0:
        parser.error("recovery checkpoint interval must be positive")
    if args.serving_route_examples <= 0:
        parser.error("serving route examples must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
