#!/usr/bin/env python3
"""Validate and measure full-stack faithful SDM copy-on-write serving."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any

import torch

from copy_on_write_sdm.model import LanguageModel
from copy_on_write_sdm.inference import (
    FaithfulSDMDecodeState,
    _decode_faithful_sdm_layer,
    _decode_faithful_sdm_layer_dense,
)


BF16_OUTPUT_RTOL = 2.0**-6
BF16_OUTPUT_ATOL = 2.0**-6


def bfloat16_outputs_close(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Accept only the narrow rounding envelope of equivalent BF16 schedules."""

    return bool(
        torch.allclose(
            left,
            right,
            rtol=BF16_OUTPUT_RTOL,
            atol=BF16_OUTPUT_ATOL,
        )
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def make_decoder(
    *,
    maximum_sequence_length: int,
    layout: str,
    width: int,
    heads: int,
    slots: int,
) -> LanguageModel:
    return LanguageModel(
        vocab_size=4096,
        maximum_sequence_length=maximum_sequence_length,
        layout=layout,
        width=width,
        heads=heads,
        slots=slots,
        reads=16,
        writes=4,
        memory_heads=1,
        mlp_expansion=4.0,
    ).eval()


@dataclass
class DenseServingState:
    position: int
    capacity: int
    memories: tuple[torch.Tensor, ...]
    dense_keys: tuple[torch.Tensor, ...]
    dense_values: tuple[torch.Tensor, ...]


def dense_from_packed(state: FaithfulSDMDecodeState) -> DenseServingState:
    return DenseServingState(
        position=state.position,
        capacity=state.capacity,
        memories=tuple(row.materialize_dense() for row in state.sdm_states),
        dense_keys=tuple(row.clone() for row in state.dense_keys),
        dense_values=tuple(row.clone() for row in state.dense_values),
    )


@torch.no_grad()
def dense_decode(
    decoder: LanguageModel,
    input_ids: torch.Tensor,
    state: DenseServingState,
) -> torch.Tensor:
    stack = decoder.stack
    position = torch.tensor(
        [state.position],
        device=input_ids.device,
        dtype=torch.int64,
    )
    tokens = decoder.token_embedding(input_ids) + decoder.position_embedding(
        position
    ).unsqueeze(0)
    dense_index = 0
    sdm_index = 0
    for physical_layer, kind in enumerate(stack.layout):
        if kind == "A":
            block = stack.attention_layers[str(physical_layer)]
            tokens = tokens + block.attention.decode_with_cache(
                tokens,
                state.dense_keys[dense_index],
                state.dense_values[dense_index],
                position=state.position,
            )
            tokens = tokens + block.mlp(tokens)
            dense_index += 1
            continue
        module = stack.sdm_layers[str(physical_layer)]
        tokens = tokens + _decode_faithful_sdm_layer_dense(
            module,
            tokens,
            state.memories[sdm_index],
        )
        tokens = tokens + stack.sdm_mlps[str(physical_layer)](tokens)
        sdm_index += 1
    state.position += 1
    return decoder.output(decoder.final_norm(tokens))


@torch.no_grad()
def paired_decode_trace(
    decoder: LanguageModel,
    input_ids: torch.Tensor,
    packed: FaithfulSDMDecodeState,
    dense: DenseServingState,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Execute matched packed/dense decode while locating the first BF16 drift."""

    if packed.positions is not None:
        raise ValueError("paired trace requires an equal-position serving batch")
    if packed.position != dense.position:
        raise ValueError("paired trace states are at different positions")
    position = packed.position
    position_tensor = torch.full(
        (input_ids.shape[0],),
        position,
        device=input_ids.device,
        dtype=torch.int64,
    )
    embedded = decoder.token_embedding(input_ids) + decoder.position_embedding(
        position_tensor
    ).unsqueeze(1)
    packed_tokens = embedded
    dense_tokens = embedded.clone()
    stack = decoder.stack
    dense_index = 0
    sdm_index = 0
    trace: list[dict[str, Any]] = []
    for physical_layer, kind in enumerate(stack.layout):
        layer: dict[str, Any] = {
            "physical_layer": physical_layer,
            "kind": kind,
            "input_maximum_difference": float(
                (packed_tokens - dense_tokens).abs().max()
            ),
        }
        if kind == "A":
            block = stack.attention_layers[str(physical_layer)]
            packed_mixed = block.attention.decode_with_cache(
                packed_tokens,
                packed.dense_keys[dense_index],
                packed.dense_values[dense_index],
                position=position,
            )
            dense_mixed = block.attention.decode_with_cache(
                dense_tokens,
                dense.dense_keys[dense_index],
                dense.dense_values[dense_index],
                position=position,
            )
            layer["key_cache_maximum_difference"] = float(
                (
                    packed.dense_keys[dense_index]
                    - dense.dense_keys[dense_index]
                )
                .abs()
                .max()
            )
            layer["value_cache_maximum_difference"] = float(
                (
                    packed.dense_values[dense_index]
                    - dense.dense_values[dense_index]
                )
                .abs()
                .max()
            )
            mlp = block.mlp
            dense_index += 1
        else:
            module = stack.sdm_layers[str(physical_layer)]
            shadow_memory = packed.sdm_states[sdm_index].materialize_dense()
            packed_mixed = _decode_faithful_sdm_layer(
                module,
                packed_tokens,
                packed.sdm_states[sdm_index],
            )
            shadow_mixed = _decode_faithful_sdm_layer_dense(
                module,
                packed_tokens,
                shadow_memory,
            )
            dense_mixed = _decode_faithful_sdm_layer_dense(
                module,
                dense_tokens,
                dense.memories[sdm_index],
            )
            layer["local_shadow_mixer_maximum_difference"] = float(
                (packed_mixed - shadow_mixed).abs().max()
            )
            layer["local_shadow_mixer_within_bf16_tolerance"] = (
                bfloat16_outputs_close(packed_mixed, shadow_mixed)
            )
            layer["local_shadow_state_maximum_difference"] = float(
                (
                    packed.sdm_states[sdm_index].materialize_dense()
                    - shadow_memory
                )
                .abs()
                .max()
            )
            layer["end_to_end_state_maximum_difference"] = float(
                (
                    packed.sdm_states[sdm_index].materialize_dense()
                    - dense.memories[sdm_index]
                )
                .abs()
                .max()
            )
            mlp = stack.sdm_mlps[str(physical_layer)]
            sdm_index += 1
        layer["mixer_maximum_difference"] = float(
            (packed_mixed - dense_mixed).abs().max()
        )
        packed_tokens = packed_tokens + packed_mixed
        dense_tokens = dense_tokens + dense_mixed
        layer["post_mixer_maximum_difference"] = float(
            (packed_tokens - dense_tokens).abs().max()
        )
        packed_mlp = mlp(packed_tokens)
        dense_mlp = mlp(dense_tokens)
        layer["mlp_maximum_difference"] = float(
            (packed_mlp - dense_mlp).abs().max()
        )
        packed_tokens = packed_tokens + packed_mlp
        dense_tokens = dense_tokens + dense_mlp
        layer["post_layer_maximum_difference"] = float(
            (packed_tokens - dense_tokens).abs().max()
        )
        trace.append(layer)
    packed_output = decoder.output(decoder.final_norm(packed_tokens))
    dense_output = decoder.output(decoder.final_norm(dense_tokens))
    packed.position += 1
    dense.position += 1
    return packed_output, dense_output, trace


@torch.no_grad()
def dense_decode_sdm_only_at_positions(
    decoder: LanguageModel,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    memories: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Dense-table control for a heterogeneous-position all-SDM batch."""

    stack = decoder.stack
    if "A" in stack.layout:
        raise ValueError("mixed-position lifecycle control must be all-SDM")
    if positions.shape != (input_ids.shape[0],):
        raise ValueError("one dense-control position is required per request")
    tokens = decoder.token_embedding(input_ids) + decoder.position_embedding(
        positions
    ).unsqueeze(1)
    sdm_index = 0
    for physical_layer in range(len(stack.layout)):
        module = stack.sdm_layers[str(physical_layer)]
        tokens = tokens + _decode_faithful_sdm_layer_dense(
            module,
            tokens,
            memories[sdm_index],
        )
        tokens = tokens + stack.sdm_mlps[str(physical_layer)](tokens)
        sdm_index += 1
    return decoder.output(decoder.final_norm(tokens))


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def dense_storage_bytes(state: DenseServingState) -> dict[str, int]:
    sdm = sum(tensor_bytes(row) for row in state.memories)
    dense_allocated = sum(
        tensor_bytes(row) for row in (*state.dense_keys, *state.dense_values)
    )
    dense_live = sum(
        row[:, :, : state.position].numel() * row.element_size()
        for row in (*state.dense_keys, *state.dense_values)
    )
    return {
        "sdm_dense_fp32": sdm,
        "dense_kv_live": dense_live,
        "dense_kv_allocated": dense_allocated,
        "live_total": sdm + dense_live,
        "allocated_total": sdm + dense_allocated,
    }


@torch.no_grad()
def correctness(
    *,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch.manual_seed(seed)
    decoder = make_decoder(
        maximum_sequence_length=32,
        layout="BABB",
        width=64,
        heads=4,
        slots=16,
    ).to(device=device, dtype=torch.bfloat16)
    input_ids = torch.randint(4096, (2, 12), device=device)
    prefix_length = 7
    committed_prefix = decoder(input_ids[:, :prefix_length])
    prefix, packed = decoder.prefill(
        input_ids[:, :prefix_length],
        capacity=input_ids.shape[1],
    )
    prefill_difference = float((prefix - committed_prefix).abs().max())
    dense = dense_from_packed(packed)
    maximum_dense_packed_output_difference = 0.0
    dense_packed_outputs_close = True
    maximum_dense_packed_state_difference = 0.0
    maximum_full_prefix_output_difference = 0.0
    maximum_full_prefix_reference_magnitude = 0.0
    for position in range(prefix_length, input_ids.shape[1]):
        token = input_ids[:, position : position + 1]
        packed_output = decoder.decode(token, packed)
        dense_output = dense_decode(decoder, token, dense)
        fresh = decoder(input_ids[:, : position + 1])[:, -1:]
        maximum_dense_packed_output_difference = max(
            maximum_dense_packed_output_difference,
            float((packed_output - dense_output).abs().max()),
        )
        dense_packed_outputs_close = (
            dense_packed_outputs_close
            and bfloat16_outputs_close(packed_output, dense_output)
        )
        maximum_full_prefix_output_difference = max(
            maximum_full_prefix_output_difference,
            float((packed_output - fresh).abs().max()),
        )
        maximum_full_prefix_reference_magnitude = max(
            maximum_full_prefix_reference_magnitude,
            float(fresh.abs().max()),
        )
        for packed_layer, dense_layer in zip(packed.sdm_states, dense.memories):
            maximum_dense_packed_state_difference = max(
                maximum_dense_packed_state_difference,
                float((packed_layer.materialize_dense() - dense_layer).abs().max()),
            )
    changed = input_ids.clone()
    changed[:, prefix_length:] = (
        changed[:, prefix_length:] + 193
    ).remainder(4096)
    causal_prefix = decoder(input_ids)[:, :prefix_length]
    changed_prefix = decoder(changed)[:, :prefix_length]
    causality_difference = float((causal_prefix - changed_prefix).abs().max())
    allocated_before_release = sum(
        int(row.allocated_rows_tensor()) for row in packed.sdm_states
    )
    released = packed.release()
    allocated_after_release = sum(
        int(row.allocated_rows_tensor()) for row in packed.sdm_states
    )
    for row in packed.sdm_states:
        row.validate_invariants()
    finite = all(
        math.isfinite(value)
        for value in (
            prefill_difference,
            maximum_dense_packed_output_difference,
            maximum_dense_packed_state_difference,
            maximum_full_prefix_output_difference,
            causality_difference,
        )
    )
    if not finite:
        raise AssertionError("faithful SDM serving produced a non-finite result")
    if (
        prefill_difference != 0.0
        or not dense_packed_outputs_close
        or maximum_dense_packed_state_difference != 0.0
        or causality_difference != 0.0
        or released != allocated_before_release
        or allocated_after_release != 0
    ):
        raise AssertionError(
            "faithful SDM packed serving failed state/causality/BF16 parity: "
            f"prefill={prefill_difference}, "
            f"output={maximum_dense_packed_output_difference}, "
            f"state={maximum_dense_packed_state_difference}, "
            f"causality={causality_difference}, "
            f"release={released}/{allocated_before_release}"
        )
    parity = {
        "schema": "faithful-sdm-packed-serving-parity-v1",
        "prefill_maximum_difference": prefill_difference,
        "dense_packed_output_maximum_difference": (
            maximum_dense_packed_output_difference
        ),
        "dense_packed_outputs_within_bf16_tolerance": dense_packed_outputs_close,
        "bf16_output_relative_tolerance": BF16_OUTPUT_RTOL,
        "bf16_output_absolute_tolerance": BF16_OUTPUT_ATOL,
        "dense_packed_state_maximum_difference": (
            maximum_dense_packed_state_difference
        ),
        "streamed_full_prefix_bf16_maximum_difference": (
            maximum_full_prefix_output_difference
        ),
        "streamed_full_prefix_bf16_reference_maximum_magnitude": (
            maximum_full_prefix_reference_magnitude
        ),
        "streamed_full_prefix_bf16_relative_to_reference_maximum": (
            maximum_full_prefix_output_difference
            / max(maximum_full_prefix_reference_magnitude, 1e-12)
        ),
        "streamed_full_prefix_interpretation": (
            "diagnostic across different BF16 execution schedules; exact "
            "acceptance is dense-table versus packed streaming parity"
        ),
        "state_dtype": "float32",
        "allocated_rows_before_release": allocated_before_release,
        "released_rows": released,
        "allocated_rows_after_release": allocated_after_release,
    }
    causality = {
        "schema": "faithful-sdm-packed-serving-causality-v1",
        "maximum_prefix_difference": causality_difference,
        "criterion": "maximum prefix difference == 0",
    }
    del decoder, packed, dense
    gc.collect()
    torch.cuda.empty_cache()
    return parity, causality


def elapsed_ms(operation: Any, *, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop)) / iterations


@torch.no_grad()
def continuous_batch_lifecycle(
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Exercise release, compaction, admission, growth, and trim on CUDA."""

    torch.manual_seed(seed)
    decoder = make_decoder(
        maximum_sequence_length=16,
        layout="BBBB",
        width=64,
        heads=4,
        slots=64,
    ).to(device=device, dtype=torch.bfloat16)
    first = torch.randint(4096, (4, 9), device=device)
    added = torch.randint(4096, (2, 6), device=device)
    _, pooled = decoder.prefill(
        first[:, :4],
        capacity=16,
        sdm_growth_quantum_rows=16,
    )
    maximum_output_difference = 0.0
    maximum_state_difference = 0.0
    outputs_within_bf16_tolerance = True
    dense_memories = tuple(
        row.materialize_dense() for row in pooled.sdm_states
    )
    for position in range(4, 6):
        positions = pooled.current_positions().clone()
        token = first[:, position : position + 1]
        pooled_output = decoder.decode(first[:, position : position + 1], pooled)
        expected = dense_decode_sdm_only_at_positions(
            decoder,
            token,
            positions,
            dense_memories,
        )
        maximum_output_difference = max(
            maximum_output_difference,
            float((pooled_output - expected).abs().max()),
        )
        outputs_within_bf16_tolerance = (
            outputs_within_bf16_tolerance
            and bfloat16_outputs_close(pooled_output, expected)
        )
        maximum_state_difference = max(
            maximum_state_difference,
            max(
                float((row.materialize_dense() - control).abs().max())
                for row, control in zip(pooled.sdm_states, dense_memories)
            ),
        )

    capacity_before_release = sum(row.capacity_rows for row in pooled.sdm_states)
    live_before_release = sum(
        int(row.allocated_rows_tensor()) for row in pooled.sdm_states
    )
    released_rows = pooled.release_and_compact(
        [0, 2],
        trim=True,
        trim_headroom_rows=16,
    )
    capacity_after_trim = sum(row.capacity_rows for row in pooled.sdm_states)
    live_after_trim = sum(
        int(row.allocated_rows_tensor()) for row in pooled.sdm_states
    )

    decoder.admit(added[:, :3], pooled)
    capacity_after_admission = sum(
        row.capacity_rows for row in pooled.sdm_states
    )
    live_after_admission = sum(
        int(row.allocated_rows_tensor()) for row in pooled.sdm_states
    )
    assert pooled.current_positions().tolist() == [6, 6, 3, 3]
    dense_memories = tuple(
        row.materialize_dense() for row in pooled.sdm_states
    )

    for offset in range(2):
        tokens = torch.cat(
            (
                first[1::2, 6 + offset : 7 + offset],
                added[:, 3 + offset : 4 + offset],
            ),
            dim=0,
        )
        positions = pooled.current_positions().clone()
        pooled_output = decoder.decode(tokens, pooled)
        expected = dense_decode_sdm_only_at_positions(
            decoder,
            tokens,
            positions,
            dense_memories,
        )
        maximum_output_difference = max(
            maximum_output_difference,
            float((pooled_output - expected).abs().max()),
        )
        outputs_within_bf16_tolerance = (
            outputs_within_bf16_tolerance
            and bfloat16_outputs_close(pooled_output, expected)
        )
        maximum_state_difference = max(
            maximum_state_difference,
            max(
                float((row.materialize_dense() - control).abs().max())
                for row, control in zip(pooled.sdm_states, dense_memories)
            ),
        )
    for row in pooled.sdm_states:
        row.validate_invariants()
    if not outputs_within_bf16_tolerance or maximum_state_difference != 0.0:
        raise AssertionError(
            "continuous-batch packed state failed faithful SDM equivalence: "
            f"output={maximum_output_difference}, "
            f"state={maximum_state_difference}, "
            f"within_bf16_tolerance={outputs_within_bf16_tolerance}"
        )
    return {
        "schema": "faithful-sdm-continuous-pool-lifecycle-v1",
        "maximum_output_difference": maximum_output_difference,
        "outputs_within_bf16_tolerance": outputs_within_bf16_tolerance,
        "bf16_output_relative_tolerance": BF16_OUTPUT_RTOL,
        "bf16_output_absolute_tolerance": BF16_OUTPUT_ATOL,
        "maximum_state_difference": maximum_state_difference,
        "initial_requests": 4,
        "released_requests": 2,
        "admitted_requests": 2,
        "final_positions": pooled.current_positions().tolist(),
        "released_rows": released_rows,
        "capacity_rows_before_release": capacity_before_release,
        "live_rows_before_release": live_before_release,
        "capacity_rows_after_trim": capacity_after_trim,
        "live_rows_after_trim": live_after_trim,
        "capacity_rows_after_admission": capacity_after_admission,
        "live_rows_after_admission": live_after_admission,
        "released_rows_reused_pool_wide": True,
        "per_request_row_quota": None,
    }


@torch.no_grad()
def _serving_case_once(
    *,
    device: torch.device,
    seed: int,
    batch: int,
    warmup: int,
    iterations: int,
    pool_mode: str = "fixed",
    growth_quantum_steps: int = 1,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    prefix_length = 16
    total_decode = warmup + iterations
    maximum_sequence_length = prefix_length + total_decode
    slots = 256
    layout = "BBBBBBBA"
    decoder = make_decoder(
        maximum_sequence_length=maximum_sequence_length,
        layout=layout,
        width=128,
        heads=4,
        slots=slots,
    ).to(device=device, dtype=torch.bfloat16)
    prefix = torch.randint(4096, (batch, prefix_length), device=device)
    decode_tokens = torch.randint(4096, (batch, total_decode), device=device)
    if pool_mode not in ("fixed", "jit"):
        raise ValueError("pool mode must be fixed or jit")
    if growth_quantum_steps <= 0:
        raise ValueError("growth quantum steps must be positive")
    # The fixed control reserves a complete trace bound per bank.  The JIT
    # implementation instead rounds only the aggregate live rows in each
    # layer to a pool-wide growth quantum.  That quantum covers at least one
    # maximum W-way decode step; it is not divided into per-request quotas.
    maximum_rows_per_bank = 4 * maximum_sequence_length
    pool_rows_per_bank = math.ceil(0.70 * slots)
    if pool_rows_per_bank < maximum_rows_per_bank:
        raise AssertionError("configured serving pool cannot cover the trace")
    growth_quantum_rows = (
        max(16, batch * 4 * growth_quantum_steps)
        if pool_mode == "jit"
        else 0
    )
    per_layer_capacity = (
        None if pool_mode == "jit" else batch * pool_rows_per_bank
    )
    _, packed = decoder.prefill(
        prefix,
        capacity=maximum_sequence_length,
        sdm_capacity_rows=per_layer_capacity,
        sdm_growth_quantum_rows=growth_quantum_rows,
    )
    capacity_rows_after_prefill = sum(
        row.capacity_rows for row in packed.sdm_states
    )
    dense = dense_from_packed(packed)
    maximum_warmup_output_difference = 0.0
    maximum_warmup_state_difference = 0.0
    maximum_local_shadow_mixer_difference = 0.0
    maximum_local_shadow_state_difference = 0.0
    end_to_end_outputs_within_bf16_tolerance = True
    local_shadow_outputs_within_bf16_tolerance = True
    for position in range(warmup):
        token = decode_tokens[:, position : position + 1]
        packed_output, dense_output, execution_trace = paired_decode_trace(
            decoder,
            token,
            packed,
            dense,
        )
        output_difference = float((packed_output - dense_output).abs().max())
        maximum_warmup_output_difference = max(
            maximum_warmup_output_difference,
            output_difference,
        )
        state_differences = [
            float((packed_layer.materialize_dense() - dense_layer).abs().max())
            for packed_layer, dense_layer in zip(
                packed.sdm_states,
                dense.memories,
            )
        ]
        step_state_difference = max(state_differences, default=0.0)
        maximum_warmup_state_difference = max(
            maximum_warmup_state_difference,
            step_state_difference,
        )
        outputs_close = bfloat16_outputs_close(packed_output, dense_output)
        end_to_end_outputs_within_bf16_tolerance = (
            end_to_end_outputs_within_bf16_tolerance and outputs_close
        )
        sdm_trace = [row for row in execution_trace if row["kind"] == "B"]
        step_local_shadow_mixer_difference = max(
            (
                row["local_shadow_mixer_maximum_difference"]
                for row in sdm_trace
            ),
            default=0.0,
        )
        step_local_shadow_state_difference = max(
            (
                row["local_shadow_state_maximum_difference"]
                for row in sdm_trace
            ),
            default=0.0,
        )
        step_local_shadow_outputs_close = all(
            row["local_shadow_mixer_within_bf16_tolerance"]
            for row in sdm_trace
        )
        maximum_local_shadow_mixer_difference = max(
            maximum_local_shadow_mixer_difference,
            step_local_shadow_mixer_difference,
        )
        maximum_local_shadow_state_difference = max(
            maximum_local_shadow_state_difference,
            step_local_shadow_state_difference,
        )
        local_shadow_outputs_within_bf16_tolerance = (
            local_shadow_outputs_within_bf16_tolerance
            and step_local_shadow_outputs_close
        )
        overflow = any(
            int(row.overflow_flag.item()) != 0 for row in packed.sdm_states
        )
        semantic_failure = (
            not step_local_shadow_outputs_close
            or step_local_shadow_state_difference != 0.0
            or overflow
            or not math.isfinite(output_difference)
            or not math.isfinite(step_state_difference)
        )
        if semantic_failure:
            layer_diagnostics = []
            for layer, (packed_layer, dense_layer) in enumerate(
                zip(packed.sdm_states, dense.memories)
            ):
                layer_diagnostics.append(
                    {
                        "sdm_layer": layer,
                        "state_maximum_difference": state_differences[layer],
                        "allocated_rows": int(
                            packed_layer.allocated_rows_tensor()
                        ),
                        "capacity_rows": packed_layer.capacity_rows,
                        "free_rows": int(packed_layer.free_count.item()),
                        "growth_quantum_rows": (
                            packed_layer.growth_quantum_rows
                        ),
                        "proven_remaining_rows": (
                            packed_layer.proven_maximum_additional_allocations
                        ),
                        "overflow_flag": int(
                            packed_layer.overflow_flag.item()
                        ),
                    }
                )
            raise AssertionError(
                "warmup failed local packed/dense SDM semantics: "
                + json.dumps(
                    {
                        "pool_mode": pool_mode,
                        "batch": batch,
                        "warmup_position": position,
                        "output_maximum_difference": output_difference,
                        "outputs_within_bf16_tolerance": outputs_close,
                        "local_shadow_mixer_maximum_difference": (
                            step_local_shadow_mixer_difference
                        ),
                        "local_shadow_state_maximum_difference": (
                            step_local_shadow_state_difference
                        ),
                        "local_shadow_outputs_within_bf16_tolerance": (
                            step_local_shadow_outputs_close
                        ),
                        "bf16_output_relative_tolerance": BF16_OUTPUT_RTOL,
                        "bf16_output_absolute_tolerance": BF16_OUTPUT_ATOL,
                        "execution_trace": execution_trace,
                        "layers": layer_diagnostics,
                    },
                    sort_keys=True,
                )
            )
    torch.cuda.synchronize(device)

    capacity_rows_after_warmup = sum(
        row.capacity_rows for row in packed.sdm_states
    )
    growth_events_before_timing = sum(
        row.growth_events for row in packed.sdm_states
    )
    growth_rows_added_before_timing = sum(
        row.growth_rows_added for row in packed.sdm_states
    )
    growth_rows_copied_before_timing = sum(
        row.growth_rows_copied for row in packed.sdm_states
    )
    packed_ms = elapsed_ms(
        lambda: decoder.decode(
            decode_tokens[:, packed.position - prefix_length : packed.position - prefix_length + 1],
            packed,
        ),
        iterations=iterations,
    )
    if packed.position != prefix_length + total_decode:
        raise AssertionError("packed timing did not consume the complete trace")
    for row in packed.sdm_states:
        row.validate_invariants()

    # Recreate the identical prefix because the first dense state was consumed
    # during paired warmup while packed continued through the timed section.
    _, dense_source = decoder.prefill(
        prefix,
        capacity=maximum_sequence_length,
        sdm_capacity_rows=per_layer_capacity,
        sdm_growth_quantum_rows=growth_quantum_rows,
    )
    dense_timed = dense_from_packed(dense_source)
    for position in range(warmup):
        dense_decode(
            decoder,
            decode_tokens[:, position : position + 1],
            dense_timed,
        )
    dense_ms = elapsed_ms(
        lambda: dense_decode(
            decoder,
            decode_tokens[
                :,
                dense_timed.position - prefix_length : dense_timed.position - prefix_length + 1,
            ],
            dense_timed,
        ),
        iterations=iterations,
    )
    if dense_timed.position != prefix_length + total_decode:
        raise AssertionError("dense timing did not consume the complete trace")

    legacy_packed = None
    legacy_packed_ms = None
    if pool_mode == "fixed":
        # Re-run the exact packed path with the earlier two-kernel allocator
        # and per-step overflow assertion.  This matched-run control attributes
        # speed movement to the narrow proven-capacity fusion.
        _, legacy_packed = decoder.prefill(
            prefix,
            capacity=maximum_sequence_length,
            sdm_capacity_rows=per_layer_capacity,
        )
        for row in legacy_packed.sdm_states:
            row.capacity_is_proven = False
        for position in range(warmup):
            decoder.decode(
                decode_tokens[:, position : position + 1],
                legacy_packed,
            )
        legacy_packed_ms = elapsed_ms(
            lambda: decoder.decode(
                decode_tokens[
                    :,
                    legacy_packed.position
                    - prefix_length : legacy_packed.position
                    - prefix_length
                    + 1,
                ],
                legacy_packed,
            ),
            iterations=iterations,
        )
        if legacy_packed.position != prefix_length + total_decode:
            raise AssertionError("legacy packed timing did not consume the trace")
        for row in legacy_packed.sdm_states:
            row.validate_invariants()
    packed_storage = packed.storage_bytes()
    dense_storage = dense_storage_bytes(dense_timed)
    allocated_rows = sum(
        int(row.allocated_rows_tensor()) for row in packed.sdm_states
    )
    capacity_rows = sum(row.capacity_rows for row in packed.sdm_states)
    total_growth_events = sum(row.growth_events for row in packed.sdm_states)
    total_growth_rows_added = sum(
        row.growth_rows_added for row in packed.sdm_states
    )
    total_growth_rows_copied = sum(
        row.growth_rows_copied for row in packed.sdm_states
    )
    logical_rows = sum(row.banks * row.slots for row in packed.sdm_states)
    result = {
        "batch": batch,
        "pool_mode": pool_mode,
        "growth_quantum_steps": (
            growth_quantum_steps if pool_mode == "jit" else None
        ),
        "layout": layout,
        "width": 128,
        "logical_slots_per_layer": slots,
        "reads": 16,
        "writes": 4,
        "prefix_length": prefix_length,
        "decode_steps": total_decode,
        "pool_rows_per_bank": (
            pool_rows_per_bank if pool_mode == "fixed" else None
        ),
        "growth_quantum_rows_per_layer": growth_quantum_rows,
        "capacity_rows_after_prefill": capacity_rows_after_prefill,
        "capacity_rows_after_warmup": capacity_rows_after_warmup,
        "total_growth_events": total_growth_events,
        "total_growth_rows_added": total_growth_rows_added,
        "total_growth_rows_copied": total_growth_rows_copied,
        "timed_growth_events": total_growth_events - growth_events_before_timing,
        "timed_growth_rows_added": (
            total_growth_rows_added - growth_rows_added_before_timing
        ),
        "timed_growth_rows_copied": (
            total_growth_rows_copied - growth_rows_copied_before_timing
        ),
        "maximum_trace_rows_per_bank": maximum_rows_per_bank,
        "allocated_rows_after_trace": allocated_rows,
        "capacity_rows": capacity_rows,
        "logical_rows": logical_rows,
        "live_row_fraction": allocated_rows / logical_rows,
        "physical_capacity_fraction": capacity_rows / logical_rows,
        "warmup_dense_packed_output_maximum_difference": (
            maximum_warmup_output_difference
        ),
        "warmup_dense_packed_outputs_within_bf16_tolerance": (
            end_to_end_outputs_within_bf16_tolerance
        ),
        "warmup_dense_packed_state_maximum_difference": (
            maximum_warmup_state_difference
        ),
        "warmup_local_shadow_mixer_maximum_difference": (
            maximum_local_shadow_mixer_difference
        ),
        "warmup_local_shadow_outputs_within_bf16_tolerance": (
            local_shadow_outputs_within_bf16_tolerance
        ),
        "warmup_local_shadow_state_maximum_difference": (
            maximum_local_shadow_state_difference
        ),
        "bf16_output_relative_tolerance": BF16_OUTPUT_RTOL,
        "bf16_output_absolute_tolerance": BF16_OUTPUT_ATOL,
        "packed_storage": packed_storage,
        "dense_storage": dense_storage,
        "packed_over_dense_live_bytes": (
            packed_storage["per_user_live_total_excluding_shared"]
            / dense_storage["live_total"]
        ),
        "packed_over_dense_reserved_bytes": (
            packed_storage["total_reserved_excluding_shared"]
            / dense_storage["allocated_total"]
        ),
        "packed_milliseconds_per_step": packed_ms,
        "legacy_packed_milliseconds_per_step": legacy_packed_ms,
        "dense_milliseconds_per_step": dense_ms,
        "packed_over_dense_decode_latency": packed_ms / dense_ms,
        "legacy_packed_over_dense_decode_latency": (
            None if legacy_packed_ms is None else legacy_packed_ms / dense_ms
        ),
        "packed_over_legacy_packed_decode_latency": (
            None
            if legacy_packed_ms is None
            else packed_ms / legacy_packed_ms
        ),
        "packed_tokens_per_second": batch * 1000.0 / packed_ms,
        "legacy_packed_tokens_per_second": (
            None
            if legacy_packed_ms is None
            else batch * 1000.0 / legacy_packed_ms
        ),
        "dense_tokens_per_second": batch * 1000.0 / dense_ms,
        "packed_backend": (
            "copy-on-write proven-capacity fused narrow Triton"
            if pool_mode == "jit"
            else "proven-capacity fused narrow Triton"
        ),
        "legacy_packed_backend": (
            "separate allocation resolution, overflow assertion, and update/read"
        ),
    }
    del decoder, packed, dense, dense_source, dense_timed, legacy_packed
    gc.collect()
    torch.cuda.empty_cache()
    return result


def serving_case(
    *,
    device: torch.device,
    seed: int,
    batch: int,
    warmup: int,
    iterations: int,
    pool_mode: str = "fixed",
    growth_quantum_steps: int = 1,
    timing_repetitions: int = 5,
) -> dict[str, Any]:
    """Report medians over repeated identical traces on one accelerator."""

    if timing_repetitions <= 0:
        raise ValueError("timing repetitions must be positive")
    trials = [
        _serving_case_once(
            device=device,
            seed=seed,
            batch=batch,
            warmup=warmup,
            iterations=iterations,
            pool_mode=pool_mode,
            growth_quantum_steps=growth_quantum_steps,
        )
        for _ in range(timing_repetitions)
    ]
    invariant_keys = (
        "allocated_rows_after_trace",
        "capacity_rows",
        "logical_rows",
        "total_growth_events",
        "total_growth_rows_added",
        "total_growth_rows_copied",
        "packed_storage",
        "dense_storage",
    )
    for key in invariant_keys:
        if any(trial[key] != trials[0][key] for trial in trials[1:]):
            raise AssertionError(f"serving timing trials changed {key}")

    result = dict(trials[0])
    packed_times = [trial["packed_milliseconds_per_step"] for trial in trials]
    dense_times = [trial["dense_milliseconds_per_step"] for trial in trials]
    packed_ms = statistics.median(packed_times)
    dense_ms = statistics.median(dense_times)
    result["timing_repetitions"] = timing_repetitions
    result["packed_timing_trials_milliseconds"] = packed_times
    result["dense_timing_trials_milliseconds"] = dense_times
    result["packed_milliseconds_per_step"] = packed_ms
    result["dense_milliseconds_per_step"] = dense_ms
    result["packed_over_dense_decode_latency"] = packed_ms / dense_ms
    result["packed_tokens_per_second"] = batch * 1000.0 / packed_ms
    result["dense_tokens_per_second"] = batch * 1000.0 / dense_ms
    legacy_times = [
        trial["legacy_packed_milliseconds_per_step"] for trial in trials
    ]
    if all(value is not None for value in legacy_times):
        legacy_ms = statistics.median(legacy_times)
        result["legacy_packed_timing_trials_milliseconds"] = legacy_times
        result["legacy_packed_milliseconds_per_step"] = legacy_ms
        result["legacy_packed_over_dense_decode_latency"] = legacy_ms / dense_ms
        result["packed_over_legacy_packed_decode_latency"] = packed_ms / legacy_ms
        result["legacy_packed_tokens_per_second"] = batch * 1000.0 / legacy_ms
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--timing-repetitions", type=int, default=5)
    parser.add_argument("--growth-quantum-steps", default="1,2,4")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("faithful SDM serving benchmark requires CUDA")
    if args.warmup < 1 or args.iterations < 1 or args.timing_repetitions < 1:
        parser.error("warmup, iterations, and timing repetitions must be positive")
    try:
        growth_quantum_steps = tuple(
            sorted(
                {
                    int(value)
                    for value in args.growth_quantum_steps.split(",")
                    if value.strip()
                }
            )
        )
    except ValueError as error:
        parser.error(f"invalid growth quantum steps: {error}")
    if not growth_quantum_steps or growth_quantum_steps[0] <= 0:
        parser.error("growth quantum steps must be positive integers")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    parity, causality = correctness(device=device, seed=args.seed + 1)
    lifecycle = continuous_batch_lifecycle(device=device, seed=args.seed + 2)
    atomic_json(args.output / "PARITY.json", parity)
    atomic_json(args.output / "CAUSALITY.json", causality)
    atomic_json(
        args.output / "HEALTHY.json",
        {
            "schema": "faithful-sdm-packed-serving-health-v1",
            "phase": "first exact fused CUDA controller/state update",
            "finite": True,
            "dense_packed_output_maximum_difference": parity[
                "dense_packed_output_maximum_difference"
            ],
            "dense_packed_outputs_within_bf16_tolerance": parity[
                "dense_packed_outputs_within_bf16_tolerance"
            ],
            "bf16_output_relative_tolerance": BF16_OUTPUT_RTOL,
            "bf16_output_absolute_tolerance": BF16_OUTPUT_ATOL,
            "dense_packed_state_maximum_difference": parity[
                "dense_packed_state_maximum_difference"
            ],
            "maximum_prefix_difference": causality["maximum_prefix_difference"],
        },
    )
    fixed_cases = [
        serving_case(
            device=device,
            seed=args.seed + 100 + index,
            batch=batch,
            warmup=args.warmup,
            iterations=args.iterations,
            pool_mode="fixed",
            timing_repetitions=args.timing_repetitions,
        )
        for index, batch in enumerate((1, 16, 64))
    ]
    jit_cases = []
    for quantum_steps in growth_quantum_steps:
        for index, batch in enumerate((1, 16, 64)):
            jit_cases.append(
                serving_case(
                    device=device,
                    seed=args.seed + 200 + index,
                    batch=batch,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    pool_mode="jit",
                    growth_quantum_steps=quantum_steps,
                    timing_repetitions=args.timing_repetitions,
                )
            )
    payload = {
        "schema": "faithful-sdm-packed-serving-v2",
        "seed": args.seed,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "controller": {
            "model": "faithful independent SDM",
            "addressing": "product-key",
            "mutation": "W4 gated delta",
            "value": "projected",
            "output_gate": "channelwise",
            "output": "projected",
            "learned_initial_memory": "model-shared M0",
            "mutable_state_dtype": "float32",
        },
        "parity": parity,
        "causality": causality,
        "continuous_batch_lifecycle": lifecycle,
        "serving_cases": fixed_cases,
        "jit_serving_cases": jit_cases,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(args.output / "result.json", payload)


if __name__ == "__main__":
    main()
