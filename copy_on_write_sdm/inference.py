"""Exact packed prefill and decode for faithful independent SDM layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn.functional as F

from .sdm import SDMRouting, SparseDeltaMemory, product_key_routes
from .allocator import (
    PackedSDMCopyOnWriteState,
    dense_sdm_sparse_step,
    packed_sdm_cow_step,
)

if TYPE_CHECKING:
    from .model import SDMDecoderStack


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


@dataclass
class FaithfulSDMDecodeState:
    """Retained state for one equal-length batch of independent SDM requests.

    Every B layer owns its faithful SDM memory.  Its learned initial table is
    shared model state; only prefix-touched rows live in the private FP32 value
    slabs.  Dense-attention layers retain their ordinary fixed-capacity K/V
    caches.
    """

    layout: str
    position: int
    capacity: int
    activation_dtype: torch.dtype
    active_sequences: torch.Tensor
    sdm_layer_positions: tuple[int, ...]
    sdm_states: tuple[PackedSDMCopyOnWriteState, ...]
    dense_keys: tuple[torch.Tensor, ...]
    dense_values: tuple[torch.Tensor, ...]
    positions: torch.Tensor | None = None

    @property
    def batch(self) -> int:
        return self.active_sequences.numel()

    @property
    def device(self) -> torch.device:
        return self.active_sequences.device

    @property
    def dtype(self) -> torch.dtype:
        return self.activation_dtype

    def current_positions(self) -> torch.Tensor:
        """Return one decode position per live request."""

        if self.positions is None:
            return torch.full(
                (self.batch,),
                self.position,
                device=self.device,
                dtype=torch.int64,
            )
        return self.positions

    @torch.no_grad()
    def release(
        self,
        sequence_indices: torch.Tensor | Sequence[int] | None = None,
    ) -> int:
        """Release selected completed requests from every SDM layer.

        Released rows return to each layer's shared slab.  A released request
        cannot be decoded again from this fixed batch state; a continuous
        batching runtime may reuse the returned rows for a later request.
        """

        if sequence_indices is None:
            selected = self.active_sequences.nonzero().flatten()
        else:
            selected = torch.as_tensor(
                sequence_indices,
                device=self.device,
                dtype=torch.int64,
            )
        if selected.ndim != 1:
            raise ValueError("released sequence indices must be a vector")
        if selected.numel() == 0:
            return 0
        ordered = selected.sort().values
        if (
            bool((selected < 0).any())
            or bool((selected >= self.batch).any())
            or (ordered.numel() > 1 and bool((ordered[1:] == ordered[:-1]).any()))
            or not bool(self.active_sequences[selected].all())
        ):
            raise ValueError("released sequence is invalid, duplicated, or inactive")
        released = 0
        for state in self.sdm_states:
            if state.banks % self.batch:
                raise AssertionError("SDM serving banks do not divide by batch")
            heads = state.banks // self.batch
            banks = (
                selected[:, None] * heads
                + torch.arange(heads, device=self.device)[None, :]
            ).flatten()
            released += int(state.release(banks))
        self.active_sequences[selected] = False
        return released

    @torch.no_grad()
    def compact(self) -> torch.Tensor:
        """Remove released request slots while preserving every live row.

        Physical SDM rows stay in the same shared slab.  Only bank metadata and
        dense K/V batch rows are compacted, so surviving requests can continue
        immediately and released rows remain available to later admissions.
        The returned vector maps compacted rows to their former indices.
        """

        previous_batch = self.batch
        retained = self.active_sequences.nonzero().flatten()
        positions = self.current_positions()
        for state in self.sdm_states:
            if previous_batch == 0:
                if state.banks != 0:
                    raise AssertionError("empty serving batch retained SDM banks")
                continue
            if state.banks % previous_batch:
                raise AssertionError("SDM serving banks do not divide by batch")
            heads = state.banks // previous_batch
            banks = (
                retained[:, None] * heads
                + torch.arange(heads, device=self.device)[None, :]
            ).flatten()
            state.retain_banks(banks)
        self.dense_keys = tuple(row[retained].contiguous() for row in self.dense_keys)
        self.dense_values = tuple(
            row[retained].contiguous() for row in self.dense_values
        )
        retained_positions = positions[retained].contiguous()
        self.positions = (
            retained_positions if self.positions is not None else None
        )
        self.active_sequences = torch.ones(
            retained.numel(),
            device=self.device,
            dtype=torch.bool,
        )
        if retained.numel():
            self.position = int(retained_positions.max().item())
        else:
            self.position = 0
        return retained

    @torch.no_grad()
    def admit(self, incoming: "FaithfulSDMDecodeState") -> int:
        """Admit a prefetched cohort into the same aggregate SDM row pools."""

        self.compact()
        incoming.compact()
        if (
            self.layout != incoming.layout
            or self.capacity != incoming.capacity
            or self.activation_dtype != incoming.activation_dtype
            or self.device != incoming.device
            or self.sdm_layer_positions != incoming.sdm_layer_positions
            or len(self.sdm_states) != len(incoming.sdm_states)
            or len(self.dense_keys) != len(incoming.dense_keys)
        ):
            raise ValueError("admitted faithful-SDM state is incompatible")
        admitted = incoming.batch
        for destination, source in zip(self.sdm_states, incoming.sdm_states):
            destination.append_banks_from(source)
        self.dense_keys = tuple(
            torch.cat((current, added), dim=0).contiguous()
            for current, added in zip(self.dense_keys, incoming.dense_keys)
        )
        self.dense_values = tuple(
            torch.cat((current, added), dim=0).contiguous()
            for current, added in zip(self.dense_values, incoming.dense_values)
        )
        admitted_positions = torch.cat(
            (self.current_positions(), incoming.current_positions()),
            dim=0,
        ).contiguous()
        if admitted_positions.numel() and bool(
            (admitted_positions == admitted_positions[0]).all()
        ):
            self.position = int(admitted_positions[0].item())
            self.positions = None
        else:
            self.positions = admitted_positions
        self.active_sequences = torch.ones(
            admitted_positions.numel(),
            device=self.device,
            dtype=torch.bool,
        )
        if admitted_positions.numel() and self.positions is not None:
            self.position = int(admitted_positions.max().item())
        return admitted

    @torch.no_grad()
    def trim_sdm_pools(self, *, headroom_rows: int = 0) -> int:
        """Return unused physical SDM pages after a lifecycle transition."""

        return sum(
            state.trim_capacity(headroom_rows=headroom_rows)
            for state in self.sdm_states
        )

    @torch.no_grad()
    def release_and_compact(
        self,
        sequence_indices: torch.Tensor | Sequence[int] | None = None,
        *,
        trim: bool = False,
        trim_headroom_rows: int = 0,
    ) -> int:
        """Release requests, compact survivors, and optionally shrink slabs."""

        released = self.release(sequence_indices)
        self.compact()
        if trim:
            self.trim_sdm_pools(headroom_rows=trim_headroom_rows)
        return released

    def storage_bytes(self) -> dict[str, int]:
        """Separate private live/reserved state, shared M0, and dense K/V."""

        private_live_values = 0
        private_metadata = 0
        private_reserved = 0
        shared_initial = 0
        logical_dense = 0
        for state in self.sdm_states:
            used = state.used_bytes()
            stored = state.storage_bytes()
            private_live_values += used["live_value_bytes"]
            private_metadata += used["allocator_metadata_bytes"]
            private_reserved += stored["private_allocator_reserved_total"]
            shared_initial += stored["shared_initial_memory"]
            logical_dense += state.banks * state.slots * state.width * 4
        dense_allocated = sum(
            _tensor_bytes(tensor)
            for tensor in (*self.dense_keys, *self.dense_values)
        )
        dense_live = sum(
            tensor[:, :, : self.position].numel() * tensor.element_size()
            for tensor in (*self.dense_keys, *self.dense_values)
        )
        lifecycle = _tensor_bytes(self.active_sequences) + (
            0 if self.positions is None else _tensor_bytes(self.positions)
        )
        return {
            "sdm_private_value_live": private_live_values,
            "sdm_private_allocator_metadata": private_metadata,
            "sdm_private_live_total": private_live_values + private_metadata,
            "sdm_private_reserved_total": private_reserved,
            "sdm_logical_dense_fp32": logical_dense,
            "sdm_shared_initial_memory": shared_initial,
            "dense_kv_live": dense_live,
            "dense_kv_allocated": dense_allocated,
            "lifecycle": lifecycle,
            "per_user_live_total_excluding_shared": (
                private_live_values + private_metadata + dense_live + lifecycle
            ),
            "total_reserved_excluding_shared": (
                private_reserved + dense_allocated + lifecycle
            ),
        }


@dataclass(frozen=True)
class _FaithfulSDMStep:
    """Position-private controller outputs shared by dense and packed state."""

    normalized: torch.Tensor
    write_indices: torch.Tensor
    write_weights: torch.Tensor
    values: torch.Tensor
    input_gate: torch.Tensor
    forget_log_gate: torch.Tensor
    read_indices: torch.Tensor
    read_weights: torch.Tensor


def _allocate_dense_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, time, head_width = key.shape
    if value.shape != key.shape or capacity < time:
        raise ValueError("dense prefill cache capacity is invalid")
    shape = (batch, heads, capacity, head_width)
    key_cache = torch.empty(shape, device=key.device, dtype=key.dtype)
    value_cache = torch.empty(shape, device=value.device, dtype=value.dtype)
    key_cache[:, :, :time].copy_(key)
    value_cache[:, :, :time].copy_(value)
    return key_cache, value_cache


def _faithful_controller_guard(module: SparseDeltaMemory) -> None:
    if not isinstance(module, SparseDeltaMemory):
        raise TypeError("packed serving requires the native SDM controller")


def _pack_prefill_layer(
    module: SparseDeltaMemory,
    routing: SDMRouting,
    *,
    capacity_rows: int | None,
    decode_capacity: int,
    growth_quantum_rows: int,
) -> PackedSDMCopyOnWriteState:
    if routing.final_memory is None:
        raise AssertionError("SDM prefill did not return terminal memory")
    batch, time, heads, writes = routing.write_indices.shape
    if heads != module.memory_heads or routing.final_memory.shape != (
        batch,
        heads,
        module.slots,
        module.head_width,
    ):
        raise ValueError("SDM prefill routing does not match its controller")
    active = torch.zeros(
        batch,
        heads,
        module.slots,
        device=routing.write_indices.device,
        dtype=torch.bool,
    )
    active.scatter_(
        2,
        routing.write_indices.permute(0, 2, 1, 3).reshape(
            batch,
            heads,
            time * writes,
        ),
        True,
    )
    templates = torch.arange(
        heads,
        device=active.device,
        dtype=torch.int64,
    ).repeat(batch)
    return PackedSDMCopyOnWriteState.from_dense_overlay(
        module.initial_memory,
        routing.final_memory.flatten(0, 1),
        active.flatten(0, 1),
        template_indices=templates,
        capacity_rows=capacity_rows,
        state_dtype=torch.float32,
        maximum_future_allocations=(
            None
            if growth_quantum_rows
            else batch * heads * writes * (decode_capacity - time)
        ),
        growth_quantum_rows=growth_quantum_rows,
    )


@torch.no_grad()
def prefill_faithful_sdm_stack(
    stack: "SDMDecoderStack",
    tokens: torch.Tensor,
    *,
    capacity: int | None = None,
    sdm_capacity_rows: int | Sequence[int] | None = None,
    sdm_growth_quantum_rows: int = 0,
) -> tuple[torch.Tensor, FaithfulSDMDecodeState]:
    """Execute a full prefix and hand every independent SDM layer to COW state."""

    if tokens.ndim != 3 or tokens.shape[-1] != stack.width:
        raise ValueError(f"tokens must be [B,T,{stack.width}]")
    if stack.training:
        raise ValueError("faithful SDM prefill requires stack.eval()")
    batch, time, _ = tokens.shape
    if time <= 0:
        raise ValueError("prefill requires at least one token")
    selected_capacity = time if capacity is None else capacity
    if selected_capacity < time:
        raise ValueError("prefill capacity is shorter than the prefix")
    if sdm_growth_quantum_rows < 0:
        raise ValueError("SDM growth quantum cannot be negative")
    layer_positions = tuple(
        layer for layer, kind in enumerate(stack.layout) if kind == "B"
    )
    if isinstance(sdm_capacity_rows, Sequence) and not isinstance(
        sdm_capacity_rows, (str, bytes)
    ):
        per_layer_capacity = tuple(int(value) for value in sdm_capacity_rows)
        if len(per_layer_capacity) != len(layer_positions):
            raise ValueError("SDM capacities must contain one value per B layer")
    else:
        per_layer_capacity = (sdm_capacity_rows,) * len(layer_positions)

    dense_keys: list[torch.Tensor] = []
    dense_values: list[torch.Tensor] = []
    sdm_states: list[PackedSDMCopyOnWriteState] = []
    bus_index = 0
    for physical_layer, kind in enumerate(stack.layout):
        if kind == "A":
            block = stack.attention_layers[str(physical_layer)]
            attended, key, value = block.attention.prefill_with_cache(tokens)
            tokens = tokens + attended
            tokens = tokens + block.mlp(tokens)
            key_cache, value_cache = _allocate_dense_cache(
                key,
                value,
                selected_capacity,
            )
            dense_keys.append(key_cache)
            dense_values.append(value_cache)
            continue
        module = stack.sdm_layers[str(physical_layer)]
        _faithful_controller_guard(module)
        mixed, routing = module(
            tokens,
            return_routing=True,
            include_final_memory=True,
        )
        tokens = tokens + mixed
        tokens = tokens + stack.sdm_mlps[str(physical_layer)](tokens)
        sdm_states.append(
            _pack_prefill_layer(
                module,
                routing,
                capacity_rows=per_layer_capacity[bus_index],
                decode_capacity=selected_capacity,
                growth_quantum_rows=sdm_growth_quantum_rows,
            )
        )
        bus_index += 1

    return tokens, FaithfulSDMDecodeState(
        layout=stack.layout,
        position=time,
        capacity=selected_capacity,
        activation_dtype=tokens.dtype,
        active_sequences=torch.ones(
            batch,
            device=tokens.device,
            dtype=torch.bool,
        ),
        sdm_layer_positions=layer_positions,
        sdm_states=tuple(sdm_states),
        dense_keys=tuple(dense_keys),
        dense_values=tuple(dense_values),
    )


def _prepare_faithful_sdm_step(
    module: SparseDeltaMemory,
    tokens: torch.Tensor,
) -> _FaithfulSDMStep:
    """Generate one faithful SDM edit and read program position-parallel."""

    _faithful_controller_guard(module)
    batch = tokens.shape[0]
    if tokens.shape != (batch, 1, module.width):
        raise ValueError("faithful SDM decode requires one token")
    normalized = module.input_norm(tokens)
    raw_reads = module.read_projection(normalized).reshape(
        batch,
        1,
        module.memory_heads,
        module.key_width_per_head,
    )
    raw_writes = module.write_projection(normalized).reshape_as(raw_reads)
    routed_reads = raw_reads.permute(0, 2, 1, 3).flatten(0, 1)
    routed_writes = raw_writes.permute(0, 2, 1, 3).flatten(0, 1)
    read_weights, read_indices = product_key_routes(
        routed_reads,
        slots=module.slots,
        selected=module.reads,
    )
    write_weights, write_indices = product_key_routes(
        routed_writes,
        slots=module.slots,
        selected=module.writes,
    )
    value_source = module.value_projection(normalized)
    values = (
        value_source.reshape(
            batch,
            1,
            module.memory_heads,
            module.head_width,
        )
        .permute(0, 2, 1, 3)
        .flatten(0, 1)[:, 0]
        .contiguous()
    )
    forget = (
        -torch.exp(module.A_log)
        * F.softplus(module.forget_projection(normalized) + module.dt_bias)
    ).permute(0, 2, 1).flatten()
    input_gate = (
        torch.sigmoid(module.input_projection(normalized))
        .permute(0, 2, 1)
        .flatten()
    )
    return _FaithfulSDMStep(
        normalized=normalized,
        write_indices=write_indices[:, 0],
        write_weights=write_weights[:, 0],
        values=values,
        input_gate=input_gate,
        forget_log_gate=forget,
        read_indices=read_indices[:, 0],
        read_weights=read_weights[:, 0],
    )


def _finish_faithful_sdm_step(
    module: SparseDeltaMemory,
    step: _FaithfulSDMStep,
    reading: torch.Tensor,
) -> torch.Tensor:
    batch = step.normalized.shape[0]
    reading = module.readings_norm(reading)
    reading = reading.reshape(batch, module.memory_heads, module.head_width).reshape(
        batch,
        1,
        module.width,
    )
    gate = torch.sigmoid(module.output_gate(step.normalized))
    return module.output_projection(gate * reading)


@torch.no_grad()
def _decode_faithful_sdm_layer(
    module: SparseDeltaMemory,
    tokens: torch.Tensor,
    state: PackedSDMCopyOnWriteState,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """Apply the released faithful controller to one token and packed state."""

    step = _prepare_faithful_sdm_step(module, tokens)
    if state.banks != tokens.shape[0] * module.memory_heads:
        raise ValueError("packed SDM banks do not match batch and memory heads")
    if state.growth_quantum_rows:
        state.prepare_step_capacity(state.banks * module.writes)
    reading, _ = packed_sdm_cow_step(
        state,
        step.write_indices,
        step.write_weights,
        step.values,
        step.input_gate,
        step.forget_log_gate,
        step.read_indices,
        step.read_weights,
        backend=backend,
        validate_routes=False,
        collect_diagnostics=False,
    )
    return _finish_faithful_sdm_step(module, step, reading)


@torch.no_grad()
def _decode_faithful_sdm_layer_dense(
    module: SparseDeltaMemory,
    tokens: torch.Tensor,
    memory: torch.Tensor,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """Dense-table serving control with the identical faithful controller."""

    step = _prepare_faithful_sdm_step(module, tokens)
    if memory.shape != (
        tokens.shape[0] * module.memory_heads,
        module.slots,
        module.head_width,
    ):
        raise ValueError("dense SDM memory does not match batch and controller")
    reading = dense_sdm_sparse_step(
        memory,
        step.write_indices,
        step.write_weights,
        step.values,
        step.input_gate,
        step.forget_log_gate,
        step.read_indices,
        step.read_weights,
        backend=backend,
        validate_routes=False,
    )
    return _finish_faithful_sdm_step(module, step, reading)


@torch.no_grad()
def decode_faithful_sdm_stack(
    stack: "SDMDecoderStack",
    tokens: torch.Tensor,
    state: FaithfulSDMDecodeState,
) -> torch.Tensor:
    """Consume one token through dense K/V and packed faithful SDM state."""

    if tokens.shape != (state.batch, 1, stack.width):
        raise ValueError(f"decode tokens must be [{state.batch},1,{stack.width}]")
    if stack.training:
        raise ValueError("faithful SDM decode requires stack.eval()")
    if state.layout != stack.layout:
        raise ValueError("decode state layout does not match the stack")
    if tokens.device != state.device or tokens.dtype != state.dtype:
        raise ValueError("decode tokens do not match state device and dtype")
    positions = state.current_positions()
    if state.positions is None:
        if state.position >= state.capacity:
            raise ValueError("decode state has reached its fixed capacity")
        attention_position: int | torch.Tensor = state.position
    else:
        if bool((positions >= state.capacity).any()):
            raise ValueError("decode state has reached its fixed capacity")
        attention_position = positions
    if not bool(state.active_sequences.all()):
        raise ValueError("released requests must be removed before further decode")
    expected_sdm = stack.layout.count("B")
    expected_dense = stack.layout.count("A")
    if (
        len(state.sdm_states) != expected_sdm
        or len(state.sdm_layer_positions) != expected_sdm
        or len(state.dense_keys) != expected_dense
        or len(state.dense_values) != expected_dense
    ):
        raise ValueError("decode state does not match the stack")

    dense_index = 0
    sdm_index = 0
    for physical_layer, kind in enumerate(stack.layout):
        if kind == "A":
            block = stack.attention_layers[str(physical_layer)]
            attended = block.attention.decode_with_cache(
                tokens,
                state.dense_keys[dense_index],
                state.dense_values[dense_index],
                position=attention_position,
            )
            tokens = tokens + attended
            tokens = tokens + block.mlp(tokens)
            dense_index += 1
            continue
        if state.sdm_layer_positions[sdm_index] != physical_layer:
            raise ValueError("SDM decode layer ordering is inconsistent")
        mixed = _decode_faithful_sdm_layer(
            stack.sdm_layers[str(physical_layer)],
            tokens,
            state.sdm_states[sdm_index],
        )
        tokens = tokens + mixed
        tokens = tokens + stack.sdm_mlps[str(physical_layer)](tokens)
        sdm_index += 1
    if state.positions is None:
        state.position += 1
    else:
        positions.add_(1)
        state.positions = positions
        state.position = int(positions.max().item())
    return tokens
