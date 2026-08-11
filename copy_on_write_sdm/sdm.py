"""Small-workspace Sparse Delta Memory with exact causal recurrence.

The controller in this module follows the released SDM semantics while using a
compact, dependency-free recurrent kernel for the deliberately small memory
banks studied here. Every physical SDM layer owns learned initial memory and
applies the released gated-delta rule.

Upstream semantic reference:
https://github.com/facebookresearch/sparse-delta-memory/tree/
183e7df809131b80ad4393741029d0f20fc3640b
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only development hosts
    triton = None
    tl = None



@dataclass(frozen=True)
class SDMRouting:
    """Sparse controller diagnostics for one SDM physical layer."""

    read_indices: torch.Tensor
    read_weights: torch.Tensor
    write_indices: torch.Tensor
    write_weights: torch.Tensor
    forget_log_gate: torch.Tensor
    erase_gate: torch.Tensor
    input_gate: torch.Tensor
    # The realized controller value is optional so lightweight synthetic
    # accounting fixtures do not need to manufacture it.  Production SDM
    # forwards populate it, allowing the inference allocator to replay exact
    # trained routes and mutations without a second controller formulation.
    values: torch.Tensor | None = None
    # Full-prefix serving handoff may request the exact FP32 recurrent state.
    # Ordinary training forwards leave this absent so diagnostics do not retain
    # an additional reference to the terminal memory.
    final_memory: torch.Tensor | None = None


@dataclass(frozen=True)
class SDMCopyOnWriteAccounting:
    """Vectorized physical-state accounting for one faithful SDM layer.

    The SDM forward pass still executes its ordinary dense logical table.  This
    object measures the exactly equivalent copy-on-write representation: every
    head owns one logical address space, and a slot becomes private on its
    first hard write.  Only ``straight_through_fraction`` carries gradient.
    """

    hard_unique_by_position: torch.Tensor
    first_touch_count_by_position: torch.Tensor
    repeated_write_count_by_position: torch.Tensor
    private_read_count_by_position: torch.Tensor
    private_read_weight_by_position: torch.Tensor
    write_route_entropy_by_position: torch.Tensor
    slot_write_counts: torch.Tensor
    hard_final_fraction: torch.Tensor
    soft_final_fraction: torch.Tensor
    straight_through_fraction: torch.Tensor


def dense_recurrence_block_width(slots: int) -> int:
    """Choose a bounded-register width tile for the exact dense trainer kernel.

    The kernel keeps one ``slots x BLOCK_D`` tile live while it walks the
    sequence. Shrinking the width tile as the logical table grows keeps the
    resident tile at no more than 4,096 elements through N=1,024. This is a
    capability-training path; the sparse serving allocator remains separate.
    """

    if slots <= 0:
        raise ValueError("slots must be positive")
    if slots <= 256:
        return 16
    if slots <= 512:
        return 8
    if slots <= 1_024:
        return 4
    raise ValueError("dense SDM training recurrence supports at most 1,024 slots")


def product_key_routes(
    projected: torch.Tensor,
    *,
    slots: int,
    selected: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Released product-key selection followed by selected-score softmax.

    ``projected`` is one product-key head with shape ``[B,T,2*sqrt(N)]``.
    The two codebooks are shortlisted independently, their scores are added,
    and the best requested Cartesian products are retained.  Selected slots
    are sorted by index exactly as in the upstream training path.
    """

    if projected.ndim != 3:
        raise ValueError("product-key projection must be [B,T,2*sqrt(N)]")
    root = math.isqrt(slots)
    if root * root != slots or projected.shape[-1] != 2 * root:
        raise ValueError("slot count and product-key projection do not match")
    if not 1 <= selected <= slots:
        raise ValueError("selected product keys must be between one and slots")
    first, second = projected.chunk(2, dim=-1)
    subselected = min(selected, root)
    first_values, first_indices = torch.topk(
        first,
        k=subselected,
        dim=-1,
    )
    second_values, second_indices = torch.topk(
        second,
        k=subselected,
        dim=-1,
    )
    pair_scores = (first_values.unsqueeze(-1) + second_values.unsqueeze(-2)).flatten(-2)
    pair_first = (
        first_indices.unsqueeze(-1)
        .expand(
            *first_indices.shape,
            subselected,
        )
        .flatten(-2)
    )
    pair_second = (
        second_indices.unsqueeze(-2)
        .expand(
            *second_indices.shape[:-1],
            subselected,
            subselected,
        )
        .flatten(-2)
    )
    final_selected = min(selected, pair_scores.shape[-1])
    values, positions = torch.topk(
        pair_scores,
        k=final_selected,
        dim=-1,
    )
    indices = torch.gather(pair_first, -1, positions) * root + torch.gather(
        pair_second, -1, positions
    )
    indices, order = indices.sort(dim=-1)
    values = values.gather(-1, order)
    weights = torch.softmax(values, dim=-1)
    return weights, indices


def dense_sparse_routes(
    weights: torch.Tensor,
    indices: torch.Tensor,
    slots: int,
) -> torch.Tensor:
    """Scatter selected route weights into a dense small-N representation."""

    if weights.shape != indices.shape:
        raise ValueError("route weights and indices must align")
    dense = weights.new_zeros(*weights.shape[:-1], slots)
    return dense.scatter(-1, indices, weights)


def sdm_copy_on_write_accounting(
    routing: SDMRouting,
    *,
    slots: int,
) -> SDMCopyOnWriteAccounting:
    """Measure exact hard occupancy and its differentiable union surrogate.

    Route generation is already position-parallel in ``SparseDeltaMemory``.
    This function consumes the complete route list with tensor scatter and
    reductions; it deliberately contains no tokenwise controller loop.

    Reads use the active set *after* the current token's writes, matching SDM's
    released read-after-write recurrence convention.
    """

    write_indices = routing.write_indices
    write_weights = routing.write_weights
    read_indices = routing.read_indices
    read_weights = routing.read_weights
    if (
        write_indices.ndim != 4
        or write_weights.shape != write_indices.shape
        or read_indices.ndim != 4
        or read_weights.shape != read_indices.shape
        or read_indices.shape[:3] != write_indices.shape[:3]
        or slots <= 0
    ):
        raise ValueError("SDM routing tensors do not align for COW accounting")
    # Avoid a scalar device synchronization in the per-step CUDA path.  The
    # model constructor guarantees bounds for generated routes; standalone CPU
    # callers still receive the defensive value check.
    if write_indices.device.type == "cpu" and write_indices.numel() and (
        int(write_indices.min()) < 0 or int(write_indices.max()) >= slots
    ):
        raise ValueError("SDM write route lies outside the logical table")
    if read_indices.device.type == "cpu" and read_indices.numel() and (
        int(read_indices.min()) < 0 or int(read_indices.max()) >= slots
    ):
        raise ValueError("SDM read route lies outside the logical table")

    with torch.autograd.profiler.record_function("sdm_copy_on_write_occupancy"):
        hard_events = torch.zeros(
            *write_indices.shape[:-1],
            slots,
            device=write_indices.device,
            dtype=torch.int32,
        ).scatter_add(
            -1,
            write_indices,
            torch.ones_like(write_indices, dtype=torch.int32),
        )
        hard_touches = hard_events.ne(0)
        active_by_position = hard_touches.to(torch.int32).cumsum(dim=1).ne(0)
        hard_unique = active_by_position.sum(dim=-1)
        prior_unique = torch.cat(
            (
                torch.zeros_like(hard_unique[:, :1]),
                hard_unique[:, :-1],
            ),
            dim=1,
        )
        first_touches = hard_unique - prior_unique
        writes_by_position = hard_events.sum(dim=-1)
        repeated_writes = writes_by_position - first_touches

        soft_routes = torch.zeros(
            *write_weights.shape[:-1],
            slots,
            device=write_weights.device,
            dtype=torch.float32,
        ).scatter_add(-1, write_indices, write_weights.float())
        # Product-key W4 weights lie in (0,1).  The clamp only protects the
        # logarithm at an exact unit boundary used by other SDM variants.
        soft_routes = soft_routes.clamp(min=0.0, max=1.0 - 1e-6)
        soft_occupied = -torch.expm1(torch.log1p(-soft_routes).sum(dim=1))
        soft_final_fraction = soft_occupied.mean(dim=(-1, -2))
        hard_final_fraction = hard_unique[:, -1].float().mean(dim=-1) / slots
        straight_through_fraction = (
            hard_final_fraction.detach()
            - soft_final_fraction.detach()
            + soft_final_fraction
        )

        private_read = torch.gather(active_by_position, -1, read_indices)
        private_read_count = private_read.sum(dim=-1)
        private_read_weight = (
            private_read.float() * read_weights.float()
        ).sum(dim=-1)
        entropy = -(
            write_weights.float().clamp_min(1e-12)
            * write_weights.float().clamp_min(1e-12).log()
        ).sum(dim=-1)

    return SDMCopyOnWriteAccounting(
        hard_unique_by_position=hard_unique,
        first_touch_count_by_position=first_touches,
        repeated_write_count_by_position=repeated_writes,
        private_read_count_by_position=private_read_count,
        private_read_weight_by_position=private_read_weight,
        write_route_entropy_by_position=entropy,
        slot_write_counts=hard_events.sum(dim=1),
        hard_final_fraction=hard_final_fraction,
        soft_final_fraction=soft_final_fraction,
        straight_through_fraction=straight_through_fraction,
    )


def aggregate_sdm_copy_on_write_accounting(
    routing: list[SDMRouting],
    *,
    slots: int,
) -> dict[str, torch.Tensor | int]:
    """Aggregate copy-on-write accounting across independent SDM banks.

    Each routing row represents one physical SDM controller.  The returned
    unique-slot counts sum physical slots across those controllers, while the
    normalized occupancy objectives average their per-controller fractions.
    This keeps the objective independent of depth without hiding the actual
    number of private values a sequence would materialize.
    """

    if not routing:
        raise ValueError("copy-on-write aggregation requires SDM routing")
    accounted = [
        sdm_copy_on_write_accounting(row, slots=slots) for row in routing
    ]
    unique_by_position = torch.stack(
        [row.hard_unique_by_position.sum(dim=-1) for row in accounted], dim=0
    ).sum(dim=0)
    first_touch_by_position = torch.stack(
        [row.first_touch_count_by_position.sum(dim=-1) for row in accounted], dim=0
    ).sum(dim=0)
    repeated_write_by_position = torch.stack(
        [row.repeated_write_count_by_position.sum(dim=-1) for row in accounted],
        dim=0,
    ).sum(dim=0)
    private_read_count_by_position = torch.stack(
        [row.private_read_count_by_position.sum(dim=-1) for row in accounted],
        dim=0,
    ).sum(dim=0)
    private_read_weight_by_position = torch.stack(
        [row.private_read_weight_by_position.sum(dim=-1) for row in accounted],
        dim=0,
    ).sum(dim=0)
    route_entropy_by_position = torch.stack(
        [row.write_route_entropy_by_position.mean(dim=-1) for row in accounted],
        dim=0,
    ).mean(dim=0)
    hard_fraction = torch.stack(
        [row.hard_final_fraction for row in accounted], dim=0
    ).mean(dim=0)
    soft_fraction = torch.stack(
        [row.soft_final_fraction for row in accounted], dim=0
    ).mean(dim=0)
    straight_through = torch.stack(
        [row.straight_through_fraction for row in accounted], dim=0
    ).mean(dim=0)
    unique_final_by_layer = torch.stack(
        [row.hard_unique_by_position[:, -1].sum(dim=-1) for row in accounted],
        dim=1,
    )
    slot_write_counts = torch.stack(
        [row.slot_write_counts for row in accounted], dim=1
    )
    read_selections_per_position = sum(
        row.read_indices.shape[2] * row.read_indices.shape[3] for row in routing
    )
    read_weight_per_position = sum(row.read_indices.shape[2] for row in routing)
    write_selections_per_position = sum(
        row.write_indices.shape[2] * row.write_indices.shape[3] for row in routing
    )
    return {
        "straight_through_fraction": straight_through,
        "hard_final_fraction": hard_fraction,
        "soft_final_fraction": soft_fraction,
        "unique_by_position": unique_by_position,
        "unique_final_by_layer": unique_final_by_layer,
        "first_touch_by_position": first_touch_by_position,
        "repeated_write_by_position": repeated_write_by_position,
        "private_read_count_by_position": private_read_count_by_position,
        "private_read_weight_by_position": private_read_weight_by_position,
        "route_entropy_by_position": route_entropy_by_position,
        "slot_write_counts": slot_write_counts,
        "layers": len(routing),
        "read_selections_per_position": read_selections_per_position,
        "read_weight_per_position": read_weight_per_position,
        "write_selections_per_position": write_selections_per_position,
    }


def serial_copy_on_write_gated_delta_recurrence(
    initial_memory: torch.Tensor,
    write_routes: torch.Tensor,
    values: torch.Tensor,
    input_gate: torch.Tensor,
    forget_log_gate: torch.Tensor,
    read_routes: torch.Tensor,
    *,
    erase_gate: torch.Tensor | None = None,
    initial_overlay: torch.Tensor | None = None,
    initial_active: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Serial semantic reference for an exact copy-on-write SDM overlay.

    Untouched logical slots read directly from shared ``initial_memory``.  A
    first write materializes that initial value into the private overlay before
    applying the unchanged SDM gated-delta update.  The returned effective
    states must therefore equal ``serial_gated_delta_recurrence`` exactly.
    """

    if initial_memory.ndim != 3:
        raise ValueError("initial memory must be [B,N,D]")
    batch, slots, width = initial_memory.shape
    if (
        write_routes.ndim != 3
        or read_routes.shape != write_routes.shape
        or write_routes.shape[0] != batch
        or write_routes.shape[2] != slots
        or values.shape != (batch, write_routes.shape[1], width)
        or input_gate.shape not in (write_routes.shape[:2], values.shape)
        or forget_log_gate.shape != input_gate.shape
        or (erase_gate is not None and erase_gate.shape != input_gate.shape)
    ):
        raise ValueError("SDM recurrence tensors do not align")
    if (initial_overlay is None) != (initial_active is None):
        raise ValueError("initial COW overlay and active mask are required together")
    if initial_overlay is not None and (
        initial_overlay.shape != initial_memory.shape
        or initial_active is None
        or initial_active.shape != initial_memory.shape[:2]
        or initial_active.dtype != torch.bool
    ):
        raise ValueError("initial COW overlay state does not align")

    shared = initial_memory.float()
    overlay = (
        torch.zeros_like(shared)
        if initial_overlay is None
        else initial_overlay.float()
    )
    active = (
        torch.zeros(
            batch,
            slots,
            dtype=torch.bool,
            device=initial_memory.device,
        )
        if initial_active is None
        else initial_active
    )
    readings: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    active_states: list[torch.Tensor] = []
    for position in range(write_routes.shape[1]):
        writes = write_routes[:, position].float()
        selected = writes.ne(0.0)
        memory = torch.where(active.unsqueeze(-1), overlay, shared)
        decay = forget_log_gate[:, position].float().exp()
        decay = decay.unsqueeze(-1) if decay.ndim == 1 else decay
        decayed = torch.where(
            selected.unsqueeze(-1),
            memory * decay[:, None, :],
            memory,
        )
        retrieved = torch.sum(writes.unsqueeze(-1) * decayed, dim=1)
        write = input_gate[:, position].float()
        write = write.unsqueeze(-1) if write.ndim == 1 else write
        erase = write if erase_gate is None else erase_gate[:, position].float()
        erase = erase.unsqueeze(-1) if erase.ndim == 1 else erase
        delta = write * values[:, position].float() - erase * retrieved
        updated = decayed + writes.unsqueeze(-1) * delta.unsqueeze(1)
        overlay = torch.where(selected.unsqueeze(-1), updated, overlay)
        active = active | selected
        memory = torch.where(active.unsqueeze(-1), overlay, shared)
        reading = torch.sum(
            read_routes[:, position].float().unsqueeze(-1) * memory,
            dim=1,
        )
        readings.append(reading)
        states.append(memory)
        active_states.append(active)
    return (
        torch.stack(readings, dim=1).to(values.dtype),
        memory,
        torch.stack(states, dim=1),
        torch.stack(active_states, dim=1),
    )


def serial_gated_delta_recurrence(
    initial_memory: torch.Tensor,
    write_routes: torch.Tensor,
    values: torch.Tensor,
    input_gate: torch.Tensor,
    forget_log_gate: torch.Tensor,
    read_routes: torch.Tensor,
    *,
    erase_gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simple differentiable token-serial oracle for the exact SDM rule.

    Reads occur after the current token's gated write, matching the released
    SDM controller.  Only nonzero write-route slots receive forget decay.
    """

    if initial_memory.ndim != 3:
        raise ValueError("initial memory must be [B,N,D]")
    batch, slots, width = initial_memory.shape
    if (
        write_routes.ndim != 3
        or read_routes.shape != write_routes.shape
        or write_routes.shape[0] != batch
        or write_routes.shape[2] != slots
        or values.shape != (batch, write_routes.shape[1], width)
        or input_gate.shape not in (write_routes.shape[:2], values.shape)
        or forget_log_gate.shape != input_gate.shape
        or (erase_gate is not None and erase_gate.shape != input_gate.shape)
    ):
        raise ValueError("SDM recurrence tensors do not align")

    memory = initial_memory.float()
    readings: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    for position in range(write_routes.shape[1]):
        writes = write_routes[:, position].float()
        decay = forget_log_gate[:, position].float().exp()
        decay = decay.unsqueeze(-1) if decay.ndim == 1 else decay
        selected = writes.ne(0.0)
        decayed = torch.where(
            selected.unsqueeze(-1),
            memory * decay[:, None, :],
            memory,
        )
        retrieved = torch.sum(writes.unsqueeze(-1) * decayed, dim=1)
        write = input_gate[:, position].float()
        write = write.unsqueeze(-1) if write.ndim == 1 else write
        erase = (
            write
            if erase_gate is None
            else erase_gate[:, position].float()
        )
        erase = erase.unsqueeze(-1) if erase.ndim == 1 else erase
        delta = write * values[:, position].float() - erase * retrieved
        memory = decayed + writes.unsqueeze(-1) * delta.unsqueeze(1)
        reading = torch.sum(
            read_routes[:, position].float().unsqueeze(-1) * memory,
            dim=1,
        )
        readings.append(reading)
        states.append(memory)
    return (
        torch.stack(readings, dim=1).to(values.dtype),
        memory,
        torch.stack(states, dim=1),
    )


if triton is not None:

    @triton.jit
    def _native_sdm_forward_kernel(
        initial_memory,
        write_routes,
        values,
        input_gate,
        erase_gate,
        forget_log_gate,
        read_routes,
        readings,
        states,
        time,
        width: tl.constexpr,
        slots: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCKS_D: tl.constexpr,
    ):
        program = tl.program_id(0)
        batch = program // BLOCKS_D
        width_block = program - batch * BLOCKS_D
        offsets_n = tl.arange(0, BLOCK_N)
        offsets_d = width_block * BLOCK_D + tl.arange(0, BLOCK_D)
        memory_offsets = (batch * slots + offsets_n[:, None]) * width + offsets_d[
            None, :
        ]
        memory = tl.load(
            initial_memory + memory_offsets,
            mask=(offsets_n[:, None] < slots) & (offsets_d[None, :] < width),
            other=0.0,
        ).to(tl.float32)

        for position in range(0, time):
            route_offsets = (batch * time + position) * slots + offsets_n
            writes = tl.load(
                write_routes + route_offsets,
                mask=offsets_n < slots,
                other=0.0,
            ).to(tl.float32)
            reads = tl.load(
                read_routes + route_offsets,
                mask=offsets_n < slots,
                other=0.0,
            ).to(tl.float32)
            alpha = tl.exp(
                tl.load(forget_log_gate + batch * time + position).to(tl.float32)
            )
            selected = writes != 0.0
            memory = tl.where(selected[:, None], memory * alpha, memory)
            retrieved = tl.sum(writes[:, None] * memory, axis=0)
            value_offsets = (batch * time + position) * width + offsets_d
            value = tl.load(
                values + value_offsets,
                mask=offsets_d < width,
                other=0.0,
            ).to(tl.float32)
            write_gate = tl.load(input_gate + batch * time + position).to(tl.float32)
            erase = tl.load(erase_gate + batch * time + position).to(tl.float32)
            delta = write_gate * value - erase * retrieved
            memory += writes[:, None] * delta[None, :]
            reading = tl.sum(reads[:, None] * memory, axis=0)
            tl.store(
                readings + value_offsets,
                reading,
                mask=offsets_d < width,
            )
            state_offsets = (
                (batch * time + position) * slots + offsets_n[:, None]
            ) * width + offsets_d[None, :]
            tl.store(
                states + state_offsets,
                memory,
                mask=(offsets_n[:, None] < slots) & (offsets_d[None, :] < width),
            )

    @triton.jit
    def _native_sdm_backward_kernel(
        initial_memory,
        write_routes,
        values,
        input_gate,
        erase_gate,
        forget_log_gate,
        read_routes,
        states,
        grad_readings,
        grad_final_memory,
        grad_initial_memory,
        grad_write_routes,
        grad_values,
        grad_input_gate,
        grad_erase_gate,
        grad_forget_log_gate,
        grad_read_routes,
        time,
        width: tl.constexpr,
        slots: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCKS_D: tl.constexpr,
    ):
        program = tl.program_id(0)
        batch = program // BLOCKS_D
        width_block = program - batch * BLOCKS_D
        offsets_n = tl.arange(0, BLOCK_N)
        offsets_d = width_block * BLOCK_D + tl.arange(0, BLOCK_D)
        memory_offsets = (batch * slots + offsets_n[:, None]) * width + offsets_d[
            None, :
        ]
        grad_memory = tl.load(
            grad_final_memory + memory_offsets,
            mask=(offsets_n[:, None] < slots) & (offsets_d[None, :] < width),
            other=0.0,
        ).to(tl.float32)

        for reverse_position in range(0, time):
            position = time - 1 - reverse_position
            route_offsets = (batch * time + position) * slots + offsets_n
            writes = tl.load(
                write_routes + route_offsets,
                mask=offsets_n < slots,
                other=0.0,
            ).to(tl.float32)
            reads = tl.load(
                read_routes + route_offsets,
                mask=offsets_n < slots,
                other=0.0,
            ).to(tl.float32)
            alpha = tl.exp(
                tl.load(forget_log_gate + batch * time + position).to(tl.float32)
            )
            write_gate = tl.load(input_gate + batch * time + position).to(tl.float32)
            erase = tl.load(erase_gate + batch * time + position).to(tl.float32)
            state_offsets = (
                (batch * time + position) * slots + offsets_n[:, None]
            ) * width + offsets_d[None, :]
            current = tl.load(
                states + state_offsets,
                mask=(offsets_n[:, None] < slots) & (offsets_d[None, :] < width),
                other=0.0,
            ).to(tl.float32)
            previous_state_offsets = (
                (batch * time + position - 1) * slots + offsets_n[:, None]
            ) * width + offsets_d[None, :]
            previous_from_states = tl.load(
                states + previous_state_offsets,
                mask=(position > 0)
                & (offsets_n[:, None] < slots)
                & (offsets_d[None, :] < width),
                other=0.0,
            ).to(tl.float32)
            previous_from_initial = tl.load(
                initial_memory + memory_offsets,
                mask=(position == 0)
                & (offsets_n[:, None] < slots)
                & (offsets_d[None, :] < width),
                other=0.0,
            ).to(tl.float32)
            previous = previous_from_states + previous_from_initial

            value_offsets = (batch * time + position) * width + offsets_d
            output_gradient = tl.load(
                grad_readings + value_offsets,
                mask=offsets_d < width,
                other=0.0,
            ).to(tl.float32)
            route_read_gradient = tl.sum(
                current * output_gradient[None, :],
                axis=1,
            )
            tl.atomic_add(
                grad_read_routes + route_offsets,
                route_read_gradient,
                mask=offsets_n < slots,
            )
            grad_memory += reads[:, None] * output_gradient[None, :]

            selected = writes != 0.0
            decayed = tl.where(selected[:, None], previous * alpha, previous)
            retrieved = tl.sum(writes[:, None] * decayed, axis=0)
            value = tl.load(
                values + value_offsets,
                mask=offsets_d < width,
                other=0.0,
            ).to(tl.float32)
            delta = write_gate * value - erase * retrieved
            route_write_gradient = tl.sum(
                grad_memory * delta[None, :],
                axis=1,
            )
            grad_delta = tl.sum(writes[:, None] * grad_memory, axis=0)
            grad_write_by_width = grad_delta * value
            grad_erase_by_width = -grad_delta * retrieved
            grad_value = write_gate * grad_delta
            grad_retrieved = -erase * grad_delta
            route_write_gradient += tl.sum(
                decayed * grad_retrieved[None, :],
                axis=1,
            )
            grad_decayed = grad_memory + writes[:, None] * grad_retrieved[None, :]
            grad_alpha_by_width = tl.sum(
                tl.where(selected[:, None], grad_decayed * previous, 0.0),
                axis=0,
            )
            grad_memory = tl.where(
                selected[:, None],
                grad_decayed * alpha,
                grad_decayed,
            )
            tl.atomic_add(
                grad_write_routes + route_offsets,
                route_write_gradient,
                mask=offsets_n < slots,
            )
            tl.store(
                grad_values + value_offsets,
                grad_value,
                mask=offsets_d < width,
            )
            grad_write = tl.sum(grad_write_by_width, axis=0)
            grad_erase = tl.sum(grad_erase_by_width, axis=0)
            grad_alpha = tl.sum(grad_alpha_by_width, axis=0)
            tl.atomic_add(
                grad_input_gate + batch * time + position,
                grad_write,
            )
            tl.atomic_add(
                grad_erase_gate + batch * time + position,
                grad_erase,
            )
            tl.atomic_add(
                grad_forget_log_gate + batch * time + position,
                grad_alpha * alpha,
            )

        tl.store(
            grad_initial_memory + memory_offsets,
            grad_memory,
            mask=(offsets_n[:, None] < slots) & (offsets_d[None, :] < width),
        )


class _NativeSDMRecurrence(torch.autograd.Function):
    """Custom autograd wrapper around the exact token-serial small-N kernel."""

    @staticmethod
    def forward(
        ctx,
        initial_memory: torch.Tensor,
        write_routes: torch.Tensor,
        values: torch.Tensor,
        input_gate: torch.Tensor,
        erase_gate: torch.Tensor,
        forget_log_gate: torch.Tensor,
        read_routes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if triton is None or not values.is_cuda:
            readings, final_memory, _ = serial_gated_delta_recurrence(
                initial_memory,
                write_routes,
                values,
                input_gate,
                forget_log_gate,
                read_routes,
                erase_gate=erase_gate,
            )
            return readings, final_memory
        batch, slots, width = initial_memory.shape
        time = values.shape[1]
        expected_gate_shape = (batch, time)
        if (
            initial_memory.dtype
            not in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            )
            or write_routes.shape != (batch, time, slots)
            or read_routes.shape != (batch, time, slots)
            or input_gate.shape != expected_gate_shape
            or erase_gate.shape != expected_gate_shape
            or forget_log_gate.shape != expected_gate_shape
        ):
            raise ValueError("optimized SDM recurrence inputs do not align")
        tensors = (
            initial_memory,
            write_routes,
            values,
            input_gate,
            erase_gate,
            forget_log_gate,
            read_routes,
        )
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("optimized SDM recurrence inputs must be contiguous")
        block_d = dense_recurrence_block_width(slots)
        block_n = triton.next_power_of_2(slots)
        if block_n > 1_024:
            raise ValueError("dense SDM training recurrence supports at most 1,024 slots")
        blocks_d = triton.cdiv(width, block_d)
        readings = torch.empty_like(values)
        states = torch.empty(
            batch,
            time,
            slots,
            width,
            device=values.device,
            dtype=torch.float32,
        )
        _native_sdm_forward_kernel[(batch * blocks_d,)](
            initial_memory,
            write_routes,
            values,
            input_gate,
            erase_gate,
            forget_log_gate,
            read_routes,
            readings,
            states,
            time,
            width=width,
            slots=slots,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
            BLOCKS_D=blocks_d,
            num_warps=4,
            num_stages=1,
        )
        final_memory = states[:, -1].clone()
        ctx.save_for_backward(
            initial_memory,
            write_routes,
            values,
            input_gate,
            erase_gate,
            forget_log_gate,
            read_routes,
            states,
        )
        ctx.block_d = block_d
        ctx.block_n = block_n
        ctx.blocks_d = blocks_d
        return readings, final_memory

    @staticmethod
    def backward(
        ctx,
        grad_readings: torch.Tensor | None,
        grad_final_memory: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        (
            initial_memory,
            write_routes,
            values,
            input_gate,
            erase_gate,
            forget_log_gate,
            read_routes,
            states,
        ) = ctx.saved_tensors
        batch, slots, width = initial_memory.shape
        time = values.shape[1]
        if grad_readings is None:
            grad_readings = torch.zeros_like(values)
        if grad_final_memory is None:
            grad_final_memory = torch.zeros_like(initial_memory)
        grad_readings = grad_readings.contiguous()
        grad_final_memory = grad_final_memory.float().contiguous()
        grad_initial_memory = torch.empty_like(initial_memory)
        grad_write_routes_f32 = torch.zeros_like(
            write_routes,
            dtype=torch.float32,
        )
        grad_values_f32 = torch.empty_like(values, dtype=torch.float32)
        grad_input_gate_f32 = torch.zeros_like(
            input_gate,
            dtype=torch.float32,
        )
        grad_erase_gate_f32 = torch.zeros_like(
            erase_gate,
            dtype=torch.float32,
        )
        grad_forget_f32 = torch.zeros_like(
            forget_log_gate,
            dtype=torch.float32,
        )
        grad_read_routes_f32 = torch.zeros_like(
            read_routes,
            dtype=torch.float32,
        )
        _native_sdm_backward_kernel[(batch * ctx.blocks_d,)](
            initial_memory,
            write_routes,
            values,
            input_gate,
            erase_gate,
            forget_log_gate,
            read_routes,
            states,
            grad_readings,
            grad_final_memory,
            grad_initial_memory,
            grad_write_routes_f32,
            grad_values_f32,
            grad_input_gate_f32,
            grad_erase_gate_f32,
            grad_forget_f32,
            grad_read_routes_f32,
            time,
            width=width,
            slots=slots,
            BLOCK_D=ctx.block_d,
            BLOCK_N=ctx.block_n,
            BLOCKS_D=ctx.blocks_d,
            num_warps=4,
            num_stages=1,
        )
        return (
            grad_initial_memory,
            grad_write_routes_f32.to(write_routes.dtype),
            grad_values_f32.to(values.dtype),
            grad_input_gate_f32.to(input_gate.dtype),
            grad_erase_gate_f32.to(erase_gate.dtype),
            grad_forget_f32.to(forget_log_gate.dtype),
            grad_read_routes_f32.to(read_routes.dtype),
        )


def gated_delta_recurrence(
    initial_memory: torch.Tensor,
    write_routes: torch.Tensor,
    values: torch.Tensor,
    input_gate: torch.Tensor,
    forget_log_gate: torch.Tensor,
    read_routes: torch.Tensor,
    *,
    erase_gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the optimized exact recurrence, or the serial oracle off CUDA."""

    if erase_gate is None:
        erase_gate = input_gate
    if triton is None or not values.is_cuda:
        readings, final_memory, _ = serial_gated_delta_recurrence(
            initial_memory,
            write_routes,
            values,
            input_gate,
            forget_log_gate,
            read_routes,
            erase_gate=erase_gate,
        )
        return readings, final_memory
    return _NativeSDMRecurrence.apply(
        initial_memory.contiguous(),
        write_routes.contiguous(),
        values.contiguous(),
        input_gate.contiguous(),
        erase_gate.contiguous(),
        forget_log_gate.contiguous(),
        read_routes.contiguous(),
    )


class SparseDeltaMemory(nn.Module):
    """Faithful small-workspace SDM controller for one physical layer.

    This class intentionally has no mechanism switches. It implements the
    released controller path used by this project: product-key routing,
    projected values, scalar-head gated-delta mutation, learned initial memory,
    normalized reads, a channelwise output gate, and an output projection.
    """

    def __init__(
        self,
        width: int,
        *,
        slots: int,
        reads: int,
        writes: int,
        memory_heads: int = 1,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if width <= 0 or slots <= 0 or memory_heads <= 0:
            raise ValueError("width, slots, and memory_heads must be positive")
        if width % memory_heads:
            raise ValueError("width must be divisible by memory_heads")
        root = math.isqrt(slots)
        if root * root != slots:
            raise ValueError("faithful SDM slots must be a perfect square")
        if not 1 <= reads <= slots or not 1 <= writes <= slots:
            raise ValueError("read and write counts must lie within the table")

        self.width = width
        self.slots = slots
        self.total_slots = slots
        self.reads = reads
        self.writes = writes
        self.memory_heads = memory_heads
        self.head_width = width // memory_heads
        self.key_width_per_head = 2 * root
        self.key_width = memory_heads * self.key_width_per_head

        self.input_norm = nn.LayerNorm(width)
        self.read_projection = nn.Linear(width, self.key_width, bias=True)
        self.write_projection = nn.Linear(width, self.key_width, bias=True)
        self.value_projection = nn.Linear(width, width, bias=True)
        self.forget_projection = nn.Linear(width, memory_heads, bias=True)
        self.input_projection = nn.Linear(width, memory_heads, bias=True)
        self.A_log = nn.Parameter(torch.empty(memory_heads))
        self.dt_bias = nn.Parameter(torch.empty(memory_heads))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True
        self.output_gate = nn.Linear(width, width, bias=True)
        self.output_projection = nn.Linear(width, width, bias=True)
        self.initial_memory = nn.Parameter(
            torch.empty(memory_heads, slots, self.head_width)
        )
        self.initial_memory._sdm_memory_bank = True
        self.readings_norm = nn.LayerNorm(self.head_width, eps=norm_eps)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = self.width**-0.5
        for projection in (
            self.read_projection,
            self.write_projection,
            self.value_projection,
            self.forget_projection,
            self.input_projection,
            self.output_gate,
            self.output_projection,
        ):
            nn.init.trunc_normal_(
                projection.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
            nn.init.zeros_(projection.bias)
        self.input_norm.reset_parameters()
        self.readings_norm.reset_parameters()
        with torch.no_grad():
            self.A_log.uniform_(0.0, 16.0).clamp_(min=1e-4).log_()
            self.dt_bias.uniform_(math.log(0.001), math.log(0.1)).exp_()
            self.dt_bias.copy_(self.dt_bias + torch.log(-torch.expm1(-self.dt_bias)))
        nn.init.trunc_normal_(
            self.initial_memory,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def reset_role_keyed(self, seed: int) -> None:
        """Initialize the released controller using stable semantic roles."""

        std = self.width**-0.5
        self.input_norm.reset_parameters()
        self.readings_norm.reset_parameters()
        for projection, projection_seed in (
            (self.read_projection, seed + 1),
            (self.write_projection, seed + 2),
            (self.value_projection, seed + 3),
            (self.forget_projection, seed + 4),
            (self.input_projection, seed + 5),
            (self.output_gate, seed + 6),
            (self.output_projection, seed + 7),
        ):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(projection_seed)
                nn.init.trunc_normal_(
                    projection.weight,
                    mean=0.0,
                    std=std,
                    a=-3 * std,
                    b=3 * std,
                )
                nn.init.zeros_(projection.bias)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 8)
            with torch.no_grad():
                self.A_log.uniform_(0.0, 16.0).clamp_(min=1e-4).log_()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 9)
            with torch.no_grad():
                self.dt_bias.uniform_(math.log(0.001), math.log(0.1)).exp_()
                self.dt_bias.copy_(
                    self.dt_bias + torch.log(-torch.expm1(-self.dt_bias))
                )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 10)
            nn.init.trunc_normal_(
                self.initial_memory,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_routing: bool = False,
        include_final_memory: bool = False,
        serial_reference: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SDMRouting]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.width:
            raise ValueError(f"SDM tokens must be [B,T,{self.width}]")
        if include_final_memory and not return_routing:
            raise ValueError("final SDM memory requires routing diagnostics")

        batch, time, _ = tokens.shape
        normalized = self.input_norm(tokens)
        with torch.autograd.profiler.record_function(
            "sdm_position_parallel_route_generation"
        ):
            raw_reads = self.read_projection(normalized).reshape(
                batch,
                time,
                self.memory_heads,
                self.key_width_per_head,
            )
            raw_writes = self.write_projection(normalized).reshape_as(raw_reads)
            routed_reads = raw_reads.permute(0, 2, 1, 3).flatten(0, 1)
            routed_writes = raw_writes.permute(0, 2, 1, 3).flatten(0, 1)
            read_weights, read_indices = product_key_routes(
                routed_reads,
                slots=self.slots,
                selected=self.reads,
            )
            write_weights, write_indices = product_key_routes(
                routed_writes,
                slots=self.slots,
                selected=self.writes,
            )

        read_routes = dense_sparse_routes(read_weights, read_indices, self.slots)
        write_routes = dense_sparse_routes(write_weights, write_indices, self.slots)
        values = (
            self.value_projection(normalized)
            .reshape(batch, time, self.memory_heads, self.head_width)
            .permute(0, 2, 1, 3)
            .flatten(0, 1)
        )
        forget_log_gate = (
            -torch.exp(self.A_log)
            * F.softplus(self.forget_projection(normalized) + self.dt_bias)
        ).permute(0, 2, 1).flatten(0, 1)
        input_gate = (
            torch.sigmoid(self.input_projection(normalized))
            .permute(0, 2, 1)
            .flatten(0, 1)
        )
        initial = (
            self.initial_memory.unsqueeze(0)
            .expand(batch, -1, -1, -1)
            .flatten(0, 1)
            .to(values.dtype)
        )
        if serial_reference:
            readings, final_memory, _ = serial_gated_delta_recurrence(
                initial,
                write_routes,
                values,
                input_gate,
                forget_log_gate,
                read_routes,
            )
        else:
            readings, final_memory = gated_delta_recurrence(
                initial,
                write_routes,
                values,
                input_gate,
                forget_log_gate,
                read_routes,
            )
        readings = self.readings_norm(readings)
        readings = (
            readings.reshape(batch, self.memory_heads, time, self.head_width)
            .permute(0, 2, 1, 3)
            .flatten(2)
        )
        output = self.output_projection(
            torch.sigmoid(self.output_gate(normalized)) * readings
        )
        if not return_routing:
            return output
        diagnostics = SDMRouting(
            read_indices=read_indices.reshape(
                batch, self.memory_heads, time, self.reads
            ).permute(0, 2, 1, 3),
            read_weights=read_weights.reshape(
                batch, self.memory_heads, time, self.reads
            ).permute(0, 2, 1, 3),
            write_indices=write_indices.reshape(
                batch, self.memory_heads, time, self.writes
            ).permute(0, 2, 1, 3),
            write_weights=write_weights.reshape(
                batch, self.memory_heads, time, self.writes
            ).permute(0, 2, 1, 3),
            forget_log_gate=forget_log_gate.reshape(
                batch, self.memory_heads, time
            ).permute(0, 2, 1),
            erase_gate=input_gate.reshape(
                batch, self.memory_heads, time
            ).permute(0, 2, 1),
            input_gate=input_gate.reshape(
                batch, self.memory_heads, time
            ).permute(0, 2, 1),
            values=values.reshape(
                batch, self.memory_heads, time, self.head_width
            ).permute(0, 2, 1, 3),
            final_memory=(
                final_memory.reshape(
                    batch,
                    self.memory_heads,
                    self.slots,
                    self.head_width,
                )
                if include_final_memory
                else None
            ),
        )
        return output, diagnostics
