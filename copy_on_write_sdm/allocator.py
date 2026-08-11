"""Packed copy-on-write state for exact faithful-SDM inference.

Faithful SDM exposes a fixed logical address space, but one sequence generally
touches only a subset of those addresses.  This module keeps the logical table
and controller semantics unchanged while storing private mutable values in a
shared slab.  Untouched logical addresses read from the model-shared learned
initial memory; a first write allocates one slab row, and later writes reuse it.

The allocator is deliberately inference-only.  Dense full-sequence training
remains the efficient semantic implementation, while this state realizes the
retained per-sequence product shape needed for serving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only development hosts
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _resolve_cow_write_rows_kernel(
        logical_to_physical,
        free_rows,
        free_count,
        overflow_flag,
        write_indices,
        resolved_rows,
        first_touches,
        logical_slots,
        WRITES: tl.constexpr,
    ):
        route_offset = tl.program_id(0)
        bank = route_offset // WRITES
        logical = tl.load(write_indices + route_offset)
        map_offset = bank * logical_slots + logical
        previous = tl.load(logical_to_physical + map_offset)
        first = previous < 0
        prior_free = tl.atomic_add(free_count, -1, mask=first)
        free_index = prior_free - 1
        available = free_index >= 0
        allocated = tl.load(
            free_rows + tl.maximum(free_index, 0),
            mask=first & available,
            other=0,
        )
        resolved = tl.where(first, allocated, previous)
        tl.store(
            logical_to_physical + map_offset,
            resolved,
            mask=~first | available,
        )
        tl.store(resolved_rows + route_offset, resolved)
        tl.store(
            first_touches + route_offset,
            first & available,
        )
        tl.atomic_max(
            overflow_flag,
            1,
            mask=first & ~available,
        )

    @triton.jit
    def _packed_cow_update_read_proven_narrow_kernel(
        shared_initial_memory,
        template_indices,
        values_slab,
        logical_to_physical,
        free_rows,
        free_count,
        overflow_flag,
        write_indices,
        first_touches,
        write_weights,
        values,
        input_gate,
        erase_gate,
        forget_log_gate,
        read_indices,
        read_weights,
        output,
        logical_slots,
        width,
        WRITES: tl.constexpr,
        READS: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
        COLLECT_DIAGNOSTICS: tl.constexpr,
    ):
        """Resolve first touches and execute one narrow SDM bank in one launch.

        The launch has exactly one program per independent bank, so allocation
        and the following value tiles cannot race across width programs.  This
        specialization is used only when the caller has proved that the slab
        can cover every remaining first touch in the fixed decode trace.
        """

        bank = tl.program_id(0)
        offsets_w = tl.arange(0, BLOCK_W)
        offsets_r = tl.arange(0, BLOCK_R)
        offsets_d = tl.arange(0, BLOCK_D)
        valid_w = offsets_w < WRITES
        valid_r = offsets_r < READS
        valid_d = offsets_d < width

        write_offset = bank * WRITES + offsets_w
        write_logical = tl.load(
            write_indices + write_offset,
            mask=valid_w,
            other=0,
        )
        map_offset = bank * logical_slots + write_logical
        previous = tl.load(
            logical_to_physical + map_offset,
            mask=valid_w,
            other=-1,
        )
        first = (previous < 0) & valid_w
        first_int = first.to(tl.int32)
        first_count = tl.sum(first_int, axis=0)
        prior_free = tl.atomic_add(free_count, -first_count)
        first_rank = tl.cumsum(first_int, axis=0) - 1
        free_index = prior_free - 1 - first_rank
        available = free_index >= 0
        allocated = tl.load(
            free_rows + tl.maximum(free_index, 0),
            mask=first & available,
            other=0,
        )
        write_physical = tl.where(first, allocated, previous)
        tl.store(
            logical_to_physical + map_offset,
            write_physical,
            mask=(~first | available) & valid_w,
        )
        if COLLECT_DIAGNOSTICS:
            tl.store(first_touches + write_offset, first & available, mask=valid_w)
        tl.atomic_max(overflow_flag, 1, mask=first_count > prior_free)

        template = tl.load(template_indices + bank)
        private_offsets = write_physical[:, None] * width + offsets_d[None, :]
        shared_offsets = (
            (template * logical_slots + write_logical[:, None]) * width
            + offsets_d[None, :]
        )
        private = tl.load(
            values_slab + private_offsets,
            mask=valid_w[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        shared = tl.load(
            shared_initial_memory + shared_offsets,
            mask=valid_w[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        current = tl.where(first[:, None], shared, private)
        alpha = tl.exp(tl.load(forget_log_gate + bank).to(tl.float32))
        write_gate = tl.load(input_gate + bank).to(tl.float32)
        erase = tl.load(erase_gate + bank).to(tl.float32)
        decayed = current * alpha
        write_weight = tl.load(
            write_weights + write_offset,
            mask=valid_w,
            other=0.0,
        ).to(tl.float32)
        retrieved = tl.sum(write_weight[:, None] * decayed, axis=0)
        value = tl.load(
            values + bank * width + offsets_d,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        delta = write_gate * value - erase * retrieved
        updated = decayed + write_weight[:, None] * delta[None, :]
        tl.store(
            values_slab + private_offsets,
            updated,
            mask=valid_w[:, None] & valid_d[None, :],
        )
        tl.debug_barrier()

        read_offset = bank * READS + offsets_r
        read_logical = tl.load(
            read_indices + read_offset,
            mask=valid_r,
            other=0,
        )
        read_physical = tl.load(
            logical_to_physical + bank * logical_slots + read_logical,
            mask=valid_r,
            other=-1,
        )
        read_is_private = read_physical >= 0
        read_private_offsets = (
            tl.maximum(read_physical, 0)[:, None] * width + offsets_d[None, :]
        )
        read_shared_offsets = (
            (template * logical_slots + read_logical[:, None]) * width
            + offsets_d[None, :]
        )
        read_private = tl.load(
            values_slab + read_private_offsets,
            mask=valid_r[:, None] & read_is_private[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        read_shared = tl.load(
            shared_initial_memory + read_shared_offsets,
            mask=valid_r[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        visible = tl.where(read_is_private[:, None], read_private, read_shared)
        read_weight = tl.load(
            read_weights + read_offset,
            mask=valid_r,
            other=0.0,
        ).to(tl.float32)
        reading = tl.sum(read_weight[:, None] * visible, axis=0)
        tl.store(output + bank * width + offsets_d, reading, mask=valid_d)

    @triton.jit
    def _packed_cow_update_read_kernel(
        shared_initial_memory,
        template_indices,
        values_slab,
        logical_to_physical,
        write_indices,
        write_rows,
        first_touches,
        write_weights,
        values,
        input_gate,
        erase_gate,
        forget_log_gate,
        read_indices,
        read_weights,
        output,
        logical_slots,
        width,
        WRITES: tl.constexpr,
        READS: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        bank = tl.program_id(0)
        width_block = tl.program_id(1)
        offsets_w = tl.arange(0, BLOCK_W)
        offsets_r = tl.arange(0, BLOCK_R)
        offsets_d = width_block * BLOCK_D + tl.arange(0, BLOCK_D)
        valid_w = offsets_w < WRITES
        valid_r = offsets_r < READS
        valid_d = offsets_d < width

        write_offset = bank * WRITES + offsets_w
        write_logical = tl.load(
            write_indices + write_offset,
            mask=valid_w,
            other=0,
        )
        write_physical = tl.load(
            write_rows + write_offset,
            mask=valid_w,
            other=0,
        )
        first = tl.load(
            first_touches + write_offset,
            mask=valid_w,
            other=0,
        ).to(tl.int1)
        template = tl.load(template_indices + bank)
        private_offsets = write_physical[:, None] * width + offsets_d[None, :]
        shared_offsets = (
            (template * logical_slots + write_logical[:, None]) * width
            + offsets_d[None, :]
        )
        private = tl.load(
            values_slab + private_offsets,
            mask=valid_w[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        shared = tl.load(
            shared_initial_memory + shared_offsets,
            mask=valid_w[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        current = tl.where(first[:, None], shared, private)
        alpha = tl.exp(tl.load(forget_log_gate + bank).to(tl.float32))
        write_gate = tl.load(input_gate + bank).to(tl.float32)
        erase = tl.load(erase_gate + bank).to(tl.float32)
        decayed = current * alpha
        write_weight = tl.load(
            write_weights + write_offset,
            mask=valid_w,
            other=0.0,
        ).to(tl.float32)
        retrieved = tl.sum(write_weight[:, None] * decayed, axis=0)
        value = tl.load(
            values + bank * width + offsets_d,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        delta = write_gate * value - erase * retrieved
        updated = decayed + write_weight[:, None] * delta[None, :]
        tl.store(
            values_slab + private_offsets,
            updated,
            mask=valid_w[:, None] & valid_d[None, :],
        )
        # Read-after-write is part of faithful SDM semantics.  Each width tile
        # is owned by this same program, so a block barrier makes the sparse
        # writes visible before any selected read reloads the slab.
        tl.debug_barrier()

        read_offset = bank * READS + offsets_r
        read_logical = tl.load(
            read_indices + read_offset,
            mask=valid_r,
            other=0,
        )
        read_physical = tl.load(
            logical_to_physical + bank * logical_slots + read_logical,
            mask=valid_r,
            other=-1,
        )
        read_is_private = read_physical >= 0
        read_private_offsets = (
            tl.maximum(read_physical, 0)[:, None] * width + offsets_d[None, :]
        )
        read_shared_offsets = (
            (template * logical_slots + read_logical[:, None]) * width
            + offsets_d[None, :]
        )
        read_private = tl.load(
            values_slab + read_private_offsets,
            mask=valid_r[:, None] & read_is_private[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        read_shared = tl.load(
            shared_initial_memory + read_shared_offsets,
            mask=valid_r[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        visible = tl.where(read_is_private[:, None], read_private, read_shared)
        read_weight = tl.load(
            read_weights + read_offset,
            mask=valid_r,
            other=0.0,
        ).to(tl.float32)
        reading = tl.sum(read_weight[:, None] * visible, axis=0)
        tl.store(
            output + bank * width + offsets_d,
            reading,
            mask=valid_d,
        )

    @triton.jit
    def _dense_sparse_update_read_kernel(
        memory,
        write_indices,
        write_weights,
        values,
        input_gate,
        erase_gate,
        forget_log_gate,
        read_indices,
        read_weights,
        output,
        logical_slots,
        width,
        WRITES: tl.constexpr,
        READS: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        bank = tl.program_id(0)
        width_block = tl.program_id(1)
        offsets_w = tl.arange(0, BLOCK_W)
        offsets_r = tl.arange(0, BLOCK_R)
        offsets_d = width_block * BLOCK_D + tl.arange(0, BLOCK_D)
        valid_w = offsets_w < WRITES
        valid_r = offsets_r < READS
        valid_d = offsets_d < width

        write_offset = bank * WRITES + offsets_w
        write_logical = tl.load(
            write_indices + write_offset,
            mask=valid_w,
            other=0,
        )
        write_memory_offset = (
            (bank * logical_slots + write_logical[:, None]) * width
            + offsets_d[None, :]
        )
        current = tl.load(
            memory + write_memory_offset,
            mask=valid_w[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        alpha = tl.exp(tl.load(forget_log_gate + bank).to(tl.float32))
        write_gate = tl.load(input_gate + bank).to(tl.float32)
        erase = tl.load(erase_gate + bank).to(tl.float32)
        decayed = current * alpha
        write_weight = tl.load(
            write_weights + write_offset,
            mask=valid_w,
            other=0.0,
        ).to(tl.float32)
        retrieved = tl.sum(write_weight[:, None] * decayed, axis=0)
        value = tl.load(
            values + bank * width + offsets_d,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        delta = write_gate * value - erase * retrieved
        updated = decayed + write_weight[:, None] * delta[None, :]
        tl.store(
            memory + write_memory_offset,
            updated,
            mask=valid_w[:, None] & valid_d[None, :],
        )
        tl.debug_barrier()

        read_offset = bank * READS + offsets_r
        read_logical = tl.load(
            read_indices + read_offset,
            mask=valid_r,
            other=0,
        )
        read_memory_offset = (
            (bank * logical_slots + read_logical[:, None]) * width
            + offsets_d[None, :]
        )
        visible = tl.load(
            memory + read_memory_offset,
            mask=valid_r[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        read_weight = tl.load(
            read_weights + read_offset,
            mask=valid_r,
            other=0.0,
        ).to(tl.float32)
        reading = tl.sum(read_weight[:, None] * visible, axis=0)
        tl.store(
            output + bank * width + offsets_d,
            reading,
            mask=valid_d,
        )


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


@dataclass(frozen=True)
class SDMCopyOnWriteStep:
    """Diagnostics for one packed SDM decode step."""

    first_touches_by_sequence: torch.Tensor
    repeated_writes_by_sequence: torch.Tensor
    allocated_rows_after_step: torch.Tensor


@dataclass
class PackedSDMCopyOnWriteState:
    """One shared slab backing several independent logical SDM tables.

    ``shared_initial_memory`` has one row per model-owned memory template, not
    one row per sequence.  ``template_indices`` maps each independent serving
    bank to its template.  In the common one-head case, every sequence maps to
    template zero.

    ``logical_to_physical`` is a small direct lookup table.  Its O(B*N)
    integer metadata is reported separately from the O(C*D) value slab, where
    B is the number of live banks and C is the shared physical-row capacity.
    The direct map is intentionally simple and deterministic for the first
    allocator; a later compressed map can replace it without changing SDM.
    """

    shared_initial_memory: torch.Tensor
    template_indices: torch.Tensor
    values: torch.Tensor
    logical_to_physical: torch.Tensor
    free_rows: torch.Tensor
    free_count: torch.Tensor
    overflow_flag: torch.Tensor
    capacity_is_proven: bool = False
    proven_maximum_additional_allocations: int = 0
    growth_quantum_rows: int = 0
    growth_events: int = 0
    growth_rows_added: int = 0
    growth_rows_copied: int = 0

    @classmethod
    def allocate(
        cls,
        shared_initial_memory: torch.Tensor,
        *,
        banks: int,
        capacity_rows: int,
        template_indices: torch.Tensor | None = None,
        state_dtype: torch.dtype = torch.float32,
        growth_quantum_rows: int = 0,
    ) -> "PackedSDMCopyOnWriteState":
        """Allocate an empty shared slab without copying learned initial state."""

        if shared_initial_memory.ndim == 2:
            shared_initial_memory = shared_initial_memory.unsqueeze(0)
        if shared_initial_memory.ndim != 3:
            raise ValueError("shared initial memory must be [H,N,D] or [N,D]")
        templates, slots, width = shared_initial_memory.shape
        if banks <= 0 or capacity_rows <= 0:
            raise ValueError("banks and physical capacity must be positive")
        if growth_quantum_rows < 0:
            raise ValueError("SDM growth quantum cannot be negative")
        if state_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("SDM COW state must use float16, bfloat16, or float32")
        device = shared_initial_memory.device
        if template_indices is None:
            if templates != 1:
                raise ValueError(
                    "multiple initial-memory templates require explicit mapping"
                )
            template_indices = torch.zeros(
                banks,
                device=device,
                dtype=torch.int64,
            )
        elif (
            template_indices.shape != (banks,)
            or template_indices.dtype not in (torch.int32, torch.int64)
            or template_indices.device != device
        ):
            raise ValueError("template indices must be integer [B] on the state device")
        else:
            template_indices = template_indices.to(torch.int64)
        if template_indices.device.type == "cpu":
            if int(template_indices.min()) < 0 or int(template_indices.max()) >= templates:
                raise ValueError("template index lies outside shared initial memory")
        elif hasattr(torch, "_assert_async"):
            torch._assert_async(
                ((template_indices >= 0) & (template_indices < templates)).all(),
                "template index lies outside shared initial memory",
            )

        return cls(
            shared_initial_memory=shared_initial_memory,
            template_indices=template_indices.contiguous(),
            values=torch.empty(
                capacity_rows,
                width,
                device=device,
                dtype=state_dtype,
            ),
            logical_to_physical=torch.full(
                (banks, slots),
                -1,
                device=device,
                dtype=torch.int32,
            ),
            free_rows=torch.arange(
                capacity_rows,
                device=device,
                dtype=torch.int32,
            ),
            free_count=torch.tensor(
                capacity_rows,
                device=device,
                dtype=torch.int32,
            ),
            overflow_flag=torch.zeros((), device=device, dtype=torch.int32),
            growth_quantum_rows=growth_quantum_rows,
        )

    @classmethod
    @torch.no_grad()
    def from_dense_overlay(
        cls,
        shared_initial_memory: torch.Tensor,
        dense_memory: torch.Tensor,
        active_slots: torch.Tensor,
        *,
        template_indices: torch.Tensor,
        capacity_rows: int | None = None,
        state_dtype: torch.dtype = torch.float32,
        maximum_future_allocations: int | None = None,
        growth_quantum_rows: int = 0,
    ) -> "PackedSDMCopyOnWriteState":
        """Pack an exact full-prefix state into a first-write overlay.

        ``dense_memory`` contains the terminal state for every serving bank,
        while ``active_slots`` is the union of that prefix's hard write
        addresses.  Untouched rows remain model-shared ``M0`` and consume no
        private value row.  This is a prefill handoff operation, not a decode
        hot-path allocator.
        """

        if shared_initial_memory.ndim == 2:
            shared_initial_memory = shared_initial_memory.unsqueeze(0)
        if shared_initial_memory.ndim != 3 or dense_memory.ndim != 3:
            raise ValueError("SDM handoff memory must be [H,N,D] and [B,N,D]")
        banks, slots, width = dense_memory.shape
        if (
            active_slots.shape != (banks, slots)
            or active_slots.dtype != torch.bool
            or active_slots.device != dense_memory.device
            or shared_initial_memory.device != dense_memory.device
            or shared_initial_memory.shape[1:] != (slots, width)
        ):
            raise ValueError("SDM handoff state and active-slot mask do not align")
        if (
            template_indices.shape != (banks,)
            or template_indices.device != dense_memory.device
        ):
            raise ValueError("SDM handoff template indices must be [B] on device")
        active_count = int(active_slots.sum().item())
        if growth_quantum_rows < 0:
            raise ValueError("SDM growth quantum cannot be negative")
        if capacity_rows is None and growth_quantum_rows:
            selected_capacity = max(
                growth_quantum_rows,
                ((active_count + growth_quantum_rows - 1) // growth_quantum_rows)
                * growth_quantum_rows,
            )
        else:
            selected_capacity = banks * slots if capacity_rows is None else capacity_rows
        if selected_capacity < active_count:
            raise ValueError(
                "SDM physical capacity is smaller than the prefill working set"
            )
        if maximum_future_allocations is not None:
            if maximum_future_allocations < 0:
                raise ValueError("maximum future allocations cannot be negative")
            possible_future_allocations = min(
                maximum_future_allocations,
                banks * slots - active_count,
            )
            if selected_capacity - active_count < possible_future_allocations:
                raise ValueError(
                    "SDM physical capacity cannot cover the declared decode trace"
                )
        else:
            possible_future_allocations = 0
        state = cls.allocate(
            shared_initial_memory,
            banks=banks,
            capacity_rows=selected_capacity,
            template_indices=template_indices,
            state_dtype=state_dtype,
            growth_quantum_rows=growth_quantum_rows,
        )
        if active_count:
            bank, slot = active_slots.nonzero(as_tuple=True)
            physical = torch.arange(
                active_count,
                device=dense_memory.device,
                dtype=torch.int32,
            )
            state.logical_to_physical[bank, slot] = physical
            state.values[:active_count].copy_(
                dense_memory[bank, slot].to(state_dtype)
            )
        free_count = selected_capacity - active_count
        if free_count:
            state.free_rows[:free_count].copy_(
                torch.arange(
                    active_count,
                    selected_capacity,
                    device=dense_memory.device,
                    dtype=torch.int32,
                )
            )
        state.free_count.fill_(free_count)
        state.capacity_is_proven = maximum_future_allocations is not None
        state.proven_maximum_additional_allocations = possible_future_allocations
        state.validate_invariants()
        return state

    @torch.no_grad()
    def grow_capacity(self, minimum_capacity_rows: int) -> int:
        """Grow the shared value slab without changing any logical SDM state.

        Growth is a request-lifecycle operation, not part of the recurrent
        controller.  Existing physical row IDs remain stable.  New rows join
        the shared free stack and are available to every bank; no bank receives
        a private quota.
        """

        if minimum_capacity_rows <= self.capacity_rows:
            return 0
        quantum = self.growth_quantum_rows
        if quantum <= 0:
            raise ValueError("SDM COW slab growth is disabled")
        selected_capacity = (
            (minimum_capacity_rows + quantum - 1) // quantum
        ) * quantum
        old_capacity = self.capacity_rows
        added = selected_capacity - old_capacity
        old_free_count = int(self.free_count.item())

        values = torch.empty(
            selected_capacity,
            self.width,
            device=self.values.device,
            dtype=self.values.dtype,
        )
        values[:old_capacity].copy_(self.values)
        free_rows = torch.empty(
            selected_capacity,
            device=self.free_rows.device,
            dtype=self.free_rows.dtype,
        )
        if old_free_count:
            free_rows[:old_free_count].copy_(self.free_rows[:old_free_count])
        free_rows[old_free_count : old_free_count + added].copy_(
            torch.arange(
                old_capacity,
                selected_capacity,
                device=self.free_rows.device,
                dtype=self.free_rows.dtype,
            )
        )
        self.values = values
        self.free_rows = free_rows
        self.free_count.fill_(old_free_count + added)
        self.proven_maximum_additional_allocations += added
        self.overflow_flag.zero_()
        self.growth_events += 1
        self.growth_rows_added += added
        self.growth_rows_copied += old_capacity
        return added

    @torch.no_grad()
    def prepare_step_capacity(self, maximum_new_rows: int) -> int:
        """Prove one fused decode step and grow only aggregate pool headroom.

        The host-side counter is deliberately pessimistic: it charges every
        possible W-way first touch until the next occasional device count
        reconciliation.  Consequently the fused kernel never observes an
        undersized slab, while the pool reservation remains within one growth
        quantum plus one step's maximum demand of the actual aggregate working
        set.
        """

        if maximum_new_rows < 0:
            raise ValueError("maximum new SDM rows cannot be negative")
        if self.growth_quantum_rows <= 0:
            if not self.capacity_is_proven:
                raise ValueError("SDM COW capacity has no decode proof")
            return 0
        if self.proven_maximum_additional_allocations < maximum_new_rows:
            free_rows = int(self.free_count.item())
            if free_rows < maximum_new_rows:
                self.grow_capacity(
                    self.capacity_rows + maximum_new_rows - free_rows
                )
                free_rows = int(self.free_count.item())
            self.proven_maximum_additional_allocations = free_rows
        self.proven_maximum_additional_allocations -= maximum_new_rows
        self.capacity_is_proven = True
        return self.capacity_rows

    @torch.no_grad()
    def trim_capacity(self, *, headroom_rows: int = 0) -> int:
        """Repack live rows and return unused slab pages to tensor storage.

        Servers should invoke this at a lifecycle boundary rather than on the
        decode hot path.  Released space is normally reused directly; trimming
        is useful after a sustained drop in aggregate demand or an unusually
        large request leaves the pool.
        """

        if headroom_rows < 0:
            raise ValueError("SDM trim headroom cannot be negative")
        quantum = self.growth_quantum_rows
        if quantum <= 0:
            raise ValueError("SDM COW slab trimming requires copy-on-write capacity")
        active = self.logical_to_physical >= 0
        active_count = int(active.sum().item())
        selected_capacity = max(
            quantum,
            ((active_count + headroom_rows + quantum - 1) // quantum) * quantum,
        )
        if selected_capacity >= self.capacity_rows:
            return 0

        values = torch.empty(
            selected_capacity,
            self.width,
            device=self.values.device,
            dtype=self.values.dtype,
        )
        mapping = torch.full_like(self.logical_to_physical, -1)
        if active_count:
            bank, slot = active.nonzero(as_tuple=True)
            source_rows = self.logical_to_physical[bank, slot].to(torch.int64)
            destination_rows = torch.arange(
                active_count,
                device=self.values.device,
                dtype=torch.int32,
            )
            values[:active_count].copy_(self.values[source_rows])
            mapping[bank, slot] = destination_rows
        free_count = selected_capacity - active_count
        free_rows = torch.empty(
            selected_capacity,
            device=self.free_rows.device,
            dtype=self.free_rows.dtype,
        )
        if free_count:
            free_rows[:free_count].copy_(
                torch.arange(
                    active_count,
                    selected_capacity,
                    device=self.free_rows.device,
                    dtype=self.free_rows.dtype,
                )
            )
        removed = self.capacity_rows - selected_capacity
        self.values = values
        self.logical_to_physical = mapping
        self.free_rows = free_rows
        self.free_count.fill_(free_count)
        self.capacity_is_proven = False
        self.proven_maximum_additional_allocations = free_count
        self.overflow_flag.zero_()
        return removed

    @torch.no_grad()
    def append_banks_from(self, other: "PackedSDMCopyOnWriteState") -> int:
        """Move another request cohort into this pool's logical bank space.

        Only live private rows are copied.  Untouched addresses continue to
        resolve through the shared learned initial memory and consume no value
        row in the destination pool.
        """

        if (
            other.slots != self.slots
            or other.width != self.width
            or other.values.device != self.values.device
            or other.values.dtype != self.values.dtype
            or other.shared_initial_memory.shape
            != self.shared_initial_memory.shape
        ):
            raise ValueError("admitted SDM banks do not match the destination pool")
        if (
            other.shared_initial_memory.data_ptr()
            != self.shared_initial_memory.data_ptr()
            and not torch.equal(
                other.shared_initial_memory,
                self.shared_initial_memory,
            )
        ):
            raise ValueError("admitted SDM banks use different learned initial memory")

        active = other.logical_to_physical >= 0
        active_count = int(active.sum().item())
        free_count = int(self.free_count.item())
        if free_count < active_count:
            self.grow_capacity(self.capacity_rows + active_count - free_count)
            free_count = int(self.free_count.item())

        admitted_mapping = torch.full(
            (other.banks, self.slots),
            -1,
            device=self.logical_to_physical.device,
            dtype=self.logical_to_physical.dtype,
        )
        if active_count:
            bank, slot = active.nonzero(as_tuple=True)
            pop_positions = torch.arange(
                free_count - 1,
                free_count - active_count - 1,
                -1,
                device=self.free_rows.device,
                dtype=torch.int64,
            )
            destination_rows = self.free_rows[pop_positions]
            source_rows = other.logical_to_physical[bank, slot].to(torch.int64)
            self.values[destination_rows.to(torch.int64)] = other.values[
                source_rows
            ]
            admitted_mapping[bank, slot] = destination_rows
            self.free_count.fill_(free_count - active_count)
        self.logical_to_physical = torch.cat(
            (self.logical_to_physical, admitted_mapping),
            dim=0,
        ).contiguous()
        self.template_indices = torch.cat(
            (self.template_indices, other.template_indices),
            dim=0,
        ).contiguous()
        if self.growth_quantum_rows:
            self.proven_maximum_additional_allocations = min(
                self.proven_maximum_additional_allocations,
                int(self.free_count.item()),
            )
        else:
            self.capacity_is_proven = False
            self.proven_maximum_additional_allocations = 0
        return active_count

    @torch.no_grad()
    def retain_banks(self, bank_indices: torch.Tensor) -> None:
        """Compact already-released banks without moving live value rows."""

        if (
            bank_indices.ndim != 1
            or bank_indices.dtype not in (torch.int32, torch.int64)
            or bank_indices.device != self.values.device
        ):
            raise ValueError("retained bank indices must be integer [S] on device")
        indices = bank_indices.to(torch.int64)
        if indices.numel() and (
            bool((indices < 0).any())
            or bool((indices >= self.banks).any())
            or indices.unique().numel() != indices.numel()
        ):
            raise ValueError("retained bank index is invalid or duplicated")
        selected = torch.zeros(
            self.banks,
            device=self.values.device,
            dtype=torch.bool,
        )
        selected[indices] = True
        if bool((self.logical_to_physical[~selected] >= 0).any()):
            raise ValueError("discarded SDM banks must be released before compaction")
        self.logical_to_physical = self.logical_to_physical[indices].contiguous()
        self.template_indices = self.template_indices[indices].contiguous()

    @property
    def banks(self) -> int:
        return self.logical_to_physical.shape[0]

    @property
    def slots(self) -> int:
        return self.logical_to_physical.shape[1]

    @property
    def width(self) -> int:
        return self.values.shape[1]

    @property
    def capacity_rows(self) -> int:
        return self.values.shape[0]

    def allocated_rows_tensor(self) -> torch.Tensor:
        """Return the device-resident number of live physical rows."""

        return self.free_count.new_tensor(self.capacity_rows) - self.free_count

    def storage_bytes(self) -> dict[str, int]:
        """Report actual retained tensor storage, separating shared model state."""

        private = {
            "value_slab_reserved": _tensor_bytes(self.values),
            "logical_to_physical": _tensor_bytes(self.logical_to_physical),
            "free_row_stack": _tensor_bytes(self.free_rows),
            "free_count": _tensor_bytes(self.free_count),
            "overflow_flag": _tensor_bytes(self.overflow_flag),
            "template_indices": _tensor_bytes(self.template_indices),
        }
        private["private_allocator_reserved_total"] = sum(private.values())
        private["shared_initial_memory"] = _tensor_bytes(self.shared_initial_memory)
        private["including_shared_initial_memory"] = (
            private["private_allocator_reserved_total"]
            + private["shared_initial_memory"]
        )
        return private

    def used_bytes(self) -> dict[str, int]:
        """Synchronously report live-row bytes plus always-resident metadata."""

        allocated = int(self.allocated_rows_tensor().item())
        live_values = allocated * self.width * self.values.element_size()
        metadata = (
            _tensor_bytes(self.logical_to_physical)
            + _tensor_bytes(self.free_rows)
            + _tensor_bytes(self.free_count)
            + _tensor_bytes(self.overflow_flag)
            + _tensor_bytes(self.template_indices)
        )
        return {
            "allocated_rows": allocated,
            "live_value_bytes": live_values,
            "allocator_metadata_bytes": metadata,
            "private_live_total": live_values + metadata,
        }

    @torch.no_grad()
    def materialize_dense(self) -> torch.Tensor:
        """Materialize the effective table for validation, never serving."""

        dense = self.shared_initial_memory[self.template_indices].to(
            self.values.dtype
        ).clone()
        active = self.logical_to_physical >= 0
        if not bool(active.any()):
            return dense
        bank, slot = active.nonzero(as_tuple=True)
        physical = self.logical_to_physical[bank, slot].to(torch.int64)
        dense[bank, slot] = self.values[physical]
        return dense

    @torch.no_grad()
    def release(self, bank_indices: torch.Tensor) -> torch.Tensor:
        """Release complete serving banks and return their row count to the slab.

        The operation is vectorized over the selected banks.  It performs no
        per-token work and is intended for request teardown or slot reuse.
        """

        if (
            bank_indices.ndim != 1
            or bank_indices.dtype not in (torch.int32, torch.int64)
            or bank_indices.device != self.values.device
        ):
            raise ValueError("released bank indices must be integer [S] on device")
        if bank_indices.numel() == 0:
            return self.free_count.new_zeros(())
        indices = bank_indices.to(torch.int64)
        sorted_indices = indices.sort().values
        unique = (
            torch.ones_like(sorted_indices, dtype=torch.bool)
            if sorted_indices.numel() == 1
            else torch.cat(
                (
                    torch.ones_like(sorted_indices[:1], dtype=torch.bool),
                    sorted_indices[1:] != sorted_indices[:-1],
                )
            )
        )
        valid = (
            (indices >= 0).all()
            & (indices < self.banks).all()
            & unique.all()
        )
        _device_assert(valid, "released bank index is invalid or duplicated")

        selected = self.logical_to_physical[indices]
        active = selected >= 0
        flat_active = active.flatten()
        flat_rows = selected.flatten()
        ranks = flat_active.to(torch.int32).cumsum(0) - 1
        released = flat_active.to(torch.int32).sum()
        old_free = self.free_count.clone()
        push_positions = old_free + ranks
        safe_positions = push_positions.clamp(0, self.capacity_rows - 1)
        self.free_rows[safe_positions[flat_active].to(torch.int64)] = flat_rows[
            flat_active
        ]
        self.free_count.copy_(old_free + released)
        self.logical_to_physical[indices] = -1
        return released

    @torch.no_grad()
    def validate_invariants(self) -> dict[str, Any]:
        """Synchronously validate row ownership and free-list partitioning."""

        mapping = self.logical_to_physical.cpu()
        free_rows = self.free_rows.cpu()
        free_count = int(self.free_count.item())
        if int(self.overflow_flag.item()) != 0:
            raise AssertionError("SDM COW allocator recorded a slab overflow")
        active = mapping[mapping >= 0].to(torch.int64)
        free = free_rows[:free_count].to(torch.int64)
        if active.numel() != active.unique().numel():
            raise AssertionError("one physical SDM row has multiple logical owners")
        if free.numel() != free.unique().numel():
            raise AssertionError("free SDM row stack contains duplicates")
        if active.numel() + free.numel() != self.capacity_rows:
            raise AssertionError("active and free SDM rows do not cover capacity")
        combined = torch.cat((active, free)).sort().values
        expected = torch.arange(self.capacity_rows, dtype=torch.int64)
        if not torch.equal(combined, expected):
            raise AssertionError("active and free SDM rows do not partition the slab")
        return {
            "capacity_rows": self.capacity_rows,
            "allocated_rows": active.numel(),
            "free_rows": free.numel(),
            "exact_partition": True,
        }


def _device_assert(condition: torch.Tensor, message: str) -> None:
    if condition.numel() != 1 or condition.dtype != torch.bool:
        raise ValueError("device assertion requires one boolean")
    if condition.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def _validate_indices(
    indices: torch.Tensor,
    *,
    name: str,
    banks: int,
    slots: int,
    device: torch.device,
    validate_bounds: bool = True,
) -> None:
    if (
        indices.ndim != 2
        or indices.shape[0] != banks
        or indices.shape[1] <= 0
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.device != device
    ):
        raise ValueError(f"{name} indices must be integer [B,K] on device")
    if validate_bounds:
        bounds = ((indices >= 0) & (indices < slots)).all()
        _device_assert(bounds, f"{name} index lies outside the logical table")


def _validate_unique_writes(write_indices: torch.Tensor) -> None:
    if write_indices.shape[1] == 1:
        return
    ordered = write_indices.sort(dim=-1).values
    unique = (ordered[:, 1:] != ordered[:, :-1]).all()
    _device_assert(unique, "one SDM step cannot write one logical slot twice")


def _expand_gate(gate: torch.Tensor, *, banks: int, width: int) -> torch.Tensor:
    if gate.shape == (banks,):
        return gate.float().reshape(banks, 1, 1)
    raise ValueError("native SDM gate must be [B]")


def _validate_gates(
    input_gate: torch.Tensor,
    erase_gate: torch.Tensor,
    forget_log_gate: torch.Tensor,
    *,
    banks: int,
    width: int,
) -> None:
    scalar = (banks,)
    if input_gate.shape != scalar:
        raise ValueError("native SDM input gate must be [B]")
    if erase_gate.shape != input_gate.shape or forget_log_gate.shape != input_gate.shape:
        raise ValueError("SDM input, erase, and forget gates must align")


def _use_triton_backend(
    backend: str,
    *,
    tensor: torch.Tensor,
    maximum_writes: int,
    maximum_reads: int,
) -> bool:
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("allocator backend must be auto, torch, or triton")
    supported = (
        triton is not None
        and tensor.is_cuda
        and tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and maximum_writes <= 64
        and maximum_reads <= 256
    )
    if backend == "triton" and not supported:
        raise RuntimeError("the requested Triton SDM COW backend is unavailable")
    return backend != "torch" and supported


def _packed_sdm_cow_step_triton(
    state: PackedSDMCopyOnWriteState,
    write_indices: torch.Tensor,
    write_weights: torch.Tensor,
    values: torch.Tensor,
    input_gate: torch.Tensor,
    erase_gate: torch.Tensor,
    forget_log_gate: torch.Tensor,
    read_indices: torch.Tensor,
    read_weights: torch.Tensor,
    *,
    collect_diagnostics: bool,
) -> tuple[torch.Tensor, SDMCopyOnWriteStep | None]:
    if triton is None:  # pragma: no cover - caller proves availability
        raise RuntimeError("Triton is unavailable")
    tensors = (
        state.shared_initial_memory,
        state.template_indices,
        state.values,
        state.logical_to_physical,
        state.free_rows,
        state.free_count,
        state.overflow_flag,
        write_indices,
        write_weights,
        values,
        input_gate,
        erase_gate,
        forget_log_gate,
        read_indices,
        read_weights,
    )
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Triton SDM COW tensors must be contiguous")
    banks, width = values.shape
    writes = write_indices.shape[1]
    reads = read_indices.shape[1]
    block_w = triton.next_power_of_2(writes)
    block_r = triton.next_power_of_2(reads)
    if state.capacity_is_proven and width <= 128:
        block_d = triton.next_power_of_2(width)
        first_touches = (
            torch.empty_like(write_indices, dtype=torch.bool)
            if collect_diagnostics
            else state.overflow_flag
        )
        output = torch.empty_like(values)
        _packed_cow_update_read_proven_narrow_kernel[(banks,)](
            state.shared_initial_memory,
            state.template_indices,
            state.values,
            state.logical_to_physical,
            state.free_rows,
            state.free_count,
            state.overflow_flag,
            write_indices,
            first_touches,
            write_weights,
            values,
            input_gate,
            erase_gate,
            forget_log_gate,
            read_indices,
            read_weights,
            output,
            state.slots,
            width,
            WRITES=writes,
            READS=reads,
            BLOCK_W=block_w,
            BLOCK_R=block_r,
            BLOCK_D=block_d,
            COLLECT_DIAGNOSTICS=collect_diagnostics,
            num_warps=4,
            num_stages=2,
        )
        if not collect_diagnostics:
            return output, None
        first_by_sequence = first_touches.sum(dim=-1)
        return output, SDMCopyOnWriteStep(
            first_touches_by_sequence=first_by_sequence,
            repeated_writes_by_sequence=(writes - first_by_sequence),
            allocated_rows_after_step=state.allocated_rows_tensor(),
        )
    block_d = 32
    resolved_rows = torch.empty_like(write_indices, dtype=torch.int32)
    first_touches = torch.empty_like(write_indices, dtype=torch.bool)
    _resolve_cow_write_rows_kernel[(banks * writes,)](
        state.logical_to_physical,
        state.free_rows,
        state.free_count,
        state.overflow_flag,
        write_indices,
        resolved_rows,
        first_touches,
        state.slots,
        WRITES=writes,
        num_warps=1,
        num_stages=1,
    )
    _device_assert(
        state.overflow_flag == 0,
        "SDM COW physical slab is exhausted",
    )
    output = torch.empty_like(values)
    _packed_cow_update_read_kernel[
        (banks, triton.cdiv(width, block_d))
    ](
        state.shared_initial_memory,
        state.template_indices,
        state.values,
        state.logical_to_physical,
        write_indices,
        resolved_rows,
        first_touches,
        write_weights,
        values,
        input_gate,
        erase_gate,
        forget_log_gate,
        read_indices,
        read_weights,
        output,
        state.slots,
        width,
        WRITES=writes,
        READS=reads,
        BLOCK_W=block_w,
        BLOCK_R=block_r,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    if not collect_diagnostics:
        return output, None
    first_by_sequence = first_touches.sum(dim=-1)
    return output, SDMCopyOnWriteStep(
        first_touches_by_sequence=first_by_sequence,
        repeated_writes_by_sequence=(writes - first_by_sequence),
        allocated_rows_after_step=state.allocated_rows_tensor(),
    )


def _dense_sdm_sparse_step_triton(
    memory: torch.Tensor,
    write_indices: torch.Tensor,
    write_weights: torch.Tensor,
    values: torch.Tensor,
    input_gate: torch.Tensor,
    erase_gate: torch.Tensor,
    forget_log_gate: torch.Tensor,
    read_indices: torch.Tensor,
    read_weights: torch.Tensor,
) -> torch.Tensor:
    if triton is None:  # pragma: no cover - caller proves availability
        raise RuntimeError("Triton is unavailable")
    tensors = (
        memory,
        write_indices,
        write_weights,
        values,
        input_gate,
        erase_gate,
        forget_log_gate,
        read_indices,
        read_weights,
    )
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Triton dense SDM tensors must be contiguous")
    banks, slots, width = memory.shape
    writes = write_indices.shape[1]
    reads = read_indices.shape[1]
    block_w = triton.next_power_of_2(writes)
    block_r = triton.next_power_of_2(reads)
    block_d = 32
    output = torch.empty_like(values)
    _dense_sparse_update_read_kernel[
        (banks, triton.cdiv(width, block_d))
    ](
        memory,
        write_indices,
        write_weights,
        values,
        input_gate,
        erase_gate,
        forget_log_gate,
        read_indices,
        read_weights,
        output,
        slots,
        width,
        WRITES=writes,
        READS=reads,
        BLOCK_W=block_w,
        BLOCK_R=block_r,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return output


@torch.no_grad()
def packed_sdm_cow_step(
    state: PackedSDMCopyOnWriteState,
    write_indices: torch.Tensor,
    write_weights: torch.Tensor,
    values: torch.Tensor,
    input_gate: torch.Tensor,
    forget_log_gate: torch.Tensor,
    read_indices: torch.Tensor,
    read_weights: torch.Tensor,
    *,
    erase_gate: torch.Tensor | None = None,
    backend: str = "auto",
    validate_routes: bool = True,
    collect_diagnostics: bool = True,
) -> tuple[torch.Tensor, SDMCopyOnWriteStep | None]:
    """Apply one exact sparse SDM write and read through the packed overlay.

    Decode is already token-serial.  This operation is fully vectorized over
    live sequences/banks and over each step's sparse W/K routes; it introduces
    no additional sequence-position loop or controller dependency.
    """

    banks, slots, width = state.banks, state.slots, state.width
    device = state.values.device
    _validate_indices(
        write_indices,
        name="write",
        banks=banks,
        slots=slots,
        device=device,
        validate_bounds=validate_routes,
    )
    _validate_indices(
        read_indices,
        name="read",
        banks=banks,
        slots=slots,
        device=device,
        validate_bounds=validate_routes,
    )
    if validate_routes:
        _validate_unique_writes(write_indices)
    if (
        write_weights.shape != write_indices.shape
        or read_weights.shape != read_indices.shape
        or values.shape != (banks, width)
    ):
        raise ValueError("SDM route weights and values do not align")
    tensors = (
        write_weights,
        values,
        input_gate,
        forget_log_gate,
        read_weights,
    ) + (() if erase_gate is None else (erase_gate,))
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("SDM COW step tensors must share one device")
    effective_erase = input_gate if erase_gate is None else erase_gate
    _validate_gates(
        input_gate,
        effective_erase,
        forget_log_gate,
        banks=banks,
        width=width,
    )
    if _use_triton_backend(
        backend,
        tensor=state.values,
        maximum_writes=write_indices.shape[1],
        maximum_reads=read_indices.shape[1],
    ):
        return _packed_sdm_cow_step_triton(
            state,
            write_indices,
            write_weights,
            values,
            input_gate,
            effective_erase,
            forget_log_gate,
            read_indices,
            read_weights,
            collect_diagnostics=collect_diagnostics,
        )

    write_long = write_indices.to(torch.int64)
    bank = torch.arange(banks, device=device, dtype=torch.int64).unsqueeze(1)
    physical_before = state.logical_to_physical.gather(1, write_long)
    first_touch = physical_before < 0

    flat_first = first_touch.flatten()
    ranks = flat_first.to(torch.int32).cumsum(0) - 1
    new_rows = flat_first.to(torch.int32).sum()
    old_free = state.free_count.clone()
    _device_assert(old_free >= new_rows, "SDM COW physical slab is exhausted")
    pop_positions = old_free - 1 - ranks
    safe_pop = pop_positions.clamp(0, state.capacity_rows - 1)
    allocated = state.free_rows[safe_pop.to(torch.int64)].reshape_as(
        physical_before
    )
    physical = torch.where(first_touch, allocated, physical_before)

    base_write = state.shared_initial_memory[
        state.template_indices[:, None], write_long
    ].float()
    safe_before = physical_before.clamp_min(0).to(torch.int64)
    private_write = state.values[safe_before].float()
    current = torch.where(first_touch.unsqueeze(-1), base_write, private_write)

    decay = _expand_gate(forget_log_gate, banks=banks, width=width).exp()
    decayed = current * decay
    write_weight = write_weights.float().unsqueeze(-1)
    retrieved = (write_weight * decayed).sum(dim=1)
    write = _expand_gate(input_gate, banks=banks, width=width).squeeze(1)
    erase = (
        write
        if erase_gate is None
        else _expand_gate(erase_gate, banks=banks, width=width).squeeze(1)
    )
    delta = write * values.float() - erase * retrieved
    updated = decayed + write_weight * delta.unsqueeze(1)

    state.logical_to_physical[bank, write_long] = physical
    state.values[physical.to(torch.int64)] = updated.to(state.values.dtype)
    state.free_count.copy_(old_free - new_rows)

    read_long = read_indices.to(torch.int64)
    read_physical = state.logical_to_physical.gather(1, read_long)
    read_private = read_physical >= 0
    base_read = state.shared_initial_memory[
        state.template_indices[:, None], read_long
    ].float()
    safe_read = read_physical.clamp_min(0).to(torch.int64)
    private_read = state.values[safe_read].float()
    visible = torch.where(read_private.unsqueeze(-1), private_read, base_read)
    reading = (read_weights.float().unsqueeze(-1) * visible).sum(dim=1)

    if not collect_diagnostics:
        return reading.to(values.dtype), None
    first_by_sequence = first_touch.sum(dim=-1)
    return reading.to(values.dtype), SDMCopyOnWriteStep(
        first_touches_by_sequence=first_by_sequence,
        repeated_writes_by_sequence=(write_indices.shape[1] - first_by_sequence),
        allocated_rows_after_step=state.allocated_rows_tensor(),
    )


@torch.no_grad()
def dense_sdm_sparse_step(
    memory: torch.Tensor,
    write_indices: torch.Tensor,
    write_weights: torch.Tensor,
    values: torch.Tensor,
    input_gate: torch.Tensor,
    forget_log_gate: torch.Tensor,
    read_indices: torch.Tensor,
    read_weights: torch.Tensor,
    *,
    erase_gate: torch.Tensor | None = None,
    backend: str = "auto",
    validate_routes: bool = True,
) -> torch.Tensor:
    """Mutate one dense table using the same sparse SDM routes.

    This is the allocator benchmark control.  It avoids an artificial O(ND)
    scan, so the measured delta isolates packed lookup/allocation overhead from
    the retained-state reduction.
    """

    if memory.ndim != 3:
        raise ValueError("dense SDM memory must be [B,N,D]")
    banks, slots, width = memory.shape
    device = memory.device
    _validate_indices(
        write_indices,
        name="write",
        banks=banks,
        slots=slots,
        device=device,
        validate_bounds=validate_routes,
    )
    _validate_indices(
        read_indices,
        name="read",
        banks=banks,
        slots=slots,
        device=device,
        validate_bounds=validate_routes,
    )
    if validate_routes:
        _validate_unique_writes(write_indices)
    if (
        write_weights.shape != write_indices.shape
        or read_weights.shape != read_indices.shape
        or values.shape != (banks, width)
    ):
        raise ValueError("dense SDM route weights and values do not align")
    tensors = (
        write_weights,
        values,
        input_gate,
        forget_log_gate,
        read_weights,
    ) + (() if erase_gate is None else (erase_gate,))
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("dense SDM step tensors must share one device")
    effective_erase = input_gate if erase_gate is None else erase_gate
    _validate_gates(
        input_gate,
        effective_erase,
        forget_log_gate,
        banks=banks,
        width=width,
    )
    if _use_triton_backend(
        backend,
        tensor=memory,
        maximum_writes=write_indices.shape[1],
        maximum_reads=read_indices.shape[1],
    ):
        return _dense_sdm_sparse_step_triton(
            memory,
            write_indices,
            write_weights,
            values,
            input_gate,
            effective_erase,
            forget_log_gate,
            read_indices,
            read_weights,
        )
    write_long = write_indices.to(torch.int64)
    bank = torch.arange(banks, device=device, dtype=torch.int64).unsqueeze(1)
    current = memory[bank, write_long].float()
    decay = _expand_gate(forget_log_gate, banks=banks, width=width).exp()
    decayed = current * decay
    write_weight = write_weights.float().unsqueeze(-1)
    retrieved = (write_weight * decayed).sum(dim=1)
    write = _expand_gate(input_gate, banks=banks, width=width).squeeze(1)
    erase = (
        write
        if erase_gate is None
        else _expand_gate(erase_gate, banks=banks, width=width).squeeze(1)
    )
    delta = write * values.float() - erase * retrieved
    memory[bank, write_long] = (
        decayed + write_weight * delta.unsqueeze(1)
    ).to(memory.dtype)
    read_long = read_indices.to(torch.int64)
    visible = memory[bank, read_long].float()
    return (
        read_weights.float().unsqueeze(-1) * visible
    ).sum(dim=1).to(values.dtype)
