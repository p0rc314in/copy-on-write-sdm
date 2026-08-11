from __future__ import annotations

import unittest

import torch

from copy_on_write_sdm.allocator import (
    PackedSDMCopyOnWriteState,
    dense_sdm_sparse_step,
    packed_sdm_cow_step,
)


class AllocatorParityTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.banks = 3
        self.slots = 16
        self.width = 8
        self.initial = torch.randn(1, self.slots, self.width)
        self.dense = self.initial.expand(self.banks, -1, -1).clone().float()
        self.packed = PackedSDMCopyOnWriteState.allocate(
            self.initial,
            banks=self.banks,
            capacity_rows=self.banks * self.slots,
            state_dtype=torch.float32,
        )

    def step(self, write_indices: torch.Tensor) -> tuple[torch.Tensor, object]:
        writes = write_indices.shape[1]
        write_weights = torch.softmax(torch.randn(self.banks, writes), dim=-1)
        values = torch.randn(self.banks, self.width)
        input_gate = torch.sigmoid(torch.randn(self.banks))
        forget_log_gate = -torch.rand(self.banks)
        read_indices = torch.tensor(
            [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]], dtype=torch.int64
        )
        read_weights = torch.softmax(torch.randn(self.banks, 4), dim=-1)
        dense_read = dense_sdm_sparse_step(
            self.dense,
            write_indices,
            write_weights,
            values,
            input_gate,
            forget_log_gate,
            read_indices,
            read_weights,
            backend="torch",
        )
        packed_read, diagnostics = packed_sdm_cow_step(
            self.packed,
            write_indices,
            write_weights,
            values,
            input_gate,
            forget_log_gate,
            read_indices,
            read_weights,
            backend="torch",
        )
        self.assertTrue(torch.equal(dense_read, packed_read))
        self.assertTrue(torch.equal(self.dense, self.packed.materialize_dense()))
        return packed_read, diagnostics

    def test_first_touch_repeated_write_and_untouched_reads(self) -> None:
        _, first = self.step(
            torch.tensor([[2, 5], [1, 7], [4, 10]], dtype=torch.int64)
        )
        self.assertEqual(int(first.first_touches_by_sequence.sum()), 6)
        self.assertEqual(int(self.packed.allocated_rows_tensor()), 6)

        _, repeated = self.step(
            torch.tensor([[2, 5], [1, 7], [4, 10]], dtype=torch.int64)
        )
        self.assertEqual(int(repeated.first_touches_by_sequence.sum()), 0)
        self.assertEqual(int(self.packed.allocated_rows_tensor()), 6)
        self.assertTrue(
            torch.equal(
                self.packed.materialize_dense()[0, 15],
                self.initial[0, 15].float(),
            )
        )

    def test_pooled_growth_preserves_state_and_accounts_for_copy(self) -> None:
        state = PackedSDMCopyOnWriteState.allocate(
            torch.zeros(1, 16, 4),
            banks=2,
            capacity_rows=4,
            growth_quantum_rows=4,
        )
        before = state.materialize_dense().clone()

        state.grow_capacity(5)

        self.assertEqual(state.capacity_rows, 8)
        self.assertEqual(state.growth_rows_added, 4)
        self.assertEqual(state.growth_rows_copied, 4)
        self.assertTrue(torch.equal(state.materialize_dense(), before))
        state.validate_invariants()


if __name__ == "__main__":
    unittest.main()
