"""Faithful Sparse Delta Memory copy-on-write allocation."""

from .allocator import PackedSDMCopyOnWriteState
from .model import LanguageModel, SDMDecoderStack
from .sdm import (
    SDMCopyOnWriteAccounting,
    SDMRouting,
    SparseDeltaMemory,
    aggregate_sdm_copy_on_write_accounting,
    product_key_routes,
    sdm_copy_on_write_accounting,
)

__all__ = [
    "LanguageModel",
    "PackedSDMCopyOnWriteState",
    "SDMCopyOnWriteAccounting",
    "SDMDecoderStack",
    "SDMRouting",
    "SparseDeltaMemory",
    "aggregate_sdm_copy_on_write_accounting",
    "product_key_routes",
    "sdm_copy_on_write_accounting",
]
