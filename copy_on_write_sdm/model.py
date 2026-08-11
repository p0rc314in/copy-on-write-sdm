"""A decoder containing only dense attention and independent native SDM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from .sdm import SDMRouting, SparseDeltaMemory

if TYPE_CHECKING:
    from .inference import FaithfulSDMDecodeState


@dataclass(frozen=True)
class LayerRouting:
    layer: int
    kind: str
    sdm: SDMRouting | None


class ResidualMLP(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        expansion: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = max(1, int(round(width * expansion)))
        self.hidden = hidden
        self.norm = nn.LayerNorm(width)
        self.input = nn.Linear(width, hidden, bias=False)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.input(self.norm(tokens)))
        return self.dropout(self.output(hidden))


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("width must be positive and divisible by heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def project_qkv(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time, _ = tokens.shape
        query, key, value = self.qkv(self.norm(tokens)).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return (
                tensor.reshape(batch, time, self.heads, self.head_width)
                .transpose(1, 2)
                .contiguous()
            )

        return split_heads(query), split_heads(key), split_heads(value)

    def attend_projected(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        causal: bool,
    ) -> torch.Tensor:
        batch, _, query_time, _ = query.shape
        if query.is_cuda:
            flash_eligible = query.dtype in (torch.float16, torch.bfloat16)
            with torch.backends.cuda.sdp_kernel(
                enable_flash=flash_eligible,
                enable_math=not flash_eligible,
                enable_mem_efficient=False,
            ):
                attended = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=causal,
                )
        else:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=causal,
            )
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch,
            query_time,
            self.width,
        )
        return self.output(attended)

    def prefill_with_cache(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = self.project_qkv(tokens)
        return self.attend_projected(query, key, value, causal=True), key, value

    def decode_with_cache(
        self,
        tokens: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        position: int | torch.Tensor,
    ) -> torch.Tensor:
        if tokens.shape[1] != 1 or key_cache.shape != value_cache.shape:
            raise ValueError("dense decode cache is invalid")
        scalar_position = isinstance(position, int)
        positions = (
            None
            if scalar_position
            else torch.as_tensor(position, device=tokens.device, dtype=torch.int64)
        )
        query, key, value = self.project_qkv(tokens)
        if scalar_position:
            key_cache[:, :, position : position + 1].copy_(key)
            value_cache[:, :, position : position + 1].copy_(value)
            return self.attend_projected(
                query,
                key_cache[:, :, : position + 1],
                value_cache[:, :, : position + 1],
                causal=False,
            )
        if positions is None or positions.shape != (tokens.shape[0],):
            raise ValueError("dense decode positions are invalid")
        batch_indices = torch.arange(tokens.shape[0], device=tokens.device)
        key_cache[batch_indices, :, positions, :] = key[:, :, 0, :]
        value_cache[batch_indices, :, positions, :] = value[:, :, 0, :]
        maximum_live = int(positions.max().item()) + 1
        visible = (
            torch.arange(maximum_live, device=tokens.device)[None, :]
            <= positions[:, None]
        )
        mask = torch.zeros(
            tokens.shape[0],
            1,
            1,
            maximum_live,
            device=tokens.device,
            dtype=query.dtype,
        ).masked_fill_(~visible[:, None, None, :], float("-inf"))
        attended = F.scaled_dot_product_attention(
            query,
            key_cache[:, :, :maximum_live],
            value_cache[:, :, :maximum_live],
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        )
        return self.output(
            attended.transpose(1, 2).contiguous().reshape(
                tokens.shape[0],
                1,
                self.width,
            )
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attended, _, _ = self.prefill_with_cache(tokens)
        return attended


class DenseAttentionBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        *,
        mlp_expansion: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = CausalSelfAttention(width, heads, dropout=dropout)
        self.mlp = ResidualMLP(
            width,
            expansion=mlp_expansion,
            dropout=dropout,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(tokens)
        return tokens + self.mlp(tokens)


class SDMDecoderStack(nn.Module):
    """Physical-layer stack with independent native SDM at every B layer."""

    def __init__(
        self,
        *,
        layout: str,
        width: int,
        heads: int,
        slots: int,
        reads: int,
        writes: int,
        memory_heads: int = 1,
        mlp_expansion: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not layout or any(kind not in "AB" for kind in layout):
            raise ValueError("layout must contain only A and B")
        self.layout = layout
        self.width = width
        self.heads = heads
        self.attention_layers = nn.ModuleDict(
            {
                str(layer): DenseAttentionBlock(
                    width,
                    heads,
                    mlp_expansion=mlp_expansion,
                    dropout=dropout,
                )
                for layer, kind in enumerate(layout)
                if kind == "A"
            }
        )
        self.sdm_layers = nn.ModuleDict(
            {
                str(layer): SparseDeltaMemory(
                    width,
                    slots=slots,
                    reads=reads,
                    writes=writes,
                    memory_heads=memory_heads,
                )
                for layer, kind in enumerate(layout)
                if kind == "B"
            }
        )
        self.sdm_mlps = nn.ModuleDict(
            {
                str(layer): ResidualMLP(
                    width,
                    expansion=mlp_expansion,
                    dropout=dropout,
                )
                for layer, kind in enumerate(layout)
                if kind == "B"
            }
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_routing: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[LayerRouting]]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.width:
            raise ValueError(f"tokens must be [B,T,{self.width}]")
        diagnostics: list[LayerRouting] = []
        for physical_layer, kind in enumerate(self.layout):
            if kind == "A":
                tokens = self.attention_layers[str(physical_layer)](tokens)
                if return_routing:
                    diagnostics.append(LayerRouting(physical_layer, kind, None))
                continue
            mixed = self.sdm_layers[str(physical_layer)](
                tokens,
                return_routing=return_routing,
            )
            if return_routing:
                mixed, routing = mixed
                diagnostics.append(LayerRouting(physical_layer, kind, routing))
            tokens = tokens + mixed
            tokens = tokens + self.sdm_mlps[str(physical_layer)](tokens)
        return (tokens, diagnostics) if return_routing else tokens

    def prefill(
        self,
        tokens: torch.Tensor,
        *,
        capacity: int | None = None,
        sdm_capacity_rows: int | tuple[int, ...] | None = None,
        sdm_growth_quantum_rows: int = 0,
    ) -> tuple[torch.Tensor, "FaithfulSDMDecodeState"]:
        from .inference import prefill_faithful_sdm_stack

        return prefill_faithful_sdm_stack(
            self,
            tokens,
            capacity=capacity,
            sdm_capacity_rows=sdm_capacity_rows,
            sdm_growth_quantum_rows=sdm_growth_quantum_rows,
        )

    def decode(
        self,
        tokens: torch.Tensor,
        state: "FaithfulSDMDecodeState",
    ) -> torch.Tensor:
        from .inference import decode_faithful_sdm_stack

        return decode_faithful_sdm_stack(self, tokens, state)


class LanguageModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        maximum_sequence_length: int,
        layout: str,
        width: int,
        heads: int,
        slots: int,
        reads: int,
        writes: int,
        memory_heads: int = 1,
        mlp_expansion: float = 4.0,
        activation_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.maximum_sequence_length = maximum_sequence_length
        self.activation_dtype = activation_dtype
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(maximum_sequence_length, width)
        self.stack = SDMDecoderStack(
            layout=layout,
            width=width,
            heads=heads,
            slots=slots,
            reads=reads,
            writes=writes,
            memory_heads=memory_heads,
            mlp_expansion=mlp_expansion,
        )
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size, bias=False)

    def initialize_role_keyed(self, seed: int) -> None:
        base = seed * 100_000
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(base + 101)
            nn.init.normal_(
                self.token_embedding.weight,
                mean=0.0,
                std=self.stack.width**-0.5,
            )
            torch.manual_seed(base + 102)
            nn.init.normal_(
                self.position_embedding.weight,
                mean=0.0,
                std=self.stack.width**-0.5,
            )
            torch.manual_seed(base + 103)
            nn.init.normal_(
                self.output.weight,
                mean=0.0,
                std=self.stack.width**-0.5,
            )
        self.final_norm.reset_parameters()
        for physical_layer, kind in enumerate(self.stack.layout):
            if kind == "A":
                block = self.stack.attention_layers[str(physical_layer)]
                _reset_children(block.mlp, base + 1_000 + physical_layer)
                _reset_children(block.attention, base + 2_000 + physical_layer)
            else:
                self.stack.sdm_layers[str(physical_layer)].reset_role_keyed(
                    base + 6_000 + physical_layer
                )
                _reset_children(
                    self.stack.sdm_mlps[str(physical_layer)],
                    base + 1_000 + physical_layer,
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_routing: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[LayerRouting]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        time = input_ids.shape[1]
        if time > self.maximum_sequence_length:
            raise ValueError("sequence exceeds maximum_sequence_length")
        positions = torch.arange(time, device=input_ids.device)
        tokens = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions).unsqueeze(0)
        ).to(self.activation_dtype)
        stacked = self.stack(tokens, return_routing=return_routing)
        if return_routing:
            hidden, routing = stacked
            return self.output(self.final_norm(hidden)), routing
        return self.output(self.final_norm(stacked))

    def prefill(
        self,
        input_ids: torch.Tensor,
        *,
        capacity: int | None = None,
        sdm_capacity_rows: int | tuple[int, ...] | None = None,
        sdm_growth_quantum_rows: int = 0,
    ) -> tuple[torch.Tensor, "FaithfulSDMDecodeState"]:
        if input_ids.ndim != 2 or input_ids.shape[1] <= 0:
            raise ValueError("prefill input_ids must be non-empty [B,T]")
        time = input_ids.shape[1]
        selected_capacity = self.maximum_sequence_length if capacity is None else capacity
        if time > selected_capacity or selected_capacity > self.maximum_sequence_length:
            raise ValueError("prefill capacity is invalid")
        positions = torch.arange(time, device=input_ids.device)
        tokens = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions).unsqueeze(0)
        ).to(self.activation_dtype)
        tokens, state = self.stack.prefill(
            tokens,
            capacity=selected_capacity,
            sdm_capacity_rows=sdm_capacity_rows,
            sdm_growth_quantum_rows=sdm_growth_quantum_rows,
        )
        return self.output(self.final_norm(tokens)), state

    def decode(
        self,
        input_ids: torch.Tensor,
        state: "FaithfulSDMDecodeState",
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape != (state.batch, 1):
            raise ValueError(f"decode input_ids must be [{state.batch},1]")
        positions = state.current_positions()
        if bool((positions >= self.maximum_sequence_length).any()):
            raise ValueError("sequence exceeds maximum_sequence_length")
        tokens = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions).unsqueeze(1)
        ).to(self.activation_dtype)
        tokens = self.stack.decode(tokens, state)
        return self.output(self.final_norm(tokens))

    @torch.no_grad()
    def admit(
        self,
        input_ids: torch.Tensor,
        state: "FaithfulSDMDecodeState",
        *,
        sdm_growth_quantum_rows: int | None = None,
    ) -> torch.Tensor:
        growth = (
            state.sdm_states[0].growth_quantum_rows
            if sdm_growth_quantum_rows is None and state.sdm_states
            else 0 if sdm_growth_quantum_rows is None else sdm_growth_quantum_rows
        )
        logits, incoming = self.prefill(
            input_ids,
            capacity=state.capacity,
            sdm_growth_quantum_rows=growth,
        )
        state.admit(incoming)
        return logits


def _reset_children(module: nn.Module, seed: int) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for child in module.modules():
            if child is module:
                continue
            reset = getattr(child, "reset_parameters", None)
            if callable(reset):
                reset()
