from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as functional
from .config import FusionConfig, MODALITY_ORDER, ADAPTER_HIDDEN_DIMENSIONS
from .data import FeatureInputs
DEVICE = torch.device('cpu')

class ModalityAdapter(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, output_dimension: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, output_dimension),
        )

    def forward(self, values: Tensor) -> Tensor:
        return functional.normalize(self.network(values), p=2, dim=1)


class ReliabilityGate(nn.Module):
    """Reliability gate: a learned bias if quality width is zero, otherwise Linear(q, 1)."""

    def __init__(self, quality_dimension: int):
        super().__init__()
        self.quality_dimension = int(quality_dimension)
        if self.quality_dimension:
            self.network: nn.Module | None = nn.Linear(self.quality_dimension, 1)
            self.bias = None
        else:
            self.network = None
            self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, quality: Tensor, available: Tensor) -> Tensor:
        if self.network is None:
            logits = self.bias.expand(available.shape[0], 1)
        else:
            logits = self.network(quality)
        return torch.sigmoid(logits) * available.to(dtype=logits.dtype).unsqueeze(1)


@dataclass(frozen=True)
class FusionDiagnostics:
    gates: Tensor
    embedding_available: Tensor
    adapted_tokens: Tensor
    final_tokens: Tensor
    attention_weights: Tensor | None = None


class FiveModalityConcatFusionEncoder(nn.Module):
    """V1/V1.1: adapt, gate, concatenate, and project."""

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.adapters = nn.ModuleDict(
            {
                name: ModalityAdapter(
                    config.input_dimensions[name],
                    ADAPTER_HIDDEN_DIMENSIONS[name],
                    config.adapter_dimension,
                )
                for name in MODALITY_ORDER
            }
        )
        self.reliability_gates = nn.ModuleDict(
            {
                name: ReliabilityGate(config.quality_dimension_map[name])
                for name in MODALITY_ORDER
            }
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.fusion_input_dimension, config.fusion_hidden_dimension),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dimension, config.output_dimension),
        )

    def forward(
        self,
        matrices: Mapping[str, Tensor],
        content_masks: Tensor,
        source_masks: Tensor,
        qualities: Mapping[str, Tensor],
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, FusionDiagnostics]:
        adapted_tokens: list[Tensor] = []
        gated_tokens: list[Tensor] = []
        gates: list[Tensor] = []

        for position, name in enumerate(MODALITY_ORDER):
            available = content_masks[:, position].bool()
            adapted = self.adapters[name](matrices[name])
            gate = self.reliability_gates[name](qualities[name], available)
            adapted_tokens.append(adapted)
            gated_tokens.append(gate * adapted)
            gates.append(gate)

        mask_features = torch.cat(
            [content_masks.to(gated_tokens[0].dtype), source_masks.to(gated_tokens[0].dtype)],
            dim=1,
        )
        fused_input = torch.cat([*gated_tokens, mask_features], dim=1)
        raw = self.fusion(fused_input)
        embedding = functional.normalize(raw, p=2, dim=1)
        embedding_available = content_masks.bool().any(dim=1)
        embedding = embedding * embedding_available.to(embedding.dtype).unsqueeze(1)

        if return_diagnostics:
            return embedding, FusionDiagnostics(
                gates=torch.cat(gates, dim=1),
                embedding_available=embedding_available,
                adapted_tokens=torch.stack(adapted_tokens, dim=1),
                final_tokens=torch.stack(gated_tokens, dim=1),
                attention_weights=None,
            )
        return embedding


class FiveTokenSelfAttentionBlock(nn.Module):
    def __init__(self, token_dimension: int, heads: int, feedforward_dimension: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=token_dimension,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(token_dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(token_dimension, feedforward_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dimension, token_dimension),
        )
        self.norm2 = nn.LayerNorm(token_dimension)

    def forward(self, tokens: Tensor, key_padding_mask: Tensor) -> tuple[Tensor, Tensor]:
        safe_mask = key_padding_mask.clone()
        all_missing = safe_mask.all(dim=1)
        if torch.any(all_missing):
            safe_mask[all_missing, 0] = False

        attention_output, attention_weights = self.attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=safe_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        values = self.norm1(tokens + attention_output)
        values = self.norm2(values + self.feed_forward(values))
        values = values.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        # The temporary sentinel key above prevents all-masked softmax NaNs.
        # It is not a real observation and must not appear in diagnostics.
        attention_weights = attention_weights.masked_fill(all_missing[:, None, None, None], 0.0)
        return values, attention_weights


class FiveTokenSelfAttentionFusionEncoder(nn.Module):
    """V2: five adapted modality tokens interact through self-attention before final fusion."""

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.adapters = nn.ModuleDict(
            {
                name: ModalityAdapter(
                    config.input_dimensions[name],
                    ADAPTER_HIDDEN_DIMENSIONS[name],
                    config.adapter_dimension,
                )
                for name in MODALITY_ORDER
            }
        )
        self.reliability_gates = nn.ModuleDict(
            {
                name: ReliabilityGate(config.quality_dimension_map[name])
                for name in MODALITY_ORDER
            }
        )
        self.modality_embeddings = nn.Parameter(
            torch.empty(len(MODALITY_ORDER), config.adapter_dimension)
        )
        nn.init.normal_(self.modality_embeddings, mean=0.0, std=0.02)
        self.attention_block = FiveTokenSelfAttentionBlock(
            token_dimension=config.adapter_dimension,
            heads=config.attention_heads,
            feedforward_dimension=config.attention_feedforward_dimension,
            dropout=config.dropout,
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.fusion_input_dimension, config.fusion_hidden_dimension),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dimension, config.output_dimension),
        )

    def forward(
        self,
        matrices: Mapping[str, Tensor],
        content_masks: Tensor,
        source_masks: Tensor,
        qualities: Mapping[str, Tensor],
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, FusionDiagnostics]:
        adapted_tokens: list[Tensor] = []
        gates: list[Tensor] = []
        for position, name in enumerate(MODALITY_ORDER):
            available = content_masks[:, position].bool()
            adapted_tokens.append(self.adapters[name](matrices[name]))
            gates.append(self.reliability_gates[name](qualities[name], available))

        adapted_stack = torch.stack(adapted_tokens, dim=1)  # [B, 5, 128]
        gate_matrix = torch.cat(gates, dim=1)  # [B, 5]
        available = content_masks.bool()

        identified_tokens = adapted_stack + self.modality_embeddings.unsqueeze(0)
        identified_tokens = identified_tokens * available.to(identified_tokens.dtype).unsqueeze(-1)
        attended_tokens, attention_weights = self.attention_block(
            identified_tokens,
            key_padding_mask=~available,
        )

        # Reliability is applied per record_id after interaction so it controls each modality's
        # final contribution to the flattened fusion representation.
        final_tokens = attended_tokens * gate_matrix.unsqueeze(-1)

        mask_features = torch.cat(
            [content_masks.to(final_tokens.dtype), source_masks.to(final_tokens.dtype)],
            dim=1,
        )
        fused_input = torch.cat([final_tokens.flatten(start_dim=1), mask_features], dim=1)
        raw = self.fusion(fused_input)
        embedding = functional.normalize(raw, p=2, dim=1)
        embedding_available = available.any(dim=1)
        embedding = embedding * embedding_available.to(embedding.dtype).unsqueeze(1)

        if return_diagnostics:
            return embedding, FusionDiagnostics(
                gates=gate_matrix,
                embedding_available=embedding_available,
                adapted_tokens=adapted_stack,
                final_tokens=final_tokens,
                attention_weights=attention_weights,
            )
        return embedding


def tensor_batch(
    aligned_inputs: FeatureInputs,
    indices: Sequence[int] | np.ndarray,
    *,
    device: torch.device = DEVICE,
) -> tuple[dict[str, Tensor], Tensor, Tensor, dict[str, Tensor]]:
    rows = np.asarray(indices, dtype=np.int64)
    matrices = {
        name: torch.as_tensor(aligned_inputs.matrices[name][rows], dtype=torch.float32, device=device)
        for name in MODALITY_ORDER
    }
    content_masks = torch.as_tensor(aligned_inputs.content_masks[rows], device=device)
    source_masks = torch.as_tensor(aligned_inputs.source_masks[rows], device=device)
    qualities = {
        name: torch.as_tensor(aligned_inputs.qualities[name][rows], dtype=torch.float32, device=device)
        for name in MODALITY_ORDER
    }
    return matrices, content_masks, source_masks, qualities


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
