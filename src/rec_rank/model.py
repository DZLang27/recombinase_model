from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("gamma_init 必须大于 0")
    return value + math.log(-math.expm1(-value))


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(values), p=2, dim=-1)


class ResidualProjectionBlock(nn.Module):
    """Pre-norm residual MLP block with a learnable LayerScale vector."""

    def __init__(
        self,
        width: int,
        expansion_dim: int,
        dropout: float,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.network = nn.Sequential(
            nn.Linear(width, expansion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion_dim, width),
            nn.Dropout(dropout),
        )
        # Per-channel LayerScale lets every output direction learn its own
        # residual strength while the small initialization keeps early
        # optimization close to the stable input projection.
        self.layer_scale = nn.Parameter(torch.full((width,), layer_scale_init))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.layer_scale * self.network(self.norm(values))


class ResidualProjectionHead(nn.Module):
    """Higher-capacity projection into a shared cosine retrieval space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        expansion_dim: int,
        residual_blocks: int,
        dropout: float,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        if residual_blocks < 1:
            raise ValueError("projection_residual_blocks 必须至少为 1")
        if expansion_dim < output_dim:
            raise ValueError("projection_expansion_dim 不能小于 projection_dim")
        if layer_scale_init < 0:
            raise ValueError("projection_layer_scale_init 不能为负数")
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, output_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualProjectionBlock(
                    output_dim,
                    expansion_dim,
                    dropout,
                    layer_scale_init,
                )
                for _ in range(residual_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(self.input_norm(values))
        for block in self.blocks:
            projected = block(projected)
        return F.normalize(self.output_norm(projected), p=2, dim=-1)


class RecRankModel(nn.Module):
    def __init__(
        self,
        protein_dim: int = 640,
        dna_dim: int = 512,
        hidden_dim: int = 512,
        projection_dim: int = 256,
        dropout: float = 0.1,
        gamma_init: float = 5.0,
        gamma_epsilon: float = 1e-6,
        projection_head_type: str = "legacy",
        projection_residual_blocks: int = 2,
        projection_expansion_dim: int | None = None,
        projection_layer_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        projection_head_type = projection_head_type.lower()
        if projection_head_type == "legacy":
            self.protein_projection = ProjectionHead(
                protein_dim, hidden_dim, projection_dim, dropout
            )
            self.dna_projection = ProjectionHead(
                dna_dim, hidden_dim, projection_dim, dropout
            )
        elif projection_head_type == "residual":
            expansion_dim = (
                int(projection_expansion_dim)
                if projection_expansion_dim is not None
                else 2 * projection_dim
            )
            self.protein_projection = ResidualProjectionHead(
                protein_dim,
                projection_dim,
                expansion_dim,
                projection_residual_blocks,
                dropout,
                projection_layer_scale_init,
            )
            self.dna_projection = ResidualProjectionHead(
                dna_dim,
                projection_dim,
                expansion_dim,
                projection_residual_blocks,
                dropout,
                projection_layer_scale_init,
            )
            projection_expansion_dim = expansion_dim
        else:
            raise ValueError("projection_head_type 必须是 legacy 或 residual")
        self.raw_gamma = nn.Parameter(torch.tensor(_inverse_softplus(gamma_init - gamma_epsilon)))
        self.bias = nn.Parameter(torch.zeros(()))
        self.gamma_epsilon = gamma_epsilon
        self.architecture = {
            "protein_dim": protein_dim,
            "dna_dim": dna_dim,
            "hidden_dim": hidden_dim,
            "projection_dim": projection_dim,
            "dropout": dropout,
            "gamma_init": gamma_init,
            "gamma_epsilon": gamma_epsilon,
            "projection_head_type": projection_head_type,
            "projection_residual_blocks": projection_residual_blocks,
            "projection_expansion_dim": projection_expansion_dim,
            "projection_layer_scale_init": projection_layer_scale_init,
        }

    @property
    def gamma(self) -> torch.Tensor:
        # softplus 没有硬上限；配合单独的 weight decay 和监控避免无约束放大。
        return F.softplus(self.raw_gamma) + self.gamma_epsilon

    def project_proteins(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.protein_projection(embeddings)

    def project_dnas(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.dna_projection(embeddings)

    @staticmethod
    def cosine_scores(dna_vectors: torch.Tensor, protein_vectors: torch.Tensor) -> torch.Tensor:
        return dna_vectors @ protein_vectors.T

    def pair_cosine(self, protein_embeddings: torch.Tensor, dna_embeddings: torch.Tensor) -> torch.Tensor:
        protein_vectors = self.project_proteins(protein_embeddings)
        dna_vectors = self.project_dnas(dna_embeddings)
        return (protein_vectors * dna_vectors).sum(dim=-1)

    def pair_logits(self, protein_embeddings: torch.Tensor, dna_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cosine = self.pair_cosine(protein_embeddings, dna_embeddings)
        return self.gamma * cosine + self.bias, cosine

    def loss(
        self,
        protein_embeddings: torch.Tensor,
        dna_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, cosine = self.pair_logits(protein_embeddings, dna_embeddings)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss, logits, cosine
