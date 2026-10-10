from __future__ import annotations

from pathlib import Path

import torch

from rec_rank.evaluation import dense_rank_descending, evaluate_score_matrix, export_dense_topk
from rec_rank.embeddings import EmbeddingStore
from rec_rank.model import RecRankModel
from rec_rank.model_io import model_from_checkpoint
from rec_rank.sampler import DnaRankingGroup, SampledPair
from rec_rank.training import (
    _augment_training_batch,
    _is_better,
    dna_centric_ranking_loss,
    explicit_ranking_loss,
    grouped_protein_ranking_loss,
    multi_positive_clip_loss,
    reverse_complement_consistency_loss,
)
from rec_rank.utils import read_tsv


def test_gamma_is_positive_and_loss_can_decrease() -> None:
    torch.manual_seed(7)
    model = RecRankModel(protein_dim=4, dna_dim=3, hidden_dim=8, projection_dim=5, dropout=0.0)
    proteins = torch.randn(8, 4)
    dnas = torch.randn(8, 3)
    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    initial = float(model.loss(proteins, dnas, labels)[0].detach())
    for _ in range(40):
        optimizer.zero_grad()
        loss, _, _ = model.loss(proteins, dnas, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    final = float(model.loss(proteins, dnas, labels)[0].detach())
    assert float(model.gamma.detach()) > 0
    assert final < initial


def test_residual_projection_heads_are_independent_normalized_and_trainable() -> None:
    torch.manual_seed(11)
    model = RecRankModel(
        protein_dim=4,
        dna_dim=3,
        hidden_dim=5,
        projection_dim=6,
        dropout=0.0,
        projection_head_type="residual",
        projection_residual_blocks=2,
        projection_expansion_dim=12,
        projection_layer_scale_init=0.1,
    )
    protein_vectors = model.project_proteins(torch.randn(4, 4))
    dna_vectors = model.project_dnas(torch.randn(4, 3))
    assert protein_vectors.shape == (4, 6)
    assert dna_vectors.shape == (4, 6)
    assert torch.allclose(protein_vectors.norm(dim=-1), torch.ones(4), atol=1e-6)
    assert torch.allclose(dna_vectors.norm(dim=-1), torch.ones(4), atol=1e-6)
    assert (
        model.protein_projection.blocks[0].layer_scale.data_ptr()
        != model.dna_projection.blocks[0].layer_scale.data_ptr()
    )

    loss = (protein_vectors * dna_vectors).sum()
    loss.backward()
    for projection in (model.protein_projection, model.dna_projection):
        assert all(block.layer_scale.grad is not None for block in projection.blocks)


def test_old_and_residual_checkpoint_architectures_load() -> None:
    legacy = RecRankModel(
        protein_dim=4,
        dna_dim=3,
        hidden_dim=8,
        projection_dim=5,
        dropout=0.0,
    )
    # Historical checkpoints do not contain any of the new head fields.
    old_architecture = {
        key: value
        for key, value in legacy.architecture.items()
        if not key.startswith("projection_") or key == "projection_dim"
    }
    old_architecture["hidden_dim"] = 8
    restored_legacy = model_from_checkpoint(
        {"model_architecture": old_architecture, "model_state": legacy.state_dict()},
        torch.device("cpu"),
    )
    assert isinstance(restored_legacy, RecRankModel)
    assert restored_legacy.architecture["projection_head_type"] == "legacy"

    residual = RecRankModel(
        protein_dim=4,
        dna_dim=3,
        hidden_dim=6,
        projection_dim=6,
        dropout=0.0,
        projection_head_type="residual",
        projection_residual_blocks=2,
        projection_expansion_dim=12,
        projection_layer_scale_init=0.1,
    )
    restored_residual = model_from_checkpoint(
        {
            "model_architecture": residual.architecture,
            "model_state": residual.state_dict(),
        },
        torch.device("cpu"),
    )
    assert restored_residual.architecture == residual.architecture
    assert torch.allclose(
        restored_residual.project_proteins(torch.randn(2, 4)).norm(dim=-1),
        torch.ones(2),
        atol=1e-6,
    )


def test_dense_rank_preserves_all_ties_and_multi_positive(tmp_path: Path) -> None:
    scores = torch.tensor([[0.9, 0.9, 0.8, 0.7, 0.7], [0.1, 0.2, 0.3, 0.4, 0.5]])
    ranks = dense_rank_descending(scores)
    assert ranks[0].tolist() == [1, 1, 2, 3, 3]
    result = evaluate_score_matrix(
        scores,
        ["D1", "D2"],
        ["P1", "P2", "P3", "P4", "P5"],
        {"D1": {"P3", "P5"}, "D2": {"P4"}},
        recall_ks=(1, 2, 10),
    )
    assert result.query_rows[0]["best_positive_rank"] == 2
    assert result.query_rows[1]["best_positive_rank"] == 2
    assert result.metrics["recall_at_1"] == 0.0
    assert result.metrics["recall_at_2"] == 1.0
    export_path = tmp_path / "top1.tsv"
    export_dense_topk(
        export_path,
        scores[:1],
        ranks[:1],
        ["D1"],
        ["P1", "P2", "P3", "P4", "P5"],
        top_k=1,
        checkpoint_id="ckpt",
        candidate_index_id="index",
    )
    exported = list(read_tsv(export_path))
    assert [row["protein_key"] for row in exported] == ["P1", "P2"]


def test_explicit_ranking_loss_and_novel_checkpoint_selection() -> None:
    batch = [
        SampledPair("P1", "D1", 1, "attP_pos", 0),
        SampledPair("P1", "D2", 0, "attP_neg", 0),
        SampledPair("P2", "D3", 1, "attB_pos", 1),
        SampledPair("P2", "D4", 0, "attB_neg", 1),
    ]
    cosine = torch.tensor([0.7, 0.6, 0.2, 0.5], requires_grad=True)
    loss, triplets, active_fraction = explicit_ranking_loss(cosine, batch, margin=0.2)
    assert triplets == 2
    assert torch.isclose(loss, torch.tensor(0.3))
    assert active_fraction == 1.0
    loss.backward()
    assert cosine.grad is not None

    rc_map = {"D1": "R1", "D2": "R2", "D3": "R3", "D4": "R4"}
    first = _augment_training_batch(batch, rc_map, 0.5, seed=42, epoch=3)
    second = _augment_training_batch(batch, rc_map, 0.5, seed=42, epoch=3)
    assert first == second
    augmented_batch, dna_keys, added = _augment_training_batch(
        batch, rc_map, 1.0, seed=42, epoch=3
    )
    assert len(augmented_batch) == 2 * len(batch)
    assert dna_keys[: len(batch)] == [pair.dna_key for pair in batch]
    assert dna_keys[len(batch) :] == [rc_map[pair.dna_key] for pair in batch]
    assert added == len(batch)

    best = {
        "epoch": 2,
        "metrics": {
            "novel_dna_recall_at_10": 0.1,
            "novel_dna_mrr": 0.2,
            "novel_dna_median_rank": 20,
            "recall_at_10": 0.3,
        },
    }
    candidate = {
        "novel_dna_recall_at_10": 0.2,
        "novel_dna_mrr": 0.1,
        "novel_dna_median_rank": 100,
        "recall_at_10": 0.2,
    }
    order = [
        "novel_dna_recall_at_10",
        "novel_dna_mrr",
        "novel_dna_median_rank",
        "recall_at_10",
        "earlier_epoch",
    ]
    assert _is_better(candidate, 3, best, order)


def test_multi_positive_clip_loss_uses_full_cross_matrix_and_unknowns() -> None:
    torch.manual_seed(19)
    model = RecRankModel(
        protein_dim=4,
        dna_dim=3,
        hidden_dim=8,
        projection_dim=5,
        dropout=0.0,
    )
    protein_store = EmbeddingStore(
        keys=["P1", "P2"],
        embeddings=torch.randn(2, 4),
        manifest={},
    )
    dna_store = EmbeddingStore(
        keys=["D1", "D2", "D3"],
        embeddings=torch.randn(3, 3),
        manifest={},
    )
    loss, stats = multi_positive_clip_loss(
        model,
        ["P1", "P2", "P1"],
        ["D1", "D2", "D3", "D2"],
        protein_store,
        dna_store,
        {"P1": {"D1", "D2"}, "P2": {"D2"}},
        {"P1": {"D3"}},
        torch.device("cpu"),
    )
    assert torch.isfinite(loss)
    assert stats["matrix_cells"] == 6
    assert stats["positive_cells"] == 3
    assert stats["explicit_negative_cells"] == 1
    assert stats["unknown_negative_cells"] == 2
    assert stats["active_proteins"] == 2
    assert stats["active_dnas"] == 2
    loss.backward()
    assert model.raw_gamma.grad is not None
    assert next(model.protein_projection.parameters()).grad is not None
    assert next(model.dna_projection.parameters()).grad is not None


def test_v12_group_balanced_dna_ranking_and_rc_consistency() -> None:
    torch.manual_seed(9)
    model = RecRankModel(
        protein_dim=4,
        dna_dim=3,
        hidden_dim=8,
        projection_dim=5,
        dropout=0.0,
    )
    protein_store = EmbeddingStore(
        keys=["P1", "P2", "P3"],
        embeddings=torch.randn(3, 4),
        manifest={},
    )
    dna_store = EmbeddingStore(
        keys=["D1", "R1"],
        embeddings=torch.randn(2, 3),
        manifest={},
    )
    groups = [
        DnaRankingGroup(
            dna_key="D1",
            positive_protein_keys=("P1", "P2"),
            negative_protein_keys=("P3",),
        )
    ]
    loss, triplets, comparisons, active, margin = dna_centric_ranking_loss(
        model,
        groups,
        protein_store,
        dna_store,
        {"D1": "R1"},
        torch.device("cpu"),
        0.1,
        smooth=True,
        include_reverse_complement=True,
    )
    assert triplets == 2
    assert comparisons == 4
    assert 0.0 <= active <= 1.0
    assert isinstance(margin, float)
    rc_loss, rc_cosine = reverse_complement_consistency_loss(
        model,
        ["D1", "D1"],
        dna_store,
        {"D1": "R1"},
        torch.device("cpu"),
    )
    (loss + 0.1 * rc_loss).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert -1.0 <= rc_cosine <= 1.0

    protein_batch = [
        SampledPair("P1", "D1", 1, "attP_pos", 0),
        SampledPair("P1", "D2", 1, "attB_pos", 0),
        SampledPair("P1", "D3", 0, "attP_neg", 0),
        SampledPair("P2", "D4", 1, "attP_pos", 1),
        SampledPair("P2", "D5", 0, "attP_neg", 1),
    ]
    protein_loss, protein_triplets, protein_groups, protein_active = (
        grouped_protein_ranking_loss(
            torch.tensor([0.8, 0.6, 0.7, 0.1, 0.4], requires_grad=True),
            protein_batch,
            0.1,
            smooth=True,
        )
    )
    assert protein_triplets == 3
    assert protein_groups == 2
    assert 0.0 <= protein_active <= 1.0
    assert protein_loss.item() > 0


def test_v14_residual_head_with_hinge_dna_ranking() -> None:
    torch.manual_seed(14)
    model = RecRankModel(
        protein_dim=4,
        dna_dim=3,
        hidden_dim=6,
        projection_dim=6,
        dropout=0.0,
        projection_head_type="residual",
        projection_residual_blocks=2,
        projection_expansion_dim=12,
        projection_layer_scale_init=0.1,
    )
    protein_store = EmbeddingStore(
        keys=["P1", "P2", "P3"],
        embeddings=torch.randn(3, 4),
        manifest={},
    )
    dna_store = EmbeddingStore(
        keys=["D1", "R1"],
        embeddings=torch.randn(2, 3),
        manifest={},
    )
    groups = [
        DnaRankingGroup(
            dna_key="D1",
            positive_protein_keys=("P1", "P2"),
            negative_protein_keys=("P3",),
        )
    ]
    loss, triplets, comparisons, active, margin = dna_centric_ranking_loss(
        model,
        groups,
        protein_store,
        dna_store,
        {"D1": "R1"},
        torch.device("cpu"),
        0.1,
        smooth=False,
        include_reverse_complement=True,
    )
    assert triplets == 2
    assert comparisons == 4
    assert 0.0 <= active <= 1.0
    assert isinstance(margin, float)
    (0.1 * loss).backward()
    assert model.protein_projection.blocks[0].layer_scale.grad is not None
    assert model.dna_projection.blocks[0].layer_scale.grad is not None
