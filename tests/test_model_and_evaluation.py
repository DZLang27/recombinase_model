from __future__ import annotations

from pathlib import Path

import torch

from rec_rank.evaluation import dense_rank_descending, evaluate_score_matrix, export_dense_topk
from rec_rank.model import RecRankModel
from rec_rank.sampler import SampledPair
from rec_rank.training import _augment_training_batch, _is_better, explicit_ranking_loss
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
