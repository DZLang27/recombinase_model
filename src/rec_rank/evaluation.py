from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .utils import read_tsv, write_tsv


def dense_rank_descending(scores: torch.Tensor) -> torch.Tensor:
    """对最后一维做降序 dense rank；完全相同的浮点值共享名次。"""
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
        squeeze = True
    elif scores.ndim == 2:
        squeeze = False
    else:
        raise ValueError("scores 必须是一维或二维 tensor")
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    sorted_scores = torch.gather(scores, 1, order)
    new_value = torch.ones_like(sorted_scores, dtype=torch.long)
    if sorted_scores.shape[1] > 1:
        new_value[:, 1:] = (sorted_scores[:, 1:] != sorted_scores[:, :-1]).long()
    sorted_ranks = torch.cumsum(new_value, dim=1)
    ranks = torch.empty_like(sorted_ranks)
    ranks.scatter_(1, order, sorted_ranks)
    return ranks.squeeze(0) if squeeze else ranks


@dataclass
class RetrievalEvaluation:
    metrics: dict[str, float | int]
    query_rows: list[dict[str, Any]]
    ranks: torch.Tensor


def load_truth(path: str | Path) -> dict[str, set[str]]:
    truth: dict[str, set[str]] = {}
    for row in read_tsv(path):
        values = {value for value in row["positive_protein_keys"].split(";") if value}
        if not values:
            raise ValueError(f"truth 中 query 没有正蛋白: {row['dna_key']}")
        truth[row["dna_key"]] = values
    return truth


def load_query_metadata(path: str | Path) -> dict[str, dict[str, str]]:
    return {row["dna_key"]: row for row in read_tsv(path)}


def _metrics_from_rows(rows: list[dict[str, Any]], recall_ks: Iterable[int]) -> dict[str, float | int]:
    if not rows:
        return {"queries": 0, **{f"recall_at_{k}": math.nan for k in recall_ks}, "mrr": math.nan, "median_rank": math.nan}
    best_ranks = np.asarray([row["best_positive_rank"] for row in rows], dtype=np.float64)
    metrics: dict[str, float | int] = {"queries": len(rows)}
    for k in recall_ks:
        metrics[f"recall_at_{k}"] = float(np.mean(best_ranks <= k))
    metrics["mrr"] = float(np.mean(1.0 / best_ranks))
    metrics["median_rank"] = float(np.median(best_ranks))
    metrics["mean_rank"] = float(np.mean(best_ranks))
    return metrics


def evaluate_score_matrix(
    scores: torch.Tensor,
    query_keys: list[str],
    candidate_keys: list[str],
    truth: dict[str, set[str]],
    recall_ks: tuple[int, ...] = (1, 5, 10),
    query_metadata: dict[str, dict[str, str]] | None = None,
) -> RetrievalEvaluation:
    if scores.shape != (len(query_keys), len(candidate_keys)):
        raise ValueError("score matrix shape 与 query/candidate 数量不一致")
    candidate_index = {key: index for index, key in enumerate(candidate_keys)}
    ranks = dense_rank_descending(scores)
    rows: list[dict[str, Any]] = []
    for query_index, query_key in enumerate(query_keys):
        if query_key not in truth:
            raise KeyError(f"truth 缺少 query: {query_key}")
        positive_keys = truth[query_key]
        missing = positive_keys - candidate_index.keys()
        if missing:
            raise ValueError(f"query {query_key} 的正蛋白不在候选库: {sorted(missing)[:3]}")
        positive_indices = [candidate_index[key] for key in sorted(positive_keys)]
        positive_ranks = ranks[query_index, positive_indices]
        best_local = int(torch.argmin(positive_ranks).item())
        best_candidate_index = positive_indices[best_local]
        row: dict[str, Any] = {
            "dna_key": query_key,
            "positive_count": len(positive_keys),
            "best_positive_key": candidate_keys[best_candidate_index],
            "best_positive_rank": int(ranks[query_index, best_candidate_index].item()),
            "best_positive_cosine": float(scores[query_index, best_candidate_index].item()),
            "top1_protein_key": candidate_keys[int(torch.argmax(scores[query_index]).item())],
            "top1_cosine": float(torch.max(scores[query_index]).item()),
        }
        if query_metadata and query_key in query_metadata:
            row["is_novel_dna"] = int(query_metadata[query_key].get("is_novel_dna", "0"))
        rows.append(row)

    metrics = _metrics_from_rows(rows, recall_ks)
    if query_metadata is not None:
        novel_rows = [row for row in rows if row.get("is_novel_dna") == 1]
        novel_metrics = _metrics_from_rows(novel_rows, recall_ks)
        metrics.update({f"novel_dna_{key}": value for key, value in novel_metrics.items()})
    return RetrievalEvaluation(metrics=metrics, query_rows=rows, ranks=ranks)


def recall_curve(query_rows: list[dict[str, Any]], max_k: int = 50) -> list[dict[str, float | int]]:
    ranks = np.asarray([row["best_positive_rank"] for row in query_rows], dtype=np.float64)
    return [
        {"k": k, "recall": float(np.mean(ranks <= k)) if ranks.size else math.nan}
        for k in range(1, max_k + 1)
    ]


def export_query_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dna_key",
        "positive_count",
        "best_positive_key",
        "best_positive_rank",
        "best_positive_cosine",
        "top1_protein_key",
        "top1_cosine",
        "is_novel_dna",
    ]
    write_tsv(path, rows, fields)


def export_dense_topk(
    path: str | Path,
    scores: torch.Tensor,
    ranks: torch.Tensor,
    query_keys: list[str],
    candidate_keys: list[str],
    top_k: int,
    checkpoint_id: str,
    candidate_index_id: str,
    sequences: dict[str, str] | None = None,
    candidate_metadata: dict[str, dict[str, str]] | None = None,
) -> None:
    rows = []
    for query_index, query_key in enumerate(query_keys):
        selected = torch.nonzero(ranks[query_index] <= top_k, as_tuple=False).flatten()
        selected = selected[
            torch.argsort(scores[query_index, selected], descending=True, stable=True)
        ]
        for candidate_index_value in selected.tolist():
            row = {
                "query_id": query_key,
                "query_sequence": (sequences or {}).get(query_key, ""),
                "protein_key": candidate_keys[candidate_index_value],
                "cosine": float(scores[query_index, candidate_index_value].item()),
                "dense_rank": int(ranks[query_index, candidate_index_value].item()),
                "checkpoint_id": checkpoint_id,
                "candidate_index_id": candidate_index_id,
            }
            metadata = (candidate_metadata or {}).get(candidate_keys[candidate_index_value], {})
            row["protein_record_ids"] = metadata.get("record_ids", "")
            row["protein_cluster_ids"] = metadata.get("cluster_ids", "")
            rows.append(row)
    write_tsv(
        path,
        rows,
        [
            "query_id",
            "query_sequence",
            "protein_key",
            "protein_record_ids",
            "protein_cluster_ids",
            "cosine",
            "dense_rank",
            "checkpoint_id",
            "candidate_index_id",
        ],
    )
