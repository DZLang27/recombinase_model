from __future__ import annotations

import platform
import random
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .data import load_groups
from .embeddings import EmbeddingStore, load_embedding_store, merge_embedding_stores
from .evaluation import (
    RetrievalEvaluation,
    evaluate_score_matrix,
    export_dense_topk,
    export_query_rows,
    load_query_metadata,
    load_truth,
    recall_curve,
)
from .model import RecRankModel
from .model_io import load_checkpoint, project_store
from .sampler import (
    DnaRankingBatchSampler,
    DnaRankingGroup,
    EpochBatchSampler,
    TripleEpochBatchSampler,
    TripleSample,
    flatten_triple_batch,
    load_dna_ranking_groups,
    load_triples,
)
from .full_library import (
    _build_truths,
    _candidate_metadata,
    _merge_candidate_stores,
    evaluate_model_full_library,
)
from .utils import (
    append_csv,
    append_jsonl,
    atomic_json_dump,
    atomic_torch_save,
    canonical_json,
    dump_yaml,
    make_run_id,
    read_tsv,
    set_global_seed,
    sha256_file,
    sha256_text,
    utc_now_iso,
    write_tsv,
)


HISTORY_FIELDS = [
    "epoch",
    "global_step",
    "train_loss",
    "train_bce_loss",
    "train_ranking_loss",
    "train_ranking_active_fraction",
    "train_ranking_triplets",
    "train_protein_ranking_loss",
    "train_protein_ranking_active_fraction",
    "train_protein_ranking_triplets",
    "train_dna_ranking_loss",
    "train_dna_ranking_active_fraction",
    "train_dna_ranking_groups",
    "train_dna_ranking_triplets",
    "train_dna_ranking_comparisons",
    "train_dna_ranking_margin_mean",
    "train_rc_consistency_loss",
    "train_rc_embedding_cosine_mean",
    "train_rc_augmentation_fraction",
    "train_rc_added_pairs",
    "train_clip_loss",
    "train_clip_protein_to_dna_loss",
    "train_clip_dna_to_protein_loss",
    "train_clip_matrix_cells",
    "train_clip_positive_cells",
    "train_clip_explicit_negative_cells",
    "train_clip_unknown_negative_cells",
    "train_clip_active_proteins",
    "train_clip_active_dnas",
    "train_clip_positive_cosine_mean",
    "train_clip_explicit_negative_cosine_mean",
    "train_clip_unknown_negative_cosine_mean",
    "train_tuple_loss",
    "train_tuple_protein_to_pair_loss",
    "train_tuple_pair_to_protein_loss",
    "train_tuple_matrix_cells",
    "train_tuple_positive_cells",
    "train_tuple_unknown_cells",
    "train_tuple_positive_score_mean",
    "train_tuple_unknown_score_mean",
    "train_structured_negative_loss",
    "train_structured_negative_active_fraction",
    "train_structured_negative_comparisons",
    "train_structured_positive_score_mean",
    "train_structured_corrupted_score_mean",
    "train_structured_margin_mean",
    "train_positive_loss",
    "train_negative_loss",
    "train_positive_cosine_mean",
    "train_positive_cosine_median",
    "train_positive_cosine_q05",
    "train_positive_cosine_q95",
    "train_negative_cosine_mean",
    "train_negative_cosine_median",
    "train_negative_cosine_q05",
    "train_negative_cosine_q95",
    "train_pair_margin_mean",
    "test_loss",
    "test_positive_loss",
    "test_negative_loss",
    "test_positive_cosine_mean",
    "test_negative_cosine_mean",
    "queries",
    "candidates",
    "novel_dna_queries",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "median_rank",
    "novel_dna_recall_at_10",
    "novel_dna_mrr",
    "novel_dna_median_rank",
    "exact_recall_at_10",
    "exact_novel_dna_recall_at_10",
    "gamma",
    "bias",
    "projection_lr",
    "gamma_lr",
    "gradient_norm_mean",
    "gradient_norm_max",
    "gradient_clip_fraction",
    "cumulative_unique_pairs",
    "cumulative_pair_coverage",
    "gpu_max_memory_gb",
    "epoch_seconds",
    "is_best",
]


def _quantiles(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_q05": float("nan"),
            f"{prefix}_q95": float("nan"),
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_q05": float(np.quantile(array, 0.05)),
        f"{prefix}_q95": float(np.quantile(array, 0.95)),
    }


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _build_checkpoint(
    model: RecRankModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    global_step: int,
    best: dict[str, Any],
    config: dict[str, Any],
    fingerprints: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "saved_at": utc_now_iso(),
        "epoch": epoch,
        "global_step": global_step,
        "model_architecture": model.architecture,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best": best,
        "resolved_config": config,
        "config_sha256": sha256_text(canonical_json(config)),
        "fingerprints": fingerprints,
        "rng_state": _capture_rng_state(),
        "sampler_state": {"next_epoch": epoch + 1},
    }


def _selection_key(
    metrics: dict[str, Any], epoch: int, selection_tiebreak: list[str]
) -> tuple[float, ...]:
    values: list[float] = []
    for name in selection_tiebreak:
        if name == "earlier_epoch":
            values.append(-float(epoch))
            continue
        if name not in metrics:
            raise KeyError(f"checkpoint 选择指标不存在: {name}")
        value = float(metrics[name])
        if "rank" in name and "recall" not in name:
            value = -value
        values.append(value)
    return tuple(values)


def _is_better(
    metrics: dict[str, Any],
    epoch: int,
    best: dict[str, Any] | None,
    selection_tiebreak: list[str] | None = None,
) -> bool:
    if best is None:
        return True
    order = selection_tiebreak or ["recall_at_10", "mrr", "median_rank", "earlier_epoch"]
    return _selection_key(metrics, epoch, order) > _selection_key(
        best["metrics"], int(best["epoch"]), order
    )


def explicit_ranking_loss(
    cosine: torch.Tensor,
    batch: list[Any],
    margin: float,
) -> tuple[torch.Tensor, int, float]:
    """只比较同一蛋白内有明确正/负标签的样本，不把 unknown 当负样本。"""
    by_group: dict[int, dict[int, list[int]]] = defaultdict(lambda: {0: [], 1: []})
    for index, pair in enumerate(batch):
        by_group[pair.group_index][pair.label].append(index)
    violations: list[torch.Tensor] = []
    for labels in by_group.values():
        for positive_index in labels[1]:
            for negative_index in labels[0]:
                violations.append(
                    F.relu(margin - cosine[positive_index] + cosine[negative_index])
                )
    if not violations:
        return cosine.sum() * 0.0, 0, 0.0
    stacked = torch.stack(violations)
    active_fraction = float((stacked.detach() > 0).float().mean().cpu())
    return stacked.mean(), len(violations), active_fraction


def grouped_protein_ranking_loss(
    cosine: torch.Tensor,
    batch: list[Any],
    margin: float,
    *,
    smooth: bool,
) -> tuple[torch.Tensor, int, int, float]:
    """按蛋白先求组内均值再跨组平均，避免高标注度蛋白支配 loss。"""
    by_group: dict[int, dict[int, list[int]]] = defaultdict(lambda: {0: [], 1: []})
    for index, pair in enumerate(batch):
        by_group[pair.group_index][pair.label].append(index)
    group_losses: list[torch.Tensor] = []
    raw_violations: list[torch.Tensor] = []
    triplet_count = 0
    for labels in by_group.values():
        if not labels[0] or not labels[1]:
            continue
        positive = cosine[labels[1]][:, None]
        negative = cosine[labels[0]][None, :]
        raw = margin - positive + negative
        losses = F.softplus(raw) if smooth else F.relu(raw)
        group_losses.append(losses.mean())
        raw_violations.append(raw.reshape(-1))
        triplet_count += raw.numel()
    if not group_losses:
        return cosine.sum() * 0.0, 0, 0, 0.0
    raw_values = torch.cat(raw_violations)
    active_fraction = float((raw_values.detach() > 0).float().mean().cpu())
    return torch.stack(group_losses).mean(), triplet_count, len(group_losses), active_fraction


def multi_positive_clip_loss(
    model: RecRankModel,
    protein_keys: list[str],
    dna_keys: list[str],
    protein_store: EmbeddingStore,
    dna_store: EmbeddingStore,
    positive_dnas_by_protein: dict[str, set[str]],
    negative_dnas_by_protein: dict[str, set[str]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """对 batch 全交叉矩阵计算对称 multi-positive InfoNCE。

    已知正 pair 使用均匀 multi-hot target；显式负 pair 与 unknown pair 都进入
    softmax 分母。没有 batch 内正样本的行/列只作为候选负样本，不作为 query。
    """
    unique_proteins = list(dict.fromkeys(protein_keys))
    unique_dnas = list(dict.fromkeys(dna_keys))
    if not unique_proteins or not unique_dnas:
        anchor = next(model.parameters())
        return anchor.sum() * 0.0, {
            "protein_to_dna_loss": 0.0,
            "dna_to_protein_loss": 0.0,
            "matrix_cells": 0,
            "positive_cells": 0,
            "explicit_negative_cells": 0,
            "unknown_negative_cells": 0,
            "active_proteins": 0,
            "active_dnas": 0,
            "positive_cosine_sum": 0.0,
            "explicit_negative_cosine_sum": 0.0,
            "unknown_negative_cosine_sum": 0.0,
        }

    dna_indices = {key: index for index, key in enumerate(unique_dnas)}
    positive_mask_cpu = torch.zeros(
        (len(unique_proteins), len(unique_dnas)), dtype=torch.bool
    )
    explicit_negative_mask_cpu = torch.zeros_like(positive_mask_cpu)
    for protein_index, protein_key in enumerate(unique_proteins):
        for dna_key in positive_dnas_by_protein.get(protein_key, set()):
            dna_index = dna_indices.get(dna_key)
            if dna_index is not None:
                positive_mask_cpu[protein_index, dna_index] = True
        for dna_key in negative_dnas_by_protein.get(protein_key, set()):
            dna_index = dna_indices.get(dna_key)
            if dna_index is not None:
                explicit_negative_mask_cpu[protein_index, dna_index] = True
    # 方向等价扩增后可能同时出现正、负记录；正标签优先，避免把合法正例送入分母标签。
    explicit_negative_mask_cpu &= ~positive_mask_cpu

    positive_mask = positive_mask_cpu.to(device=device)
    explicit_negative_mask = explicit_negative_mask_cpu.to(device=device)
    unknown_negative_mask = ~(positive_mask | explicit_negative_mask)

    protein_raw = protein_store.rows(unique_proteins, device=device).float()
    dna_raw = dna_store.rows(unique_dnas, device=device).float()
    protein_vectors = model.project_proteins(protein_raw)
    dna_vectors = model.project_dnas(dna_raw)
    cosine = protein_vectors @ dna_vectors.T
    logits = model.gamma * cosine

    row_positive_counts = positive_mask.sum(dim=1)
    column_positive_counts = positive_mask.sum(dim=0)
    active_rows = row_positive_counts > 0
    active_columns = column_positive_counts > 0
    if not bool(active_rows.any()) or not bool(active_columns.any()):
        raise ValueError("CLIP-style batch 中没有可用的已知正 pair")

    row_targets = positive_mask[active_rows].to(logits.dtype)
    row_targets = row_targets / row_positive_counts[active_rows, None].to(logits.dtype)
    protein_to_dna_loss = -(
        F.log_softmax(logits[active_rows], dim=1) * row_targets
    ).sum(dim=1).mean()

    column_targets = positive_mask[:, active_columns].T.to(logits.dtype)
    column_targets = column_targets / column_positive_counts[active_columns, None].to(
        logits.dtype
    )
    dna_to_protein_loss = -(
        F.log_softmax(logits[:, active_columns].T, dim=1) * column_targets
    ).sum(dim=1).mean()
    loss = 0.5 * (protein_to_dna_loss + dna_to_protein_loss)

    detached_cosine = cosine.detach().float()
    positive_cells = int(positive_mask_cpu.sum())
    explicit_negative_cells = int(explicit_negative_mask_cpu.sum())
    matrix_cells = positive_mask_cpu.numel()
    unknown_negative_cells = matrix_cells - positive_cells - explicit_negative_cells

    def masked_sum(mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return 0.0
        return float(detached_cosine[mask].sum().cpu())

    return loss, {
        "protein_to_dna_loss": float(protein_to_dna_loss.detach().float().cpu()),
        "dna_to_protein_loss": float(dna_to_protein_loss.detach().float().cpu()),
        "matrix_cells": matrix_cells,
        "positive_cells": positive_cells,
        "explicit_negative_cells": explicit_negative_cells,
        "unknown_negative_cells": unknown_negative_cells,
        "active_proteins": int(active_rows.sum().detach().cpu()),
        "active_dnas": int(active_columns.sum().detach().cpu()),
        "positive_cosine_sum": masked_sum(positive_mask),
        "explicit_negative_cosine_sum": masked_sum(explicit_negative_mask),
        "unknown_negative_cosine_sum": masked_sum(unknown_negative_mask),
    }


def normalized_softmin(
    first: torch.Tensor, second: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Smooth minimum whose value is unchanged when both inputs are equal."""
    if temperature <= 0:
        raise ValueError("triple_softmin_temperature 必须大于 0")
    stacked = torch.stack((first, second), dim=-1)
    return -temperature * (
        torch.logsumexp(-stacked / temperature, dim=-1)
        - torch.log(torch.tensor(2.0, device=stacked.device, dtype=stacked.dtype))
    )


def _canonical_site_key(dna_key: str, rc_map: dict[str, str]) -> str:
    return min(dna_key, rc_map[dna_key])


def triple_pair_key(triple: TripleSample, rc_map: dict[str, str]) -> tuple[str, str]:
    """Represent the positive site pair without attP/attB order or strand direction."""
    first, second = sorted(
        (
            _canonical_site_key(triple.attp_pos_key, rc_map),
            _canonical_site_key(triple.attb_pos_key, rc_map),
        )
    )
    return first, second


def _orientation_ensemble_matrix(
    model: RecRankModel,
    protein_vectors: torch.Tensor,
    dna_keys: list[str],
    dna_store: EmbeddingStore,
    rc_map: dict[str, str],
    device: torch.device,
) -> torch.Tensor:
    original = model.project_dnas(dna_store.rows(dna_keys, device=device).float())
    reverse = model.project_dnas(
        dna_store.rows([rc_map[key] for key in dna_keys], device=device).float()
    )
    direction_invariant = F.normalize(original + reverse, p=2, dim=-1)
    return protein_vectors @ direction_invariant.T


def triple_tuple_contrastive_loss(
    model: RecRankModel,
    triples: list[TripleSample],
    protein_store: EmbeddingStore,
    dna_store: EmbeddingStore,
    rc_map: dict[str, str],
    positive_pairs_by_protein: dict[str, set[tuple[str, str]]],
    device: torch.device,
    softmin_temperature: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Contrast enzymes against intact, orientation-invariant (attP, attB) pairs."""
    if not triples:
        anchor = next(model.parameters())
        return anchor.sum() * 0.0, {
            "protein_to_pair_loss": 0.0,
            "pair_to_protein_loss": 0.0,
            "matrix_cells": 0,
            "positive_cells": 0,
            "unknown_cells": 0,
            "positive_score_sum": 0.0,
            "unknown_score_sum": 0.0,
        }
    protein_keys = [triple.protein_key for triple in triples]
    if len(set(protein_keys)) != len(protein_keys):
        raise ValueError("triple-aware batch 必须保证每个蛋白最多出现一次")
    protein_vectors = model.project_proteins(
        protein_store.rows(protein_keys, device=device).float()
    )
    attp_scores = _orientation_ensemble_matrix(
        model,
        protein_vectors,
        [triple.attp_pos_key for triple in triples],
        dna_store,
        rc_map,
        device,
    )
    attb_scores = _orientation_ensemble_matrix(
        model,
        protein_vectors,
        [triple.attb_pos_key for triple in triples],
        dna_store,
        rc_map,
        device,
    )
    joint_scores = normalized_softmin(attp_scores, attb_scores, softmin_temperature)

    candidate_pair_keys = [triple_pair_key(triple, rc_map) for triple in triples]
    positive_mask_cpu = torch.zeros(
        (len(triples), len(triples)), dtype=torch.bool
    )
    for protein_index, protein_key in enumerate(protein_keys):
        known_pairs = positive_pairs_by_protein.get(protein_key, set())
        for pair_index, pair_key in enumerate(candidate_pair_keys):
            positive_mask_cpu[protein_index, pair_index] = pair_key in known_pairs
    positive_mask = positive_mask_cpu.to(device=device)
    row_counts = positive_mask.sum(dim=1)
    column_counts = positive_mask.sum(dim=0)
    active_rows = row_counts > 0
    active_columns = column_counts > 0
    if not bool(active_rows.all()):
        raise ValueError("triple-aware batch 存在没有已知正位点对的蛋白")
    if not bool(active_columns.all()):
        raise ValueError("triple-aware batch 存在没有已知正蛋白的位点对")

    logits = model.gamma * joint_scores
    row_targets = positive_mask.to(logits.dtype) / row_counts[:, None].to(logits.dtype)
    protein_to_pair_loss = -(
        F.log_softmax(logits, dim=1) * row_targets
    ).sum(dim=1).mean()
    column_targets = positive_mask.T.to(logits.dtype) / column_counts[:, None].to(
        logits.dtype
    )
    pair_to_protein_loss = -(
        F.log_softmax(logits.T, dim=1) * column_targets
    ).sum(dim=1).mean()
    loss = 0.5 * (protein_to_pair_loss + pair_to_protein_loss)

    detached = joint_scores.detach().float()
    unknown_mask = ~positive_mask
    positive_cells = int(positive_mask_cpu.sum())
    unknown_cells = int(unknown_mask.sum().detach().cpu())
    return loss, {
        "protein_to_pair_loss": float(protein_to_pair_loss.detach().float().cpu()),
        "pair_to_protein_loss": float(pair_to_protein_loss.detach().float().cpu()),
        "matrix_cells": positive_mask_cpu.numel(),
        "positive_cells": positive_cells,
        "unknown_cells": unknown_cells,
        "positive_score_sum": float(detached[positive_mask].sum().cpu()),
        "unknown_score_sum": float(detached[unknown_mask].sum().cpu()),
    }


def triple_structured_negative_loss(
    model: RecRankModel,
    triples: list[TripleSample],
    protein_store: EmbeddingStore,
    dna_store: EmbeddingStore,
    rc_map: dict[str, str],
    device: torch.device,
    margin: float,
    softmin_temperature: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Rank each intact positive pair above its one-site explicit corruptions."""
    if not triples:
        anchor = next(model.parameters())
        return anchor.sum() * 0.0, {
            "comparisons": 0,
            "active_fraction": 0.0,
            "positive_score_sum": 0.0,
            "corrupted_score_sum": 0.0,
            "margin_sum": 0.0,
        }
    protein_keys = [triple.protein_key for triple in triples]
    protein_vectors = model.project_proteins(
        protein_store.rows(protein_keys, device=device).float()
    )

    def aligned_scores(dna_keys: list[str]) -> torch.Tensor:
        original = model.project_dnas(
            dna_store.rows(dna_keys, device=device).float()
        )
        reverse = model.project_dnas(
            dna_store.rows([rc_map[key] for key in dna_keys], device=device).float()
        )
        direction_invariant = F.normalize(original + reverse, p=2, dim=-1)
        return (protein_vectors * direction_invariant).sum(dim=-1)

    attp_positive = aligned_scores([triple.attp_pos_key for triple in triples])
    attb_positive = aligned_scores([triple.attb_pos_key for triple in triples])
    attp_negative = aligned_scores([triple.attp_neg_key for triple in triples])
    attb_negative = aligned_scores([triple.attb_neg_key for triple in triples])
    positive_joint = normalized_softmin(
        attp_positive, attb_positive, softmin_temperature
    )
    corrupted_attp = normalized_softmin(
        attp_negative, attb_positive, softmin_temperature
    )
    corrupted_attb = normalized_softmin(
        attp_positive, attb_negative, softmin_temperature
    )
    corrupted = torch.cat((corrupted_attp, corrupted_attb))
    repeated_positive = torch.cat((positive_joint, positive_joint))
    raw = margin - repeated_positive + corrupted
    losses = F.relu(raw)
    score_margins = repeated_positive - corrupted
    return losses.mean(), {
        "comparisons": int(losses.numel()),
        "active_fraction": float((raw.detach() > 0).float().mean().cpu()),
        "positive_score_sum": float(repeated_positive.detach().float().sum().cpu()),
        "corrupted_score_sum": float(corrupted.detach().float().sum().cpu()),
        "margin_sum": float(score_margins.detach().float().sum().cpu()),
    }


def dna_centric_ranking_loss(
    model: RecRankModel,
    groups: list[DnaRankingGroup],
    protein_store: EmbeddingStore,
    dna_store: EmbeddingStore,
    rc_map: dict[str, str] | None,
    device: torch.device,
    margin: float,
    *,
    smooth: bool,
    include_reverse_complement: bool,
) -> tuple[torch.Tensor, int, int, float, float]:
    """给定 DNA，仅在显式正、负蛋白之间构造检索方向的排序监督。"""
    if not groups:
        anchor = next(model.parameters())
        return anchor.sum() * 0.0, 0, 0, 0.0, 0.0
    dna_keys = [group.dna_key for group in groups]
    dna_raw = dna_store.rows(dna_keys, device=device).float()
    dna_vectors = model.project_dnas(dna_raw)
    rc_vectors: torch.Tensor | None = None
    if include_reverse_complement:
        if rc_map is None:
            raise ValueError("DNA-centric ranking 启用反向互补时缺少 RC 映射")
        rc_keys = [rc_map[key] for key in dna_keys]
        rc_raw = dna_store.rows(rc_keys, device=device).float()
        rc_vectors = model.project_dnas(rc_raw)

    group_losses: list[torch.Tensor] = []
    raw_violations: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    triplet_count = 0
    comparison_count = 0
    for index, group in enumerate(groups):
        protein_keys = [*group.positive_protein_keys, *group.negative_protein_keys]
        protein_raw = protein_store.rows(protein_keys, device=device).float()
        protein_vectors = model.project_proteins(protein_raw)
        positive_count = len(group.positive_protein_keys)
        orientation_vectors = [dna_vectors[index]]
        if rc_vectors is not None:
            orientation_vectors.append(rc_vectors[index])
        orientation_losses: list[torch.Tensor] = []
        for dna_vector in orientation_vectors:
            scores = protein_vectors @ dna_vector
            positive = scores[:positive_count][:, None]
            negative = scores[positive_count:][None, :]
            raw = margin - positive + negative
            losses = F.softplus(raw) if smooth else F.relu(raw)
            orientation_losses.append(losses.mean())
            raw_violations.append(raw.reshape(-1))
            margins.append((positive - negative).reshape(-1))
            comparison_count += raw.numel()
        group_losses.append(torch.stack(orientation_losses).mean())
        triplet_count += group.triplet_count

    raw_values = torch.cat(raw_violations)
    margin_values = torch.cat(margins)
    return (
        torch.stack(group_losses).mean(),
        triplet_count,
        comparison_count,
        float((raw_values.detach() > 0).float().mean().cpu()),
        float(margin_values.detach().float().mean().cpu()),
    )


def reverse_complement_consistency_loss(
    model: RecRankModel,
    dna_keys: list[str],
    dna_store: EmbeddingStore,
    rc_map: dict[str, str],
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """在投影空间约束 DNA 与其反向互补方向一致，按 unique DNA 等权。"""
    unique_keys = sorted(set(dna_keys))
    original = dna_store.rows(unique_keys, device=device).float()
    reverse = dna_store.rows([rc_map[key] for key in unique_keys], device=device).float()
    original_vectors = model.project_dnas(original)
    reverse_vectors = model.project_dnas(reverse)
    cosine = (original_vectors * reverse_vectors).sum(dim=-1)
    return (1.0 - cosine).mean(), float(cosine.detach().float().mean().cpu())


def _load_rc_map(path: Path) -> dict[str, str]:
    mapping = {
        row["dna_key"]: row["reverse_complement_dna_key"] for row in read_tsv(path)
    }
    if not mapping:
        raise ValueError(f"反向互补映射为空: {path}")
    return mapping


def _augment_training_batch(
    batch: list[Any], rc_map: dict[str, str], fraction: float, seed: int, epoch: int
) -> tuple[list[Any], list[str], int]:
    """保留全部原样本，并为确定性选中的样本追加反向互补副本。"""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("reverse_complement_augmentation_fraction 必须在 [0, 1]")
    augmented_batch = list(batch)
    dna_keys = [pair.dna_key for pair in batch]
    rc_count = 0
    cutoff = int(fraction * (2**64 - 1))
    for pair in batch:
        use_rc = fraction == 1.0
        if 0.0 < fraction < 1.0:
            token = f"{seed}:{epoch}:{pair.protein_key}:{pair.dna_key}:{pair.category}"
            use_rc = int(sha256_text(token)[:16], 16) <= cutoff
        if use_rc:
            # SampledPair 保留原 dna_key，确保 pair coverage 仍以原始监督关系计数；
            # 真正送入模型的 DNA embedding key 使用对应的反向互补 key。
            augmented_batch.append(pair)
            dna_keys.append(rc_map[pair.dna_key])
            rc_count += 1
    return augmented_batch, dna_keys, rc_count


_project_store = project_store


def _load_pairs(path: str | Path) -> list[tuple[str, str, float]]:
    return [(row["protein_key"], row["dna_key"], float(row["label"])) for row in read_tsv(path)]


def _evaluate_labeled_pairs(
    model: RecRankModel,
    pairs: list[tuple[str, str, float]],
    protein_store: EmbeddingStore,
    dna_store: EmbeddingStore,
    device: torch.device,
    batch_size: int = 4096,
) -> tuple[dict[str, float], list[float], list[float]]:
    total_loss = 0.0
    positive_losses: list[float] = []
    negative_losses: list[float] = []
    positive_cosines: list[float] = []
    negative_cosines: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            proteins = protein_store.rows([item[0] for item in batch], device=device).float()
            dnas = dna_store.rows([item[1] for item in batch], device=device).float()
            labels = torch.tensor([item[2] for item in batch], device=device, dtype=torch.float32)
            logits, cosine = model.pair_logits(proteins, dnas)
            losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            total_loss += float(losses.sum().item())
            positive_mask = labels == 1
            negative_mask = ~positive_mask
            positive_losses.extend(losses[positive_mask].float().cpu().tolist())
            negative_losses.extend(losses[negative_mask].float().cpu().tolist())
            positive_cosines.extend(cosine[positive_mask].float().cpu().tolist())
            negative_cosines.extend(cosine[negative_mask].float().cpu().tolist())
    metrics = {
        "test_loss": total_loss / len(pairs),
        "test_positive_loss": float(np.mean(positive_losses)),
        "test_negative_loss": float(np.mean(negative_losses)),
        "test_positive_cosine_mean": float(np.mean(positive_cosines)),
        "test_negative_cosine_mean": float(np.mean(negative_cosines)),
    }
    return metrics, positive_cosines, negative_cosines


def _retrieval_evaluation(
    model: RecRankModel,
    candidate_store: EmbeddingStore,
    test_dna_store: EmbeddingStore,
    query_keys: list[str],
    candidate_keys: list[str],
    truth: dict[str, set[str]],
    query_metadata: dict[str, dict[str, str]],
    device: torch.device,
    recall_ks: tuple[int, ...],
) -> tuple[RetrievalEvaluation, torch.Tensor]:
    candidate_vectors = _project_store(model, candidate_store, candidate_keys, "protein", device)
    query_vectors = _project_store(model, test_dna_store, query_keys, "dna", device)
    scores = (query_vectors.to(device) @ candidate_vectors.to(device).T).float().cpu()
    evaluation = evaluate_score_matrix(
        scores,
        query_keys,
        candidate_keys,
        truth,
        recall_ks=recall_ks,
        query_metadata=query_metadata,
    )
    return evaluation, scores


def _random_baseline(
    query_keys: list[str],
    candidate_keys: list[str],
    truth: dict[str, set[str]],
    query_metadata: dict[str, dict[str, str]],
    seed: int,
    recall_ks: tuple[int, ...],
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_scores = torch.rand((len(query_keys), len(candidate_keys)), generator=generator)
    return evaluate_score_matrix(
        random_scores,
        query_keys,
        candidate_keys,
        truth,
        recall_ks=recall_ks,
        query_metadata=query_metadata,
    ).metrics


def _write_epoch_artifacts(
    run_dir: Path,
    epoch: int,
    evaluation: RetrievalEvaluation,
    positive_cosines: list[float],
    negative_cosines: list[float],
    recall_curve_max_k: int,
) -> None:
    evaluation_dir = run_dir / "evaluation"
    atomic_torch_save(
        evaluation_dir / f"labeled_cosines_epoch_{epoch:03d}.pt",
        {
            "positive": torch.tensor(positive_cosines, dtype=torch.float32),
            "negative": torch.tensor(negative_cosines, dtype=torch.float32),
        },
    )
    write_tsv(
        evaluation_dir / f"recall_curve_epoch_{epoch:03d}.tsv",
        recall_curve(evaluation.query_rows, recall_curve_max_k),
        ["k", "recall"],
    )


def _export_full_top_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    write_tsv(
        path,
        rows,
        [
            "query_id",
            "query_sequence",
            "protein_key",
            "protein_record_ids",
            "protein_cluster_ids",
            "candidate_source",
            "cosine",
            "dense_rank",
            "checkpoint_id",
            "candidate_library_id",
        ],
    )


def _log_tensorboard(writer: Any, row: dict[str, Any], epoch: int) -> None:
    for key, value in row.items():
        if key in {"epoch", "is_best"} or value in {"", None}:
            continue
        if isinstance(value, (int, float)) and np.isfinite(value):
            writer.add_scalar(key.replace("_", "/", 1), value, epoch)


def _runtime_manifest(config: dict[str, Any], fingerprints: dict[str, Any], run_id: str) -> dict[str, Any]:
    evaluation = config["evaluation"]
    return {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "seed": config["training"]["seed"],
        "config_sha256": sha256_text(canonical_json(config)),
        "fingerprints": fingerprints,
        "primary_metric": evaluation.get("primary_metric", "recall_at_10"),
        "candidate_library": evaluation.get("candidate_library", "test_only"),
        "truth_mode": evaluation.get("truth_mode", "exact"),
        "reverse_complement_ensemble": bool(
            evaluation.get("reverse_complement_ensemble", False)
        ),
        "test_usage_notice": "测试集在本项目决策中用于逐 epoch 模型选择，因此不是未触碰的最终独立估计。",
    }


def train(
    config: dict[str, Any],
    device: torch.device,
    run_dir: str | Path | None = None,
    resume: bool = False,
) -> Path:
    training_cfg = config["training"]
    evaluation_cfg = config["evaluation"]
    data_cfg = config["data"]
    runtime_cfg = config["runtime"]
    seed = int(training_cfg["seed"])
    set_global_seed(seed)

    processed = Path(data_cfg["processed_dir"])
    run_root = Path(runtime_cfg["run_root"])
    if run_dir is None:
        run_path = run_root / make_run_id(config, seed)
    else:
        run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=resume)
    for relative in ["logs/tensorboard", "checkpoints", "evaluation", "figures", "exports"]:
        (run_path / relative).mkdir(parents=True, exist_ok=True)

    train_protein_store = load_embedding_store(runtime_cfg["protein_embedding_cache"])
    train_dna_primary_store = load_embedding_store(runtime_cfg["dna_embedding_cache"])
    test_candidate_store = load_embedding_store(runtime_cfg["test_candidate_embedding_cache"])
    test_dna_store = load_embedding_store(runtime_cfg["test_dna_embedding_cache"])

    rc_augmentation_fraction = float(
        training_cfg.get("reverse_complement_augmentation_fraction", 0.0)
    )
    train_rc_map: dict[str, str] | None = None
    train_dna_store = train_dna_primary_store
    if rc_augmentation_fraction > 0:
        train_rc_map = _load_rc_map(processed / "train_dnas_rc_map.tsv")
        missing_rc_keys = sorted(set(train_rc_map.values()) - set(train_dna_primary_store.keys))
        if missing_rc_keys:
            rc_store = load_embedding_store(runtime_cfg["train_rc_embedding_cache"])
            if not rc_store.contains_all(missing_rc_keys):
                raise ValueError("训练反向互补 embedding cache 不完整")
            train_dna_store = merge_embedding_stores(
                [train_dna_primary_store, rc_store], ["train_dna", "train_dna_rc"]
            )

    full_library_mode = evaluation_cfg.get("candidate_library") == "train_test_union"
    if full_library_mode:
        candidate_store, candidate_audit = _merge_candidate_stores(
            runtime_cfg["protein_embedding_cache"],
            runtime_cfg["test_candidate_embedding_cache"],
        )
        (
            query_keys,
            exact_truth,
            direction_truth,
            query_metadata,
            truth_audit,
        ) = _build_truths(processed)
        truth_mode = str(evaluation_cfg.get("truth_mode", "direction_invariant"))
        if truth_mode not in {"exact", "direction_invariant"}:
            raise ValueError("evaluation.truth_mode 必须是 exact 或 direction_invariant")
        truth = exact_truth if truth_mode == "exact" else direction_truth
        candidate_metadata = _candidate_metadata(processed)
        if set(candidate_store.keys) != set(candidate_metadata):
            raise ValueError("完整候选 embedding 与蛋白元数据的 key 集合不一致")
    else:
        candidate_store = test_candidate_store
        truth = load_truth(processed / "test_truth.tsv")
        exact_truth = truth
        direction_truth = truth
        query_metadata = load_query_metadata(processed / "test_queries.tsv")
        query_keys = list(truth)
        candidate_metadata = {
            row["protein_key"]: row for row in read_tsv(processed / "test_candidates.tsv")
        }
        candidate_audit = {"scope": "test_only", "count": len(candidate_store.keys)}
        truth_audit = {"scope": "test_truth_only"}

    expected_candidate_count = evaluation_cfg.get("expected_candidate_count")
    if expected_candidate_count is not None and len(candidate_store.keys) != int(
        expected_candidate_count
    ):
        raise ValueError(
            "候选库数量不符合配置："
            f"expected={expected_candidate_count}, actual={len(candidate_store.keys)}"
        )

    query_rc_map: dict[str, str] | None = None
    query_eval_store = test_dna_store
    if bool(evaluation_cfg.get("reverse_complement_ensemble", False)):
        query_rc_map = _load_rc_map(processed / "test_queries_rc_map.tsv")
        missing_query_rc = sorted(set(query_rc_map.values()) - set(test_dna_store.keys))
        if missing_query_rc:
            query_rc_only = load_embedding_store(runtime_cfg["test_query_rc_embedding_cache"])
            if not query_rc_only.contains_all(missing_query_rc):
                raise ValueError("测试 query 反向互补 embedding cache 不完整")
            query_eval_store = merge_embedding_stores(
                [test_dna_store, query_rc_only], ["test_dna", "test_query_rc"]
            )
    dna_ranking_path = processed / "train_dna_ranking_groups.jsonl"
    dna_ranking_weight = float(training_cfg.get("dna_ranking_loss_weight", 0.0))
    dna_ranking_groups: list[DnaRankingGroup] = []
    dna_ranking_sampler: DnaRankingBatchSampler | None = None
    if dna_ranking_weight > 0:
        if not dna_ranking_path.exists():
            raise ValueError(
                "启用 DNA-centric ranking 前必须重新运行 prepare，"
                "生成 train_dna_ranking_groups.jsonl"
            )
        dna_ranking_groups = load_dna_ranking_groups(dna_ranking_path)
        expected_queries = training_cfg.get("dna_ranking_expected_queries")
        expected_triplets = training_cfg.get("dna_ranking_expected_triplets")
        actual_triplets = sum(group.triplet_count for group in dna_ranking_groups)
        if expected_queries is not None and len(dna_ranking_groups) != int(expected_queries):
            raise ValueError(
                "DNA ranking query 数量不符合配置："
                f"expected={expected_queries}, actual={len(dna_ranking_groups)}"
            )
        if expected_triplets is not None and actual_triplets != int(expected_triplets):
            raise ValueError(
                "DNA ranking triplet 数量不符合配置："
                f"expected={expected_triplets}, actual={actual_triplets}"
            )
        dna_ranking_sampler = DnaRankingBatchSampler(
            dna_ranking_groups,
            int(training_cfg.get("dna_ranking_queries_per_batch", 8)),
            seed,
        )

    clip_style_loss_weight = float(training_cfg.get("clip_style_loss_weight", 0.0))
    if clip_style_loss_weight < 0:
        raise ValueError("training.clip_style_loss_weight 不能小于 0")
    bce_loss_weight = float(training_cfg.get("bce_loss_weight", 1.0))
    triple_aware = bool(training_cfg.get("triple_aware", False))
    triple_tuple_loss_weight = float(
        training_cfg.get("triple_tuple_loss_weight", 0.0)
    )
    triple_structured_negative_loss_weight = float(
        training_cfg.get("triple_structured_negative_loss_weight", 0.0)
    )
    triple_softmin_temperature = float(
        training_cfg.get("triple_softmin_temperature", 0.1)
    )
    triple_structured_negative_margin = float(
        training_cfg.get("triple_structured_negative_margin", 0.1)
    )
    for name, value in (
        ("bce_loss_weight", bce_loss_weight),
        ("triple_tuple_loss_weight", triple_tuple_loss_weight),
        (
            "triple_structured_negative_loss_weight",
            triple_structured_negative_loss_weight,
        ),
    ):
        if value < 0:
            raise ValueError(f"training.{name} 不能小于 0")
    if (triple_tuple_loss_weight > 0 or triple_structured_negative_loss_weight > 0) and not triple_aware:
        raise ValueError("启用 triple loss 时必须设置 training.triple_aware=true")
    if triple_aware and train_rc_map is None:
        raise ValueError("triple-aware 训练需要启用反向互补增强并提供完整 RC 映射")
    train_pairs_path = processed / "train_pairs.tsv"
    positive_dnas_by_protein: dict[str, set[str]] = defaultdict(set)
    negative_dnas_by_protein: dict[str, set[str]] = defaultdict(set)
    if clip_style_loss_weight > 0:
        for row in read_tsv(train_pairs_path):
            protein_key = row["protein_key"]
            dna_key = row["dna_key"]
            target = (
                positive_dnas_by_protein
                if int(row["label"]) == 1
                else negative_dnas_by_protein
            )
            target[protein_key].add(dna_key)
            if train_rc_map is not None:
                target[protein_key].add(train_rc_map[dna_key])

    triples_path = processed / "train_triples.jsonl"
    triples: list[TripleSample] = []
    positive_pairs_by_protein: dict[str, set[tuple[str, str]]] = defaultdict(set)
    triple_sampler: TripleEpochBatchSampler | None = None
    if triple_aware:
        if not triples_path.exists():
            raise ValueError(
                "启用 triple-aware 训练前必须重新运行 prepare，生成 train_triples.jsonl"
            )
        triples = load_triples(triples_path)
        if not triples:
            raise ValueError("train_triples.jsonl 中没有可训练的完整三元组")
        expected_triples = training_cfg.get("triple_expected_count")
        expected_proteins = training_cfg.get("triple_expected_unique_proteins")
        actual_proteins = len({triple.protein_key for triple in triples})
        if expected_triples is not None and len(triples) != int(expected_triples):
            raise ValueError(
                "完整三元组数量不符合配置："
                f"expected={expected_triples}, actual={len(triples)}"
            )
        if expected_proteins is not None and actual_proteins != int(expected_proteins):
            raise ValueError(
                "三元组蛋白数量不符合配置："
                f"expected={expected_proteins}, actual={actual_proteins}"
            )
        for triple in triples:
            positive_pairs_by_protein[triple.protein_key].add(
                triple_pair_key(triple, train_rc_map)
            )
        triple_sampler = TripleEpochBatchSampler(
            triples, int(training_cfg["proteins_per_batch"]), seed
        )

    fingerprints = {
        "train_groups_sha256": sha256_file(processed / "train_groups.jsonl"),
        "test_truth_sha256": sha256_file(processed / "test_truth.tsv"),
        "train_protein_cache": train_protein_store.manifest,
        "train_dna_cache": train_dna_primary_store.manifest,
        "test_candidate_cache": test_candidate_store.manifest,
        "test_dna_cache": test_dna_store.manifest,
        "training_dna_union": train_dna_store.manifest,
        "candidate_library": candidate_store.manifest,
        "evaluation_query_union": query_eval_store.manifest,
        "candidate_audit": candidate_audit,
        "truth_audit": truth_audit,
    }
    if dna_ranking_weight > 0:
        fingerprints["train_dna_ranking_groups_sha256"] = sha256_file(dna_ranking_path)
    if clip_style_loss_weight > 0:
        fingerprints["train_pairs_sha256"] = sha256_file(train_pairs_path)
    if triple_aware:
        fingerprints["train_triples_sha256"] = sha256_file(triples_path)

    groups = load_groups(processed / "train_groups.jsonl")
    sampler: EpochBatchSampler | TripleEpochBatchSampler
    if triple_sampler is not None:
        sampler = triple_sampler
    else:
        sampler = EpochBatchSampler(
            groups, int(training_cfg["proteins_per_batch"]), seed
        )
    test_pairs = _load_pairs(processed / "test_pairs.tsv")
    candidate_keys = candidate_store.keys
    if not test_dna_store.contains_all(query_keys):
        raise ValueError("test DNA embedding cache 缺少 retrieval query")

    protein_dim = int(config["backbones"]["protein"]["hidden_dim"])
    dna_dim = int(config["backbones"]["dna"]["hidden_dim"])
    if train_protein_store.embeddings.shape[1] != protein_dim:
        raise ValueError("protein embedding 维度与配置不一致")
    if train_dna_store.embeddings.shape[1] != dna_dim:
        raise ValueError("DNA embedding 维度与配置不一致")
    for train_store, test_store, label in (
        (train_protein_store, test_candidate_store, "protein"),
        (train_dna_store, test_dna_store, "DNA"),
    ):
        for field in ("model", "resolved_revision", "pooling", "normalization_version"):
            train_value = train_store.manifest.get(field)
            test_value = test_store.manifest.get(field)
            if train_value is not None and test_value is not None and train_value != test_value:
                raise ValueError(f"train/test {label} embedding cache 的 {field} 不一致")
    model_cfg = config["model"]
    model = RecRankModel(
        protein_dim=protein_dim,
        dna_dim=dna_dim,
        hidden_dim=int(model_cfg["projection_hidden_dim"]),
        projection_dim=int(model_cfg["projection_dim"]),
        dropout=float(model_cfg["dropout"]),
        gamma_init=float(model_cfg["gamma_init"]),
        gamma_epsilon=float(model_cfg["gamma_epsilon"]),
        projection_head_type=str(model_cfg.get("projection_head_type", "legacy")),
        projection_residual_blocks=int(model_cfg.get("projection_residual_blocks", 2)),
        projection_expansion_dim=(
            int(model_cfg["projection_expansion_dim"])
            if model_cfg.get("projection_expansion_dim") is not None
            else None
        ),
        projection_layer_scale_init=float(
            model_cfg.get("projection_layer_scale_init", 0.1)
        ),
    ).to(device)
    projection_parameters = [
        *model.protein_projection.parameters(),
        *model.dna_projection.parameters(),
        model.bias,
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": projection_parameters,
                "lr": float(training_cfg["projection_learning_rate"]),
                "weight_decay": float(training_cfg["projection_weight_decay"]),
                "name": "projection",
            },
            {
                "params": [model.raw_gamma],
                "lr": float(training_cfg["gamma_learning_rate"]),
                "weight_decay": float(training_cfg["gamma_weight_decay"]),
                "name": "gamma",
            },
        ]
    )
    amp_enabled = bool(training_cfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    scheduler_name = str(training_cfg.get("scheduler", "none")).lower()
    if scheduler_name == "none":
        scheduler = None
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(training_cfg["epochs"])
        )
    else:
        raise ValueError("training.scheduler 目前只支持 none 或 cosine")

    start_epoch = 1
    global_step = 0
    best: dict[str, Any] | None = None
    seen_pairs: set[tuple[str, str]] = set()
    if resume:
        checkpoint = load_checkpoint(run_path / "checkpoints" / "last.pt", device)
        expected_hash = sha256_text(canonical_json(config))
        if checkpoint["config_sha256"] != expected_hash:
            raise ValueError("resume 配置与 checkpoint 不一致")
        if checkpoint["fingerprints"] != fingerprints:
            raise ValueError("resume 数据或 embedding cache 指纹与 checkpoint 不一致")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        if scheduler is not None:
            if checkpoint.get("scheduler_state") is None:
                raise ValueError("配置要求 scheduler，但 checkpoint 没有 scheduler state")
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        _restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best = checkpoint["best"]
        for epoch_value in range(1, start_epoch):
            for pair in sampler.epoch_pairs(epoch_value):
                seen_pairs.add((pair.protein_key, pair.dna_key))
    else:
        dump_yaml(run_path / "resolved_config.yaml", config)
        atomic_json_dump(run_path / "manifest.json", _runtime_manifest(config, fingerprints, run_path.name))
        atomic_json_dump(
            run_path / "evaluation" / "evaluation_scope.json",
            {
                "candidate_library": candidate_audit,
                "truth": truth_audit,
                "candidate_count": len(candidate_keys),
                "query_count": len(query_keys),
                "novel_dna_query_count": sum(
                    int(query_metadata[key].get("is_novel_dna", "0"))
                    for key in query_keys
                ),
                "primary_metric": evaluation_cfg.get("primary_metric", "recall_at_10"),
                "truth_mode": evaluation_cfg.get("truth_mode", "exact"),
                "reverse_complement_ensemble": bool(
                    evaluation_cfg.get("reverse_complement_ensemble", False)
                ),
                "dna_ranking_supervision": {
                    "enabled": dna_ranking_weight > 0,
                    "query_count": len(dna_ranking_groups),
                    "triplet_count": sum(
                        group.triplet_count for group in dna_ranking_groups
                    ),
                    "unknown_pairs_used_as_negatives": False,
                },
                "clip_style_unknown_supervision": {
                    "enabled": clip_style_loss_weight > 0,
                    "loss": "symmetric_multi_positive_infonce",
                    "loss_weight": clip_style_loss_weight,
                    "matrix": "unique_batch_proteins_x_unique_batch_dnas",
                    "known_positive_mask": "all_direction_equivalent_train_positives",
                    "negative_denominator": "explicit_negatives_and_all_unknown_cells",
                    "reverse_complement_dnas_in_matrix": rc_augmentation_fraction > 0,
                    "retains_v1_3_supervised_bce_and_protein_ranking": (
                        bce_loss_weight > 0
                        or float(training_cfg.get("ranking_loss_weight", 0.0)) > 0
                    ),
                    "interpretation": "experimental_clip_style_unknown_negatives",
                },
                "triple_aware_supervision": {
                    "enabled": triple_aware,
                    "source_unit": "intact_raw_recombinase_attP_attB_row",
                    "triple_count": len(triples),
                    "unique_proteins": len(
                        {triple.protein_key for triple in triples}
                    ),
                    "tuple_loss_weight": triple_tuple_loss_weight,
                    "structured_negative_loss_weight": (
                        triple_structured_negative_loss_weight
                    ),
                    "structured_negative_margin": (
                        triple_structured_negative_margin
                    ),
                    "softmin_temperature": triple_softmin_temperature,
                    "site_pair_order_invariant": True,
                    "strand_score": "cosine_with_normalized_original_plus_reverse_complement",
                    "joint_score": "normalized_softmin_of_two_site_scores",
                    "inference_changed": False,
                },
            },
        )
        atomic_json_dump(run_path / "status.json", {"status": "running", "updated_at": utc_now_iso()})

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("缺少 tensorboard；请在 rec_rank 环境安装 requirements.txt") from exc
    writer = SummaryWriter(log_dir=run_path / "logs" / "tensorboard", purge_step=start_epoch if resume else None)
    recall_ks = tuple(int(value) for value in evaluation_cfg["recall_ks"])
    selection_tiebreak = list(
        evaluation_cfg.get(
            "selection_tiebreak",
            [evaluation_cfg.get("primary_metric", "recall_at_10"), "mrr", "median_rank", "earlier_epoch"],
        )
    )
    primary_metric = str(evaluation_cfg.get("primary_metric", "recall_at_10"))
    if not selection_tiebreak or selection_tiebreak[0] != primary_metric:
        raise ValueError("evaluation.selection_tiebreak 第一项必须等于 primary_metric")
    query_batch_size = int(evaluation_cfg.get("query_batch_size", 64))

    def run_retrieval() -> tuple[RetrievalEvaluation, torch.Tensor | None, list[dict[str, Any]]]:
        if not full_library_mode:
            evaluation, scores = _retrieval_evaluation(
                model,
                candidate_store,
                test_dna_store,
                query_keys,
                candidate_keys,
                truth,
                query_metadata,
                device,
                recall_ks,
            )
            return evaluation, scores, []
        full = evaluate_model_full_library(
            model=model,
            candidates=candidate_store,
            query_store=test_dna_store,
            query_keys=query_keys,
            exact_truth=exact_truth,
            direction_truth=direction_truth,
            query_metadata=query_metadata,
            candidate_metadata=candidate_metadata,
            device=device,
            recall_ks=recall_ks,
            query_batch_size=query_batch_size,
            top_k=10,
            query_rc_store=query_eval_store if query_rc_map is not None else None,
            query_rc_map=query_rc_map,
        )
        primary_metrics = (
            full.exact_metrics
            if evaluation_cfg.get("truth_mode") == "exact"
            else full.direction_metrics
        )
        primary_rows = (
            full.exact_rows
            if evaluation_cfg.get("truth_mode") == "exact"
            else full.direction_rows
        )
        metrics = dict(primary_metrics)
        metrics.update({f"exact_{key}": value for key, value in full.exact_metrics.items()})
        return RetrievalEvaluation(metrics=metrics, query_rows=primary_rows, ranks=torch.empty(0)), None, full.top_rows

    def export_best_retrieval(
        evaluation: RetrievalEvaluation,
        scores: torch.Tensor | None,
        top_rows: list[dict[str, Any]],
        checkpoint_sha: str,
    ) -> None:
        if full_library_mode:
            for row in top_rows:
                row["checkpoint_id"] = checkpoint_sha[:20]
            _export_full_top_rows(run_path / "evaluation" / "best_dense_top10.tsv", top_rows)
        else:
            if scores is None:
                raise RuntimeError("test-only retrieval 缺少 score matrix")
            export_dense_topk(
                run_path / "evaluation" / "best_dense_top10.tsv",
                scores,
                evaluation.ranks,
                query_keys,
                candidate_keys,
                10,
                checkpoint_sha[:20],
                sha256_text(canonical_json(candidate_store.manifest))[:20],
                sequences={key: row.get("sequence", "") for key, row in query_metadata.items()},
                candidate_metadata=candidate_metadata,
            )

    total_unique_pairs = len(
        {
            (group["protein_key"], dna_key)
            for group in groups
            for category in ("attP_pos", "attB_pos", "attP_neg", "attB_neg")
            for dna_key in group.get(category, [])
        }
    )

    stop_requested = False
    previous_handlers: dict[int, Any] = {}

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        append_jsonl(run_path / "logs" / "events.jsonl", {"event": "signal", "signal": signum, "time": utc_now_iso()})

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)

    try:
        if not resume:
            evaluation0, scores0, top_rows0 = run_retrieval()
            if full_library_mode:
                random_metrics = {
                    key: value
                    for key, value in evaluation0.metrics.items()
                    if "random_recall" in key
                }
                random_metrics.update(
                    {"queries": len(query_keys), "candidates": len(candidate_keys)}
                )
            else:
                random_metrics = _random_baseline(
                    query_keys, candidate_keys, truth, query_metadata, seed, recall_ks
                )
            atomic_json_dump(run_path / "evaluation" / "random_baseline.json", random_metrics)
            test_pair_metrics0, test_pos0, test_neg0 = _evaluate_labeled_pairs(
                model, test_pairs, test_candidate_store, test_dna_store, device
            )
            row0: dict[str, Any] = {
                "epoch": 0,
                "global_step": 0,
                **test_pair_metrics0,
                **evaluation0.metrics,
                "gamma": float(model.gamma.detach().cpu()),
                "bias": float(model.bias.detach().cpu()),
                "projection_lr": optimizer.param_groups[0]["lr"],
                "gamma_lr": optimizer.param_groups[1]["lr"],
                "cumulative_unique_pairs": 0,
                "cumulative_pair_coverage": 0.0,
                "is_best": 1,
            }
            best = {"epoch": 0, "metrics": evaluation0.metrics}
            checkpoint0 = _build_checkpoint(
                model, optimizer, scaler, scheduler, 0, 0, best, config, fingerprints
            )
            atomic_torch_save(run_path / "checkpoints" / "best.pt", checkpoint0)
            atomic_torch_save(run_path / "checkpoints" / "last.pt", checkpoint0)
            checkpoint0_sha = sha256_file(run_path / "checkpoints" / "best.pt")
            export_best_retrieval(evaluation0, scores0, top_rows0, checkpoint0_sha)
            atomic_json_dump(run_path / "evaluation" / "best_metrics.json", best)
            export_query_rows(
                run_path / "evaluation" / "best_query_metrics.tsv", evaluation0.query_rows
            )
            append_csv(run_path / "logs" / "history.csv", row0, HISTORY_FIELDS)
            append_jsonl(run_path / "logs" / "events.jsonl", {"event": "epoch", **row0})
            _log_tensorboard(writer, row0, 0)
            _write_epoch_artifacts(
                run_path,
                0,
                evaluation0,
                test_pos0,
                test_neg0,
                int(evaluation_cfg["recall_curve_max_k"]),
            )

        for epoch in range(start_epoch, int(training_cfg["epochs"]) + 1):
            if stop_requested:
                break
            epoch_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model.train()
            total_bce_loss_sum = 0.0
            total_clip_loss_sum = 0.0
            total_clip_protein_to_dna_loss_sum = 0.0
            total_clip_dna_to_protein_loss_sum = 0.0
            clip_batch_count = 0
            clip_matrix_cells = 0
            clip_positive_cells = 0
            clip_explicit_negative_cells = 0
            clip_unknown_negative_cells = 0
            clip_active_proteins = 0
            clip_active_dnas = 0
            clip_positive_cosine_sum = 0.0
            clip_explicit_negative_cosine_sum = 0.0
            clip_unknown_negative_cosine_sum = 0.0
            total_tuple_loss_sum = 0.0
            total_tuple_protein_to_pair_loss_sum = 0.0
            total_tuple_pair_to_protein_loss_sum = 0.0
            tuple_batch_count = 0
            tuple_matrix_cells = 0
            tuple_positive_cells = 0
            tuple_unknown_cells = 0
            tuple_positive_score_sum = 0.0
            tuple_unknown_score_sum = 0.0
            total_structured_negative_loss_sum = 0.0
            structured_batch_count = 0
            structured_comparisons = 0
            structured_active_count = 0.0
            structured_positive_score_sum = 0.0
            structured_corrupted_score_sum = 0.0
            structured_margin_sum = 0.0
            total_ranking_loss_sum = 0.0
            ranking_triplet_count = 0
            ranking_active_count = 0.0
            protein_ranking_normalizer = 0
            total_dna_ranking_loss_sum = 0.0
            dna_ranking_group_count = 0
            dna_ranking_triplet_count = 0
            dna_ranking_comparison_count = 0
            dna_ranking_active_count = 0.0
            dna_ranking_margin_sum = 0.0
            total_rc_consistency_loss_sum = 0.0
            rc_consistency_dna_count = 0
            rc_embedding_cosine_sum = 0.0
            rc_pair_count = 0
            original_pair_count = 0
            pair_count = 0
            positive_loss_values: list[float] = []
            negative_loss_values: list[float] = []
            positive_cosines: list[float] = []
            negative_cosines: list[float] = []
            margins: list[float] = []
            gradient_norms: list[float] = []
            clipped_steps = 0
            clip_norm = float(training_cfg["gradient_clip_norm"])
            ranking_weight = float(training_cfg.get("ranking_loss_weight", 0.0))
            ranking_margin = float(training_cfg.get("ranking_margin", 0.1))
            ranking_loss_type = str(
                training_cfg.get("protein_ranking_loss_type", "hinge")
            ).lower()
            protein_group_balanced = bool(
                training_cfg.get("protein_ranking_group_balanced", False)
            )
            if ranking_loss_type not in {"hinge", "smooth"}:
                raise ValueError("protein_ranking_loss_type 必须是 hinge 或 smooth")
            dna_ranking_margin = float(
                training_cfg.get("dna_ranking_margin", ranking_margin)
            )
            dna_ranking_loss_type = str(
                training_cfg.get("dna_ranking_loss_type", "smooth")
            ).lower()
            if dna_ranking_loss_type not in {"hinge", "smooth"}:
                raise ValueError("dna_ranking_loss_type 必须是 hinge 或 smooth")
            dna_ranking_use_rc = bool(
                training_cfg.get("dna_ranking_use_reverse_complement", True)
            )
            rc_consistency_weight = float(
                training_cfg.get("rc_consistency_loss_weight", 0.0)
            )
            if (dna_ranking_use_rc and dna_ranking_weight > 0) or rc_consistency_weight > 0:
                if train_rc_map is None:
                    raise ValueError("V1.2 的 RC ranking/consistency 需要完整训练 RC 映射")
            dna_ranking_batches = (
                iter(dna_ranking_sampler.iter_batches(epoch))
                if dna_ranking_sampler is not None
                else iter(())
            )

            if triple_sampler is not None:
                training_batches = (
                    (triple_batch, flatten_triple_batch(triple_batch))
                    for triple_batch in triple_sampler.iter_triple_batches(epoch)
                )
            else:
                training_batches = (
                    ([], pair_batch) for pair_batch in sampler.iter_batches(epoch)
                )

            for triple_batch, original_batch in training_batches:
                if stop_requested:
                    break
                if train_rc_map is not None:
                    batch, dna_keys, batch_rc_count = _augment_training_batch(
                        original_batch,
                        train_rc_map,
                        rc_augmentation_fraction,
                        seed,
                        epoch,
                    )
                else:
                    batch = original_batch
                    dna_keys = [pair.dna_key for pair in original_batch]
                    batch_rc_count = 0
                proteins = train_protein_store.rows(
                    [pair.protein_key for pair in batch], device=device
                ).float()
                dnas = train_dna_store.rows(dna_keys, device=device).float()
                labels = torch.tensor(
                    [pair.label for pair in batch], device=device, dtype=torch.float32
                )
                dna_ranking_batch = next(dna_ranking_batches, [])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
                ):
                    bce_loss, logits, cosine = model.loss(proteins, dnas, labels)
                    if protein_group_balanced or ranking_loss_type == "smooth":
                        (
                            ranking_loss,
                            batch_triplets,
                            batch_ranking_groups,
                            batch_active_fraction,
                        ) = grouped_protein_ranking_loss(
                            cosine,
                            batch,
                            ranking_margin,
                            smooth=ranking_loss_type == "smooth",
                        )
                        batch_ranking_normalizer = batch_ranking_groups
                    else:
                        (
                            ranking_loss,
                            batch_triplets,
                            batch_active_fraction,
                        ) = explicit_ranking_loss(cosine, batch, ranking_margin)
                        batch_ranking_groups = 0
                        batch_ranking_normalizer = batch_triplets
                    if dna_ranking_batch:
                        (
                            dna_ranking_loss,
                            batch_dna_triplets,
                            batch_dna_comparisons,
                            batch_dna_active_fraction,
                            batch_dna_margin,
                        ) = dna_centric_ranking_loss(
                            model,
                            dna_ranking_batch,
                            train_protein_store,
                            train_dna_store,
                            train_rc_map,
                            device,
                            dna_ranking_margin,
                            smooth=dna_ranking_loss_type == "smooth",
                            include_reverse_complement=dna_ranking_use_rc,
                        )
                    else:
                        dna_ranking_loss = bce_loss * 0.0
                        batch_dna_triplets = 0
                        batch_dna_comparisons = 0
                        batch_dna_active_fraction = 0.0
                        batch_dna_margin = 0.0
                    if clip_style_loss_weight > 0:
                        clip_loss, batch_clip_stats = multi_positive_clip_loss(
                            model,
                            [pair.protein_key for pair in original_batch],
                            dna_keys,
                            train_protein_store,
                            train_dna_store,
                            positive_dnas_by_protein,
                            negative_dnas_by_protein,
                            device,
                        )
                    else:
                        clip_loss = bce_loss * 0.0
                        batch_clip_stats = {
                            "protein_to_dna_loss": 0.0,
                            "dna_to_protein_loss": 0.0,
                            "matrix_cells": 0,
                            "positive_cells": 0,
                            "explicit_negative_cells": 0,
                            "unknown_negative_cells": 0,
                            "active_proteins": 0,
                            "active_dnas": 0,
                            "positive_cosine_sum": 0.0,
                            "explicit_negative_cosine_sum": 0.0,
                            "unknown_negative_cosine_sum": 0.0,
                        }
                    if triple_tuple_loss_weight > 0:
                        tuple_loss, batch_tuple_stats = triple_tuple_contrastive_loss(
                            model,
                            triple_batch,
                            train_protein_store,
                            train_dna_store,
                            train_rc_map,
                            positive_pairs_by_protein,
                            device,
                            triple_softmin_temperature,
                        )
                    else:
                        tuple_loss = bce_loss * 0.0
                        batch_tuple_stats = {
                            "protein_to_pair_loss": 0.0,
                            "pair_to_protein_loss": 0.0,
                            "matrix_cells": 0,
                            "positive_cells": 0,
                            "unknown_cells": 0,
                            "positive_score_sum": 0.0,
                            "unknown_score_sum": 0.0,
                        }
                    if triple_structured_negative_loss_weight > 0:
                        (
                            structured_negative_loss,
                            batch_structured_stats,
                        ) = triple_structured_negative_loss(
                            model,
                            triple_batch,
                            train_protein_store,
                            train_dna_store,
                            train_rc_map,
                            device,
                            triple_structured_negative_margin,
                            triple_softmin_temperature,
                        )
                    else:
                        structured_negative_loss = bce_loss * 0.0
                        batch_structured_stats = {
                            "comparisons": 0,
                            "active_fraction": 0.0,
                            "positive_score_sum": 0.0,
                            "corrupted_score_sum": 0.0,
                            "margin_sum": 0.0,
                        }
                    if rc_consistency_weight > 0:
                        rc_consistency_loss, batch_rc_embedding_cosine = (
                            reverse_complement_consistency_loss(
                                model,
                                [pair.dna_key for pair in original_batch],
                                train_dna_store,
                                train_rc_map,
                                device,
                            )
                        )
                        batch_rc_unique_count = len(
                            {pair.dna_key for pair in original_batch}
                        )
                    else:
                        rc_consistency_loss = bce_loss * 0.0
                        batch_rc_embedding_cosine = 0.0
                        batch_rc_unique_count = 0
                    loss = (
                        bce_loss_weight * bce_loss
                        + ranking_weight * ranking_loss
                        + dna_ranking_weight * dna_ranking_loss
                        + rc_consistency_weight * rc_consistency_loss
                        + clip_style_loss_weight * clip_loss
                        + triple_tuple_loss_weight * tuple_loss
                        + triple_structured_negative_loss_weight
                        * structured_negative_loss
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"epoch {epoch} step {global_step}: loss 为 NaN/Inf")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                gradient_norm_value = float(gradient_norm.detach().cpu())
                if not np.isfinite(gradient_norm_value):
                    raise FloatingPointError(
                        f"epoch {epoch} step {global_step}: gradient norm 为 NaN/Inf"
                    )
                clipped_steps += int(gradient_norm_value > clip_norm)
                gradient_norms.append(gradient_norm_value)
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
                    positive_mask = labels == 1
                    negative_mask = ~positive_mask
                    positive_loss_values.extend(losses[positive_mask].float().cpu().tolist())
                    negative_loss_values.extend(losses[negative_mask].float().cpu().tolist())
                    positive_cosines.extend(cosine[positive_mask].float().cpu().tolist())
                    negative_cosines.extend(cosine[negative_mask].float().cpu().tolist())
                    group_values: dict[int, dict[int, list[float]]] = defaultdict(
                        lambda: {0: [], 1: []}
                    )
                    for pair, value in zip(batch, cosine.float().cpu().tolist(), strict=True):
                        group_values[pair.group_index][pair.label].append(value)
                        seen_pairs.add((pair.protein_key, pair.dna_key))
                    for values in group_values.values():
                        if values[0] and values[1]:
                            margins.append(float(np.mean(values[1]) - np.mean(values[0])))
                total_bce_loss_sum += float(bce_loss.detach().cpu()) * len(batch)
                if clip_style_loss_weight > 0:
                    total_clip_loss_sum += float(clip_loss.detach().float().cpu())
                    total_clip_protein_to_dna_loss_sum += float(
                        batch_clip_stats["protein_to_dna_loss"]
                    )
                    total_clip_dna_to_protein_loss_sum += float(
                        batch_clip_stats["dna_to_protein_loss"]
                    )
                    clip_batch_count += 1
                    clip_matrix_cells += int(batch_clip_stats["matrix_cells"])
                    clip_positive_cells += int(batch_clip_stats["positive_cells"])
                    clip_explicit_negative_cells += int(
                        batch_clip_stats["explicit_negative_cells"]
                    )
                    clip_unknown_negative_cells += int(
                        batch_clip_stats["unknown_negative_cells"]
                    )
                    clip_active_proteins += int(batch_clip_stats["active_proteins"])
                    clip_active_dnas += int(batch_clip_stats["active_dnas"])
                    clip_positive_cosine_sum += float(
                        batch_clip_stats["positive_cosine_sum"]
                    )
                    clip_explicit_negative_cosine_sum += float(
                        batch_clip_stats["explicit_negative_cosine_sum"]
                    )
                    clip_unknown_negative_cosine_sum += float(
                        batch_clip_stats["unknown_negative_cosine_sum"]
                    )
                if triple_tuple_loss_weight > 0:
                    total_tuple_loss_sum += float(tuple_loss.detach().float().cpu())
                    total_tuple_protein_to_pair_loss_sum += float(
                        batch_tuple_stats["protein_to_pair_loss"]
                    )
                    total_tuple_pair_to_protein_loss_sum += float(
                        batch_tuple_stats["pair_to_protein_loss"]
                    )
                    tuple_batch_count += 1
                    tuple_matrix_cells += int(batch_tuple_stats["matrix_cells"])
                    tuple_positive_cells += int(batch_tuple_stats["positive_cells"])
                    tuple_unknown_cells += int(batch_tuple_stats["unknown_cells"])
                    tuple_positive_score_sum += float(
                        batch_tuple_stats["positive_score_sum"]
                    )
                    tuple_unknown_score_sum += float(
                        batch_tuple_stats["unknown_score_sum"]
                    )
                if triple_structured_negative_loss_weight > 0:
                    total_structured_negative_loss_sum += float(
                        structured_negative_loss.detach().float().cpu()
                    )
                    structured_batch_count += 1
                    batch_comparisons = int(batch_structured_stats["comparisons"])
                    structured_comparisons += batch_comparisons
                    structured_active_count += (
                        float(batch_structured_stats["active_fraction"])
                        * batch_comparisons
                    )
                    structured_positive_score_sum += float(
                        batch_structured_stats["positive_score_sum"]
                    )
                    structured_corrupted_score_sum += float(
                        batch_structured_stats["corrupted_score_sum"]
                    )
                    structured_margin_sum += float(
                        batch_structured_stats["margin_sum"]
                    )
                total_ranking_loss_sum += (
                    float(ranking_loss.detach().cpu()) * batch_ranking_normalizer
                )
                protein_ranking_normalizer += batch_ranking_normalizer
                ranking_triplet_count += batch_triplets
                ranking_active_count += batch_active_fraction * batch_triplets
                if dna_ranking_batch:
                    batch_group_count = len(dna_ranking_batch)
                    total_dna_ranking_loss_sum += (
                        float(dna_ranking_loss.detach().cpu()) * batch_group_count
                    )
                    dna_ranking_group_count += batch_group_count
                    dna_ranking_triplet_count += batch_dna_triplets
                    dna_ranking_comparison_count += batch_dna_comparisons
                    dna_ranking_active_count += (
                        batch_dna_active_fraction * batch_dna_comparisons
                    )
                    dna_ranking_margin_sum += batch_dna_margin * batch_dna_comparisons
                if batch_rc_unique_count:
                    total_rc_consistency_loss_sum += (
                        float(rc_consistency_loss.detach().cpu())
                        * batch_rc_unique_count
                    )
                    rc_embedding_cosine_sum += (
                        batch_rc_embedding_cosine * batch_rc_unique_count
                    )
                    rc_consistency_dna_count += batch_rc_unique_count
                rc_pair_count += batch_rc_count
                original_pair_count += len(original_batch)
                pair_count += len(batch)
                global_step += 1

            if next(dna_ranking_batches, None) is not None:
                raise RuntimeError(
                    "DNA ranking batch 数多于主训练 batch；请增大 "
                    "training.dna_ranking_queries_per_batch"
                )

            if stop_requested:
                break
            mean_bce_loss = total_bce_loss_sum / pair_count
            mean_clip_loss = (
                total_clip_loss_sum / clip_batch_count if clip_batch_count else 0.0
            )
            mean_clip_protein_to_dna_loss = (
                total_clip_protein_to_dna_loss_sum / clip_batch_count
                if clip_batch_count
                else 0.0
            )
            mean_clip_dna_to_protein_loss = (
                total_clip_dna_to_protein_loss_sum / clip_batch_count
                if clip_batch_count
                else 0.0
            )
            mean_tuple_loss = (
                total_tuple_loss_sum / tuple_batch_count if tuple_batch_count else 0.0
            )
            mean_tuple_protein_to_pair_loss = (
                total_tuple_protein_to_pair_loss_sum / tuple_batch_count
                if tuple_batch_count
                else 0.0
            )
            mean_tuple_pair_to_protein_loss = (
                total_tuple_pair_to_protein_loss_sum / tuple_batch_count
                if tuple_batch_count
                else 0.0
            )
            mean_structured_negative_loss = (
                total_structured_negative_loss_sum / structured_batch_count
                if structured_batch_count
                else 0.0
            )
            mean_ranking_loss = (
                total_ranking_loss_sum / protein_ranking_normalizer
                if protein_ranking_normalizer
                else 0.0
            )
            mean_dna_ranking_loss = (
                total_dna_ranking_loss_sum / dna_ranking_group_count
                if dna_ranking_group_count
                else 0.0
            )
            mean_rc_consistency_loss = (
                total_rc_consistency_loss_sum / rc_consistency_dna_count
                if rc_consistency_dna_count
                else 0.0
            )
            train_metrics = {
                "train_loss": (
                    bce_loss_weight * mean_bce_loss
                    + ranking_weight * mean_ranking_loss
                    + dna_ranking_weight * mean_dna_ranking_loss
                    + rc_consistency_weight * mean_rc_consistency_loss
                    + clip_style_loss_weight * mean_clip_loss
                    + triple_tuple_loss_weight * mean_tuple_loss
                    + triple_structured_negative_loss_weight
                    * mean_structured_negative_loss
                ),
                "train_bce_loss": mean_bce_loss,
                "train_clip_loss": mean_clip_loss,
                "train_clip_protein_to_dna_loss": mean_clip_protein_to_dna_loss,
                "train_clip_dna_to_protein_loss": mean_clip_dna_to_protein_loss,
                "train_clip_matrix_cells": clip_matrix_cells,
                "train_clip_positive_cells": clip_positive_cells,
                "train_clip_explicit_negative_cells": clip_explicit_negative_cells,
                "train_clip_unknown_negative_cells": clip_unknown_negative_cells,
                "train_clip_active_proteins": clip_active_proteins,
                "train_clip_active_dnas": clip_active_dnas,
                "train_clip_positive_cosine_mean": (
                    clip_positive_cosine_sum / clip_positive_cells
                    if clip_positive_cells
                    else float("nan")
                ),
                "train_clip_explicit_negative_cosine_mean": (
                    clip_explicit_negative_cosine_sum / clip_explicit_negative_cells
                    if clip_explicit_negative_cells
                    else float("nan")
                ),
                "train_clip_unknown_negative_cosine_mean": (
                    clip_unknown_negative_cosine_sum / clip_unknown_negative_cells
                    if clip_unknown_negative_cells
                    else float("nan")
                ),
                "train_tuple_loss": mean_tuple_loss,
                "train_tuple_protein_to_pair_loss": (
                    mean_tuple_protein_to_pair_loss
                ),
                "train_tuple_pair_to_protein_loss": (
                    mean_tuple_pair_to_protein_loss
                ),
                "train_tuple_matrix_cells": tuple_matrix_cells,
                "train_tuple_positive_cells": tuple_positive_cells,
                "train_tuple_unknown_cells": tuple_unknown_cells,
                "train_tuple_positive_score_mean": (
                    tuple_positive_score_sum / tuple_positive_cells
                    if tuple_positive_cells
                    else float("nan")
                ),
                "train_tuple_unknown_score_mean": (
                    tuple_unknown_score_sum / tuple_unknown_cells
                    if tuple_unknown_cells
                    else float("nan")
                ),
                "train_structured_negative_loss": (
                    mean_structured_negative_loss
                ),
                "train_structured_negative_active_fraction": (
                    structured_active_count / structured_comparisons
                    if structured_comparisons
                    else 0.0
                ),
                "train_structured_negative_comparisons": structured_comparisons,
                "train_structured_positive_score_mean": (
                    structured_positive_score_sum / structured_comparisons
                    if structured_comparisons
                    else float("nan")
                ),
                "train_structured_corrupted_score_mean": (
                    structured_corrupted_score_sum / structured_comparisons
                    if structured_comparisons
                    else float("nan")
                ),
                "train_structured_margin_mean": (
                    structured_margin_sum / structured_comparisons
                    if structured_comparisons
                    else float("nan")
                ),
                "train_ranking_loss": mean_ranking_loss,
                "train_ranking_active_fraction": (
                    ranking_active_count / ranking_triplet_count
                    if ranking_triplet_count
                    else 0.0
                ),
                "train_ranking_triplets": ranking_triplet_count,
                "train_protein_ranking_loss": mean_ranking_loss,
                "train_protein_ranking_active_fraction": (
                    ranking_active_count / ranking_triplet_count
                    if ranking_triplet_count
                    else 0.0
                ),
                "train_protein_ranking_triplets": ranking_triplet_count,
                "train_dna_ranking_loss": mean_dna_ranking_loss,
                "train_dna_ranking_active_fraction": (
                    dna_ranking_active_count / dna_ranking_comparison_count
                    if dna_ranking_comparison_count
                    else 0.0
                ),
                "train_dna_ranking_groups": dna_ranking_group_count,
                "train_dna_ranking_triplets": dna_ranking_triplet_count,
                "train_dna_ranking_comparisons": dna_ranking_comparison_count,
                "train_dna_ranking_margin_mean": (
                    dna_ranking_margin_sum / dna_ranking_comparison_count
                    if dna_ranking_comparison_count
                    else 0.0
                ),
                "train_rc_consistency_loss": mean_rc_consistency_loss,
                "train_rc_embedding_cosine_mean": (
                    rc_embedding_cosine_sum / rc_consistency_dna_count
                    if rc_consistency_dna_count
                    else 0.0
                ),
                "train_rc_augmentation_fraction": rc_pair_count / original_pair_count,
                "train_rc_added_pairs": rc_pair_count,
                "train_positive_loss": float(np.mean(positive_loss_values)),
                "train_negative_loss": float(np.mean(negative_loss_values)),
                **_quantiles(positive_cosines, "train_positive_cosine"),
                **_quantiles(negative_cosines, "train_negative_cosine"),
                "train_pair_margin_mean": float(np.mean(margins)),
            }
            evaluation, scores, top_rows = run_retrieval()
            test_pair_metrics, test_positive_cosines, test_negative_cosines = _evaluate_labeled_pairs(
                model, test_pairs, test_candidate_store, test_dna_store, device
            )
            is_best = _is_better(
                evaluation.metrics, epoch, best, selection_tiebreak=selection_tiebreak
            )
            if is_best:
                best = {"epoch": epoch, "metrics": evaluation.metrics}
            epoch_seconds = time.perf_counter() - epoch_start
            gpu_memory = (
                torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
            )
            row = {
                "epoch": epoch,
                "global_step": global_step,
                **train_metrics,
                **test_pair_metrics,
                **evaluation.metrics,
                "gamma": float(model.gamma.detach().cpu()),
                "bias": float(model.bias.detach().cpu()),
                "projection_lr": optimizer.param_groups[0]["lr"],
                "gamma_lr": optimizer.param_groups[1]["lr"],
                "gradient_norm_mean": float(np.mean(gradient_norms)),
                "gradient_norm_max": float(np.max(gradient_norms)),
                "gradient_clip_fraction": clipped_steps / len(gradient_norms),
                "cumulative_unique_pairs": len(seen_pairs),
                "cumulative_pair_coverage": len(seen_pairs) / total_unique_pairs,
                "gpu_max_memory_gb": gpu_memory,
                "epoch_seconds": epoch_seconds,
                "is_best": int(is_best),
            }
            if scheduler is not None:
                scheduler.step()
            checkpoint = _build_checkpoint(
                model, optimizer, scaler, scheduler, epoch, global_step, best, config, fingerprints
            )
            atomic_torch_save(run_path / "checkpoints" / "last.pt", checkpoint)
            if is_best:
                atomic_torch_save(run_path / "checkpoints" / "best.pt", checkpoint)
                best_checkpoint_sha = sha256_file(run_path / "checkpoints" / "best.pt")
                export_best_retrieval(evaluation, scores, top_rows, best_checkpoint_sha)
                atomic_json_dump(run_path / "evaluation" / "best_metrics.json", best)
                export_query_rows(
                    run_path / "evaluation" / "best_query_metrics.tsv", evaluation.query_rows
                )
            append_csv(run_path / "logs" / "history.csv", row, HISTORY_FIELDS)
            append_jsonl(run_path / "logs" / "events.jsonl", {"event": "epoch", **row})
            gamma_warning_threshold = float(model_cfg.get("gamma_warning_threshold", 100.0))
            if row["gamma"] >= gamma_warning_threshold:
                append_jsonl(
                    run_path / "logs" / "events.jsonl",
                    {
                        "event": "gamma_warning",
                        "epoch": epoch,
                        "gamma": row["gamma"],
                        "threshold": gamma_warning_threshold,
                        "message": "gamma 已达到监控阈值；应检查是否主要依赖放大 logit 而非改善投影。",
                    },
                )
            _log_tensorboard(writer, row, epoch)
            writer.flush()
            _write_epoch_artifacts(
                run_path,
                epoch,
                evaluation,
                test_positive_cosines,
                test_negative_cosines,
                int(evaluation_cfg["recall_curve_max_k"]),
            )
            atomic_json_dump(
                run_path / "status.json",
                {
                    "status": "running",
                    "updated_at": utc_now_iso(),
                    "epoch": epoch,
                    "best_epoch": best["epoch"],
                },
            )

        final_status = "interrupted" if stop_requested else "complete"
        atomic_json_dump(
            run_path / "status.json",
            {
                "status": final_status,
                "updated_at": utc_now_iso(),
                "best_epoch": best["epoch"] if best else None,
            },
        )
    except BaseException as exc:
        try:
            diagnostic = {
                "saved_at": utc_now_iso(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "model_architecture": model.architecture,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "rng_state": _capture_rng_state(),
            }
            atomic_torch_save(run_path / "checkpoints" / "failure_diagnostic.pt", diagnostic)
        except BaseException:
            pass
        atomic_json_dump(
            run_path / "status.json",
            {
                "status": "failed",
                "updated_at": utc_now_iso(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        append_jsonl(
            run_path / "logs" / "events.jsonl",
            {"event": "failure", "type": type(exc).__name__, "message": str(exc), "time": utc_now_iso()},
        )
        raise
    finally:
        writer.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    registry_row = {
        "run_id": run_path.name,
        "seed": seed,
        "config_sha256": sha256_text(canonical_json(config)),
        "status": "interrupted" if stop_requested else "complete",
        "best_epoch": best["epoch"] if best else "",
        "recall_at_10": best["metrics"]["recall_at_10"] if best else "",
        "mrr": best["metrics"]["mrr"] if best else "",
        "median_rank": best["metrics"]["median_rank"] if best else "",
        "gamma": float(model.gamma.detach().cpu()),
        "checkpoint": str((run_path / "checkpoints" / "best.pt").resolve()),
        "completed_at": utc_now_iso(),
    }
    append_csv(
        run_root / "experiments.csv",
        registry_row,
        [
            "run_id",
            "seed",
            "config_sha256",
            "status",
            "best_epoch",
            "recall_at_10",
            "mrr",
            "median_rank",
            "gamma",
            "checkpoint",
            "completed_at",
        ],
    )
    return run_path
