from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .data import reverse_complement
from .embeddings import EmbeddingStore, load_embedding_store
from .evaluation import dense_rank_descending, load_query_metadata, load_truth, recall_curve
from .training import _project_store, load_checkpoint, model_from_checkpoint
from .utils import (
    atomic_json_dump,
    canonical_json,
    read_tsv,
    sha256_file,
    sha256_text,
    utc_now_iso,
    write_tsv,
)


TRUTH_FIELDS = ["dna_key", "positive_protein_keys"]
QUERY_FIELDS = [
    "dna_key",
    "positive_count",
    "best_positive_key",
    "best_positive_rank",
    "best_positive_cosine",
    "top1_protein_key",
    "top1_cosine",
    "is_novel_dna",
]


def _validate_store_compatibility(stores: list[EmbeddingStore]) -> None:
    if not stores:
        raise ValueError("至少需要一个候选蛋白 embedding cache")
    reference = stores[0]
    fields = ("model", "resolved_revision", "pooling", "normalization_version")
    for store in stores[1:]:
        if store.embeddings.shape[1] != reference.embeddings.shape[1]:
            raise ValueError("候选蛋白 embedding cache 的维度不一致")
        for field in fields:
            left = reference.manifest.get(field)
            right = store.manifest.get(field)
            if left is not None and right is not None and left != right:
                raise ValueError(f"候选蛋白 embedding cache 的 {field} 不一致")


def _merge_candidate_stores(
    train_cache: str | Path, test_cache: str | Path
) -> tuple[EmbeddingStore, dict[str, Any]]:
    train = load_embedding_store(train_cache)
    test = load_embedding_store(test_cache)
    _validate_store_compatibility([train, test])

    train_keys = set(train.keys)
    duplicate_keys = sorted(train_keys.intersection(test.keys))
    for key in duplicate_keys:
        train_value = train.embeddings[train.key_to_index[key]].float()
        test_value = test.embeddings[test.key_to_index[key]].float()
        if not torch.allclose(train_value, test_value, rtol=1e-5, atol=1e-6):
            raise ValueError(f"训练/测试候选 cache 对重复蛋白 {key} 的 embedding 不一致")

    test_only_keys = [key for key in test.keys if key not in train_keys]
    test_only_embeddings = test.rows(test_only_keys)
    keys = [*train.keys, *test_only_keys]
    embeddings = torch.cat([train.embeddings, test_only_embeddings], dim=0)
    source_fingerprints = {
        "train_cache_manifest_sha256": sha256_text(canonical_json(train.manifest)),
        "test_cache_manifest_sha256": sha256_text(canonical_json(test.manifest)),
    }
    manifest = {
        "kind": "in_memory_train_test_candidate_union",
        "count": len(keys),
        "shape": list(embeddings.shape),
        "ordered_keys_sha256": sha256_text("\n".join(keys)),
        "model": train.manifest.get("model"),
        "resolved_revision": train.manifest.get("resolved_revision"),
        "pooling": train.manifest.get("pooling"),
        "normalization_version": train.manifest.get("normalization_version"),
        "train_count": len(train.keys),
        "test_count": len(test.keys),
        "duplicate_count": len(duplicate_keys),
        "train_only_count": len(train.keys) - len(duplicate_keys),
        "test_only_count": len(test_only_keys),
        **source_fingerprints,
    }
    return EmbeddingStore(keys=keys, embeddings=embeddings, manifest=manifest), manifest


def _split_values(value: str) -> set[str]:
    return {item for item in value.split(";") if item}


def _candidate_metadata(processed_dir: Path) -> dict[str, dict[str, str]]:
    values: dict[str, dict[str, Any]] = {}
    for source, name in (("train", "train_proteins.tsv"), ("test", "test_candidates.tsv")):
        for row in read_tsv(processed_dir / name):
            key = row["protein_key"]
            current = values.setdefault(
                key,
                {
                    "protein_key": key,
                    "record_ids": set(),
                    "cluster_ids": set(),
                    "sources": set(),
                },
            )
            current["record_ids"].update(_split_values(row.get("record_ids", "")))
            current["cluster_ids"].update(_split_values(row.get("cluster_ids", "")))
            current["sources"].add(source)
    return {
        key: {
            "protein_key": key,
            "record_ids": ";".join(sorted(value["record_ids"])),
            "cluster_ids": ";".join(sorted(value["cluster_ids"])),
            "candidate_source": "+".join(sorted(value["sources"])),
        }
        for key, value in values.items()
    }


def _build_truths(
    processed_dir: Path,
) -> tuple[
    list[str],
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, dict[str, str]],
    dict[str, Any],
]:
    query_metadata = load_query_metadata(processed_dir / "test_queries.tsv")
    original_truth = load_truth(processed_dir / "test_truth.tsv")
    query_keys = list(original_truth)

    dna_sequences: dict[str, str] = {}
    for name in ("train_dnas.tsv", "test_dnas.tsv"):
        for row in read_tsv(processed_dir / name):
            key, sequence = row["dna_key"], row["sequence"]
            previous = dna_sequences.setdefault(key, sequence)
            if previous != sequence:
                raise ValueError(f"DNA key {key} 在训练/测试数据中对应不同序列")

    exact_positives: dict[str, set[str]] = defaultdict(set)
    direction_positives: dict[str, set[str]] = defaultdict(set)
    for name in ("train_pairs.tsv", "test_pairs.tsv"):
        for row in read_tsv(processed_dir / name):
            if int(row["label"]) != 1:
                continue
            sequence = dna_sequences[row["dna_key"]]
            canonical = min(sequence, reverse_complement(sequence))
            exact_positives[sequence].add(row["protein_key"])
            direction_positives[canonical].add(row["protein_key"])

    exact_truth: dict[str, set[str]] = {}
    direction_truth: dict[str, set[str]] = {}
    for key in query_keys:
        sequence = query_metadata[key]["sequence"]
        canonical = min(sequence, reverse_complement(sequence))
        exact_truth[key] = set(exact_positives[sequence])
        direction_truth[key] = set(direction_positives[canonical])
        if not original_truth[key].issubset(exact_truth[key]):
            raise ValueError(f"合并后的 exact truth 丢失了测试正标签: {key}")
        if not exact_truth[key].issubset(direction_truth[key]):
            raise ValueError(f"direction-invariant truth 未包含 exact truth: {key}")

    def truth_stats(truth: dict[str, set[str]]) -> dict[str, int]:
        return {
            "queries": len(truth),
            "truth_pairs": sum(len(values) for values in truth.values()),
            "changed_queries_vs_test_truth": sum(
                truth[key] != original_truth[key] for key in query_keys
            ),
            "added_truth_pairs_vs_test_truth": sum(
                len(truth[key] - original_truth[key]) for key in query_keys
            ),
            "removed_truth_pairs_vs_test_truth": sum(
                len(original_truth[key] - truth[key]) for key in query_keys
            ),
            "max_positives_per_query": max(len(values) for values in truth.values()),
        }

    audit = {
        "original_test_truth": truth_stats(original_truth),
        "exact_train_test_union": truth_stats(exact_truth),
        "direction_invariant_train_test_union": truth_stats(direction_truth),
    }
    return query_keys, exact_truth, direction_truth, query_metadata, audit


def _random_recall(candidate_count: int, positive_count: int, k: int) -> float:
    k = min(k, candidate_count)
    if positive_count <= 0:
        return 0.0
    if positive_count >= candidate_count - k + 1:
        return 1.0
    miss = 1.0
    for offset in range(k):
        miss *= (candidate_count - positive_count - offset) / (candidate_count - offset)
    return 1.0 - miss


def _metrics_from_rows(
    rows: list[dict[str, Any]], candidate_count: int, recall_ks: Iterable[int]
) -> dict[str, float | int]:
    ranks = np.asarray([row["best_positive_rank"] for row in rows], dtype=np.float64)
    result: dict[str, float | int] = {
        "queries": len(rows),
        "candidates": candidate_count,
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }
    for k in recall_ks:
        observed = float(np.mean(ranks <= k))
        random_value = float(
            np.mean(
                [
                    _random_recall(candidate_count, int(row["positive_count"]), k)
                    for row in rows
                ]
            )
        )
        result[f"recall_at_{k}"] = observed
        result[f"random_recall_at_{k}"] = random_value
        result[f"recall_at_{k}_enrichment"] = observed / random_value if random_value else math.nan

    novel_rows = [row for row in rows if int(row["is_novel_dna"]) == 1]
    result["novel_dna_queries"] = len(novel_rows)
    if not novel_rows:
        result["novel_dna_mrr"] = math.nan
        result["novel_dna_median_rank"] = math.nan
        result["novel_dna_mean_rank"] = math.nan
        for k in recall_ks:
            result[f"novel_dna_recall_at_{k}"] = math.nan
            result[f"novel_dna_random_recall_at_{k}"] = math.nan
            result[f"novel_dna_recall_at_{k}_enrichment"] = math.nan
        return result

    novel_ranks = np.asarray(
        [row["best_positive_rank"] for row in novel_rows], dtype=np.float64
    )
    result["novel_dna_mrr"] = float(np.mean(1.0 / novel_ranks))
    result["novel_dna_median_rank"] = float(np.median(novel_ranks))
    result["novel_dna_mean_rank"] = float(np.mean(novel_ranks))
    for k in recall_ks:
        observed = float(np.mean(novel_ranks <= k))
        random_value = float(
            np.mean(
                [
                    _random_recall(candidate_count, int(row["positive_count"]), k)
                    for row in novel_rows
                ]
            )
        )
        result[f"novel_dna_recall_at_{k}"] = observed
        result[f"novel_dna_random_recall_at_{k}"] = random_value
        result[f"novel_dna_recall_at_{k}_enrichment"] = (
            observed / random_value if random_value else math.nan
        )
    return result


def _query_row(
    query_index: int,
    query_key: str,
    scores: torch.Tensor,
    ranks: torch.Tensor,
    truth: dict[str, set[str]],
    candidate_keys: list[str],
    candidate_index: dict[str, int],
    query_metadata: dict[str, dict[str, str]],
) -> dict[str, Any]:
    positive_keys = truth[query_key]
    positive_indices = [candidate_index[key] for key in sorted(positive_keys)]
    positive_ranks = ranks[query_index, positive_indices]
    best_local = int(torch.argmin(positive_ranks).item())
    best_index = positive_indices[best_local]
    top1_index = int(torch.argmax(scores[query_index]).item())
    return {
        "dna_key": query_key,
        "positive_count": len(positive_keys),
        "best_positive_key": candidate_keys[best_index],
        "best_positive_rank": int(ranks[query_index, best_index].item()),
        "best_positive_cosine": float(scores[query_index, best_index].item()),
        "top1_protein_key": candidate_keys[top1_index],
        "top1_cosine": float(scores[query_index, top1_index].item()),
        "is_novel_dna": int(query_metadata[query_key].get("is_novel_dna", "0")),
    }


def _evaluate_one(
    checkpoint_path: Path,
    candidates: EmbeddingStore,
    query_store: EmbeddingStore,
    query_keys: list[str],
    exact_truth: dict[str, set[str]],
    direction_truth: dict[str, set[str]],
    query_metadata: dict[str, dict[str, str]],
    candidate_metadata: dict[str, dict[str, str]],
    output_dir: Path,
    device: torch.device,
    query_batch_size: int,
    top_k: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    model = model_from_checkpoint(checkpoint, device)

    expected_dim = int(checkpoint["model_architecture"]["protein_dim"])
    if candidates.embeddings.shape[1] != expected_dim:
        raise ValueError("完整候选库 embedding 维度与 checkpoint 不一致")
    if not query_store.contains_all(query_keys):
        raise ValueError("测试 DNA embedding cache 缺少 truth query")
    fingerprints = checkpoint.get("fingerprints", {})
    for field in ("model", "resolved_revision", "pooling", "normalization_version"):
        expected_protein = fingerprints.get("train_protein_cache", {}).get(field)
        actual_protein = candidates.manifest.get(field)
        if (
            expected_protein is not None
            and actual_protein is not None
            and expected_protein != actual_protein
        ):
            raise ValueError(f"完整候选库的 {field} 与 checkpoint 不一致")
        expected_dna = fingerprints.get("test_dna_cache", {}).get(field)
        actual_dna = query_store.manifest.get(field)
        if expected_dna is not None and actual_dna is not None and expected_dna != actual_dna:
            raise ValueError(f"测试 DNA cache 的 {field} 与 checkpoint 不一致")

    candidate_keys = candidates.keys
    candidate_index = {key: index for index, key in enumerate(candidate_keys)}
    for name, truth in (("exact", exact_truth), ("direction-invariant", direction_truth)):
        missing = set().union(*(truth[key] for key in query_keys)) - candidate_index.keys()
        if missing:
            raise ValueError(f"{name} truth 中有蛋白不在完整候选库: {sorted(missing)[:3]}")

    candidate_vectors = _project_store(model, candidates, candidate_keys, "protein", device)
    query_vectors = _project_store(model, query_store, query_keys, "dna", device)
    candidate_vectors_device = candidate_vectors.to(device=device, dtype=torch.float32)

    exact_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    checkpoint_sha = sha256_file(checkpoint_path)
    library_id = sha256_text(canonical_json(candidates.manifest))[:20]

    with torch.inference_mode():
        for start in range(0, len(query_keys), query_batch_size):
            batch_keys = query_keys[start : start + query_batch_size]
            batch_queries = query_vectors[start : start + query_batch_size].to(device)
            batch_scores_device = batch_queries @ candidate_vectors_device.T
            batch_ranks_device = dense_rank_descending(batch_scores_device)
            batch_scores = batch_scores_device.float().cpu()
            batch_ranks = batch_ranks_device.cpu()

            for local_index, query_key in enumerate(batch_keys):
                exact_rows.append(
                    _query_row(
                        local_index,
                        query_key,
                        batch_scores,
                        batch_ranks,
                        exact_truth,
                        candidate_keys,
                        candidate_index,
                        query_metadata,
                    )
                )
                direction_rows.append(
                    _query_row(
                        local_index,
                        query_key,
                        batch_scores,
                        batch_ranks,
                        direction_truth,
                        candidate_keys,
                        candidate_index,
                        query_metadata,
                    )
                )
                selected = torch.nonzero(
                    batch_ranks[local_index] <= top_k, as_tuple=False
                ).flatten()
                selected = selected[
                    torch.argsort(
                        batch_scores[local_index, selected], descending=True, stable=True
                    )
                ]
                for candidate_value in selected.tolist():
                    protein_key = candidate_keys[candidate_value]
                    metadata = candidate_metadata.get(protein_key, {})
                    top_rows.append(
                        {
                            "query_id": query_key,
                            "query_sequence": query_metadata[query_key].get("sequence", ""),
                            "protein_key": protein_key,
                            "protein_record_ids": metadata.get("record_ids", ""),
                            "protein_cluster_ids": metadata.get("cluster_ids", ""),
                            "candidate_source": metadata.get("candidate_source", ""),
                            "cosine": float(batch_scores[local_index, candidate_value].item()),
                            "dense_rank": int(batch_ranks[local_index, candidate_value].item()),
                            "checkpoint_id": checkpoint_sha[:20],
                            "candidate_library_id": library_id,
                        }
                    )

            del batch_scores_device, batch_ranks_device, batch_scores, batch_ranks

    output_dir.mkdir(parents=True, exist_ok=True)
    exact_metrics = _metrics_from_rows(exact_rows, len(candidate_keys), (1, 5, 10))
    direction_metrics = _metrics_from_rows(direction_rows, len(candidate_keys), (1, 5, 10))
    payload = {
        "completed_at": utc_now_iso(),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "candidate_library_id": library_id,
        "candidate_count": len(candidate_keys),
        "query_count": len(query_keys),
        "query_batch_size": query_batch_size,
        "elapsed_seconds": time.perf_counter() - started,
        "exact_truth": exact_metrics,
        "direction_invariant_truth": direction_metrics,
    }
    atomic_json_dump(output_dir / "metrics.json", payload)
    write_tsv(output_dir / "query_metrics_exact.tsv", exact_rows, QUERY_FIELDS)
    write_tsv(
        output_dir / "query_metrics_direction_invariant.tsv", direction_rows, QUERY_FIELDS
    )
    write_tsv(
        output_dir / "recall_curve_exact.tsv",
        recall_curve(exact_rows, 50),
        ["k", "recall"],
    )
    write_tsv(
        output_dir / "recall_curve_direction_invariant.tsv",
        recall_curve(direction_rows, 50),
        ["k", "recall"],
    )
    write_tsv(
        output_dir / f"dense_top{top_k}.tsv",
        top_rows,
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

    del model, candidate_vectors, query_vectors, candidate_vectors_device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def evaluate_full_library_checkpoints(
    checkpoint_paths: list[str | Path],
    train_candidate_cache: str | Path,
    test_candidate_cache: str | Path,
    query_cache: str | Path,
    processed_dir: str | Path,
    output_dir: str | Path,
    device: torch.device,
    query_batch_size: int = 64,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    if query_batch_size <= 0:
        raise ValueError("query_batch_size 必须大于 0")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if not checkpoint_paths:
        raise ValueError("至少需要一个 checkpoint")

    processed = Path(processed_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates, candidate_audit = _merge_candidate_stores(
        train_candidate_cache, test_candidate_cache
    )
    query_store = load_embedding_store(query_cache)
    query_keys, exact_truth, direction_truth, query_metadata, truth_audit = _build_truths(
        processed
    )
    metadata = _candidate_metadata(processed)
    if set(candidates.keys) != set(metadata):
        raise ValueError("完整候选 embedding 与蛋白元数据的 key 集合不一致")

    write_tsv(
        output / "candidate_library.tsv",
        (metadata[key] for key in candidates.keys),
        ["protein_key", "record_ids", "cluster_ids", "candidate_source"],
    )
    write_tsv(
        output / "truth_exact.tsv",
        (
            {"dna_key": key, "positive_protein_keys": ";".join(sorted(exact_truth[key]))}
            for key in query_keys
        ),
        TRUTH_FIELDS,
    )
    write_tsv(
        output / "truth_direction_invariant.tsv",
        (
            {
                "dna_key": key,
                "positive_protein_keys": ";".join(sorted(direction_truth[key])),
            }
            for key in query_keys
        ),
        TRUTH_FIELDS,
    )
    audit = {
        "created_at": utc_now_iso(),
        "candidate_library": candidate_audit,
        "truth": truth_audit,
        "query_count": len(query_keys),
        "novel_dna_query_count": sum(
            int(query_metadata[key].get("is_novel_dna", "0")) for key in query_keys
        ),
    }
    atomic_json_dump(output / "evaluation_scope.json", audit)

    labels_seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for checkpoint_value in checkpoint_paths:
        checkpoint_path = Path(checkpoint_value)
        run_name = checkpoint_path.parent.parent.name
        label = f"{run_name}_{checkpoint_path.stem}"
        if label in labels_seen:
            raise ValueError(f"checkpoint 输出标签重复: {label}")
        labels_seen.add(label)
        result = _evaluate_one(
            checkpoint_path,
            candidates,
            query_store,
            query_keys,
            exact_truth,
            direction_truth,
            query_metadata,
            metadata,
            output / label,
            device,
            query_batch_size,
            top_k,
        )
        result["label"] = label
        results.append(result)

    summary_rows = []
    for result in results:
        exact = result["exact_truth"]
        direction = result["direction_invariant_truth"]
        summary_rows.append(
            {
                "label": result["label"],
                "checkpoint_epoch": result["checkpoint_epoch"],
                "candidate_count": result["candidate_count"],
                "exact_recall_at_1": exact["recall_at_1"],
                "exact_recall_at_5": exact["recall_at_5"],
                "exact_recall_at_10": exact["recall_at_10"],
                "exact_novel_recall_at_10": exact["novel_dna_recall_at_10"],
                "exact_mrr": exact["mrr"],
                "exact_median_rank": exact["median_rank"],
                "direction_recall_at_1": direction["recall_at_1"],
                "direction_recall_at_5": direction["recall_at_5"],
                "direction_recall_at_10": direction["recall_at_10"],
                "direction_novel_recall_at_10": direction["novel_dna_recall_at_10"],
                "direction_mrr": direction["mrr"],
                "direction_median_rank": direction["median_rank"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
    summary_fields = list(summary_rows[0])
    write_tsv(output / "comparison.tsv", summary_rows, summary_fields)
    atomic_json_dump(output / "comparison.json", {"scope": audit, "results": results})
    return results
