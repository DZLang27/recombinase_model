from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .data import normalize_dna, stable_key
from .embeddings import BackboneEncoder, EmbeddingStore, load_embedding_store, save_embedding_store
from .evaluation import (
    dense_rank_descending,
    evaluate_score_matrix,
    export_dense_topk,
    export_query_rows,
    load_query_metadata,
    load_truth,
    recall_curve,
)
from .training import _project_store, load_checkpoint, model_from_checkpoint
from .utils import atomic_json_dump, canonical_json, read_tsv, sha256_file, sha256_text, write_tsv


def build_candidate_index(
    checkpoint_path: str | Path,
    candidate_embedding_cache: str | Path,
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    model = model_from_checkpoint(checkpoint, device)
    raw_store = load_embedding_store(candidate_embedding_cache)
    expected_backbone = checkpoint.get("fingerprints", {}).get("train_protein_cache", {})
    for field in ("model", "resolved_revision", "pooling", "normalization_version"):
        expected_value = expected_backbone.get(field)
        actual_value = raw_store.manifest.get(field)
        if expected_value is not None and actual_value is not None and expected_value != actual_value:
            raise ValueError(f"候选 protein cache 的 {field} 与训练 checkpoint 不一致")
    if raw_store.embeddings.shape[1] != checkpoint["model_architecture"]["protein_dim"]:
        raise ValueError("候选 protein embedding 维度与 checkpoint 不一致")
    projected = _project_store(model, raw_store, raw_store.keys, "protein", device)
    checkpoint_sha = sha256_file(checkpoint_path)
    source_manifest_sha = sha256_text(canonical_json(raw_store.manifest))
    manifest = save_embedding_store(
        output_dir,
        raw_store.keys,
        projected,
        {
            "kind": "projected_candidate_index",
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_epoch": checkpoint["epoch"],
            "source_embedding_manifest_sha256": source_manifest_sha,
            "source_embedding_cache": str(Path(candidate_embedding_cache).resolve()),
            "projection_dim": projected.shape[1],
        },
    )
    manifest["candidate_index_id"] = sha256_text(canonical_json(manifest))[:20]
    atomic_json_dump(Path(output_dir) / "manifest.json", manifest)
    metadata_path = raw_store.manifest.get("input_path")
    if metadata_path and Path(metadata_path).exists():
        candidate_key_set = set(raw_store.keys)
        metadata_rows = [
            row for row in read_tsv(metadata_path) if row.get("protein_key") in candidate_key_set
        ]
        write_tsv(
            Path(output_dir) / "candidate_metadata.tsv",
            metadata_rows,
            ["protein_key", "sequence", "record_ids", "cluster_ids"],
        )
    return manifest


def _load_query_tsv(path: str | Path) -> tuple[list[str], list[str]]:
    keys: list[str] = []
    sequences: list[str] = []
    for index, row in enumerate(read_tsv(path), start=1):
        sequence = normalize_dna(row.get("sequence"))
        if sequence is None:
            raise ValueError(f"query TSV 第 {index} 行不是合法 DNA")
        key = row.get("dna_key") or row.get("query_id") or stable_key(sequence, "dna")
        keys.append(key)
        sequences.append(sequence)
    if len(keys) != len(set(keys)):
        raise ValueError("query TSV 中 query key 重复")
    return keys, sequences


def screen(
    checkpoint_path: str | Path,
    candidate_index_dir: str | Path,
    output_path: str | Path,
    device: torch.device,
    dna: str | None = None,
    query_tsv: str | Path | None = None,
    query_embedding_cache: str | Path | None = None,
    top_k: int = 10,
) -> int:
    selected_inputs = sum(value is not None for value in (dna, query_tsv, query_embedding_cache))
    if selected_inputs != 1:
        raise ValueError("--dna、--query-tsv、--query-embedding-cache 必须且只能提供一个")
    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    checkpoint_sha = sha256_file(checkpoint_path)
    index_store = load_embedding_store(candidate_index_dir)
    expected_checkpoint = index_store.manifest.get("checkpoint_sha256")
    if expected_checkpoint != checkpoint_sha:
        raise ValueError("candidate index 与 checkpoint 不匹配，拒绝筛选")
    model = model_from_checkpoint(checkpoint, device)
    model.eval()

    sequences: list[str] = []
    if dna is not None:
        normalized = normalize_dna(dna)
        if normalized is None:
            raise ValueError("--dna 不是合法 DNA 序列")
        query_keys = [stable_key(normalized, "dna")]
        sequences = [normalized]
    elif query_tsv is not None:
        query_keys, sequences = _load_query_tsv(query_tsv)
    else:
        query_store = load_embedding_store(query_embedding_cache)  # type: ignore[arg-type]
        expected_dna_backbone = checkpoint.get("fingerprints", {}).get("train_dna_cache", {})
        for field in ("model", "resolved_revision", "pooling", "normalization_version"):
            expected_value = expected_dna_backbone.get(field)
            actual_value = query_store.manifest.get(field)
            if expected_value is not None and actual_value is not None and expected_value != actual_value:
                raise ValueError(f"query DNA cache 的 {field} 与训练 checkpoint 不一致")
        if query_store.embeddings.shape[1] != checkpoint["model_architecture"]["dna_dim"]:
            raise ValueError("query DNA embedding 维度与 checkpoint 不一致")
        query_keys = query_store.keys
        raw_query_embeddings = query_store.embeddings

    if sequences:
        backbone_cfg = checkpoint["resolved_config"]["backbones"]["dna"]
        training_dna_manifest = checkpoint.get("fingerprints", {}).get("test_dna_cache", {})
        pinned_revision = backbone_cfg.get("revision") or training_dna_manifest.get(
            "resolved_revision"
        )
        encoder = BackboneEncoder(
            model_name=backbone_cfg["model"],
            device=device,
            revision=pinned_revision,
            trust_remote_code=bool(backbone_cfg.get("trust_remote_code", False)),
        )
        batch_size = int(backbone_cfg.get("batch_size", 64))
        encoded_batches = [
            encoder.encode(sequences[start : start + batch_size])
            for start in range(0, len(sequences), batch_size)
        ]
        raw_query_embeddings = torch.cat(encoded_batches, dim=0)

    with torch.inference_mode():
        query_vectors = []
        for start in range(0, len(query_keys), 4096):
            raw = raw_query_embeddings[start : start + 4096].to(device).float()
            query_vectors.append(model.project_dnas(raw).float().cpu())
        projected_queries = torch.cat(query_vectors, dim=0)
    scores = (
        projected_queries.to(device) @ index_store.embeddings.to(device=device, dtype=torch.float32).T
    ).float().cpu()
    ranks = dense_rank_descending(scores)
    sequence_map = dict(zip(query_keys, sequences, strict=True)) if sequences else None
    metadata_file = Path(candidate_index_dir) / "candidate_metadata.tsv"
    candidate_metadata = (
        {row["protein_key"]: row for row in read_tsv(metadata_file)} if metadata_file.exists() else None
    )
    export_dense_topk(
        output_path,
        scores,
        ranks,
        query_keys,
        index_store.keys,
        top_k,
        checkpoint_id=checkpoint_sha[:20],
        candidate_index_id=index_store.manifest["candidate_index_id"],
        sequences=sequence_map,
        candidate_metadata=candidate_metadata,
    )
    return int((ranks <= top_k).sum().item())


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    candidate_embedding_cache: str | Path,
    query_embedding_cache: str | Path,
    truth_path: str | Path,
    query_metadata_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
    top_k: int = 10,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    model = model_from_checkpoint(checkpoint, device)
    candidates = load_embedding_store(candidate_embedding_cache)
    queries = load_embedding_store(query_embedding_cache)
    truth = load_truth(truth_path)
    metadata = load_query_metadata(query_metadata_path)
    query_keys = list(truth)
    candidate_vectors = _project_store(model, candidates, candidates.keys, "protein", device)
    query_vectors = _project_store(model, queries, query_keys, "dna", device)
    scores = (query_vectors.to(device) @ candidate_vectors.to(device).T).float().cpu()
    evaluation = evaluate_score_matrix(
        scores,
        query_keys,
        candidates.keys,
        truth,
        recall_ks=(1, 5, 10),
        query_metadata=metadata,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(output / "metrics.json", evaluation.metrics)
    export_query_rows(output / "query_metrics.tsv", evaluation.query_rows)
    write_tsv(output / "recall_curve.tsv", recall_curve(evaluation.query_rows, 50), ["k", "recall"])
    checkpoint_sha = sha256_file(checkpoint_path)
    cache_id = sha256_text(canonical_json(candidates.manifest))[:20]
    sequences = {key: row["sequence"] for key, row in metadata.items() if "sequence" in row}
    candidate_metadata_path = candidates.manifest.get("input_path")
    candidate_metadata = (
        {row["protein_key"]: row for row in read_tsv(candidate_metadata_path)}
        if candidate_metadata_path and Path(candidate_metadata_path).exists()
        else None
    )
    export_dense_topk(
        output / f"dense_top{top_k}.tsv",
        scores,
        evaluation.ranks,
        query_keys,
        candidates.keys,
        top_k,
        checkpoint_sha[:20],
        cache_id,
        sequences=sequences,
        candidate_metadata=candidate_metadata,
    )
    return evaluation.metrics
