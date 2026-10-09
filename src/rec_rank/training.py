from __future__ import annotations

import csv
import json
import os
import platform
import random
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F

from .data import load_groups
from .embeddings import EmbeddingStore, load_embedding_store
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
from .sampler import EpochBatchSampler
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
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "median_rank",
    "novel_dna_recall_at_10",
    "novel_dna_mrr",
    "novel_dna_median_rank",
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


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device) -> RecRankModel:
    model = RecRankModel(**checkpoint["model_architecture"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device)


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


def _is_better(metrics: dict[str, Any], epoch: int, best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    candidate_key = (
        float(metrics["recall_at_10"]),
        float(metrics["mrr"]),
        -float(metrics["median_rank"]),
        -epoch,
    )
    best_metrics = best["metrics"]
    best_key = (
        float(best_metrics["recall_at_10"]),
        float(best_metrics["mrr"]),
        -float(best_metrics["median_rank"]),
        -int(best["epoch"]),
    )
    return candidate_key > best_key


def _project_store(
    model: RecRankModel,
    store: EmbeddingStore,
    keys: list[str],
    modality: str,
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(keys), batch_size):
            batch_keys = keys[start : start + batch_size]
            values = store.rows(batch_keys, device=device).float()
            projected = (
                model.project_proteins(values) if modality == "protein" else model.project_dnas(values)
            )
            outputs.append(projected.float().cpu())
    return torch.cat(outputs, dim=0)


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


def _log_tensorboard(writer: Any, row: dict[str, Any], epoch: int) -> None:
    for key, value in row.items():
        if key in {"epoch", "is_best"} or value in {"", None}:
            continue
        if isinstance(value, (int, float)) and np.isfinite(value):
            writer.add_scalar(key.replace("_", "/", 1), value, epoch)


def _runtime_manifest(config: dict[str, Any], fingerprints: dict[str, Any], run_id: str) -> dict[str, Any]:
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
    train_dna_store = load_embedding_store(runtime_cfg["dna_embedding_cache"])
    candidate_store = load_embedding_store(runtime_cfg["test_candidate_embedding_cache"])
    test_dna_store = load_embedding_store(runtime_cfg["test_dna_embedding_cache"])
    fingerprints = {
        "train_groups_sha256": sha256_file(processed / "train_groups.jsonl"),
        "test_truth_sha256": sha256_file(processed / "test_truth.tsv"),
        "train_protein_cache": train_protein_store.manifest,
        "train_dna_cache": train_dna_store.manifest,
        "test_candidate_cache": candidate_store.manifest,
        "test_dna_cache": test_dna_store.manifest,
    }

    groups = load_groups(processed / "train_groups.jsonl")
    sampler = EpochBatchSampler(groups, int(training_cfg["proteins_per_batch"]), seed)
    test_pairs = _load_pairs(processed / "test_pairs.tsv")
    truth = load_truth(processed / "test_truth.tsv")
    query_metadata = load_query_metadata(processed / "test_queries.tsv")
    query_keys = list(truth)
    candidate_keys = candidate_store.keys
    candidate_metadata = {
        row["protein_key"]: row for row in read_tsv(processed / "test_candidates.tsv")
    }
    if set(candidate_keys) != set(candidate_metadata):
        raise ValueError("test candidate embedding cache 与 prepared candidate 集合不一致")
    if not test_dna_store.contains_all(query_keys):
        raise ValueError("test DNA embedding cache 缺少 retrieval query")

    protein_dim = int(config["backbones"]["protein"]["hidden_dim"])
    dna_dim = int(config["backbones"]["dna"]["hidden_dim"])
    if train_protein_store.embeddings.shape[1] != protein_dim:
        raise ValueError("protein embedding 维度与配置不一致")
    if train_dna_store.embeddings.shape[1] != dna_dim:
        raise ValueError("DNA embedding 维度与配置不一致")
    for train_store, test_store, label in (
        (train_protein_store, candidate_store, "protein"),
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
        atomic_json_dump(run_path / "status.json", {"status": "running", "updated_at": utc_now_iso()})

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("缺少 tensorboard；请在 rec_rank 环境安装 requirements.txt") from exc
    writer = SummaryWriter(log_dir=run_path / "logs" / "tensorboard", purge_step=start_epoch if resume else None)
    recall_ks = tuple(int(value) for value in evaluation_cfg["recall_ks"])
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
            random_metrics = _random_baseline(
                query_keys, candidate_keys, truth, query_metadata, seed, recall_ks
            )
            atomic_json_dump(run_path / "evaluation" / "random_baseline.json", random_metrics)
            evaluation0, scores0 = _retrieval_evaluation(
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
            test_pair_metrics0, test_pos0, test_neg0 = _evaluate_labeled_pairs(
                model, test_pairs, candidate_store, test_dna_store, device
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
            candidate_space_id = sha256_text(canonical_json(candidate_store.manifest))[:20]
            export_dense_topk(
                run_path / "evaluation" / "best_dense_top10.tsv",
                scores0,
                evaluation0.ranks,
                query_keys,
                candidate_keys,
                10,
                checkpoint0_sha[:20],
                candidate_space_id,
                sequences={key: row.get("sequence", "") for key, row in query_metadata.items()},
                candidate_metadata=candidate_metadata,
            )
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
            total_loss_sum = 0.0
            pair_count = 0
            positive_loss_values: list[float] = []
            negative_loss_values: list[float] = []
            positive_cosines: list[float] = []
            negative_cosines: list[float] = []
            margins: list[float] = []
            gradient_norms: list[float] = []
            clipped_steps = 0
            clip_norm = float(training_cfg["gradient_clip_norm"])

            for batch in sampler.iter_batches(epoch):
                if stop_requested:
                    break
                proteins = train_protein_store.rows(
                    [pair.protein_key for pair in batch], device=device
                ).float()
                dnas = train_dna_store.rows([pair.dna_key for pair in batch], device=device).float()
                labels = torch.tensor(
                    [pair.label for pair in batch], device=device, dtype=torch.float32
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
                ):
                    loss, logits, cosine = model.loss(proteins, dnas, labels)
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
                total_loss_sum += float(loss.detach().cpu()) * len(batch)
                pair_count += len(batch)
                global_step += 1

            if stop_requested:
                break
            train_metrics = {
                "train_loss": total_loss_sum / pair_count,
                "train_positive_loss": float(np.mean(positive_loss_values)),
                "train_negative_loss": float(np.mean(negative_loss_values)),
                **_quantiles(positive_cosines, "train_positive_cosine"),
                **_quantiles(negative_cosines, "train_negative_cosine"),
                "train_pair_margin_mean": float(np.mean(margins)),
            }
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
            test_pair_metrics, test_positive_cosines, test_negative_cosines = _evaluate_labeled_pairs(
                model, test_pairs, candidate_store, test_dna_store, device
            )
            is_best = _is_better(evaluation.metrics, epoch, best)
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
                export_dense_topk(
                    run_path / "evaluation" / "best_dense_top10.tsv",
                    scores,
                    evaluation.ranks,
                    query_keys,
                    candidate_keys,
                    10,
                    best_checkpoint_sha[:20],
                    sha256_text(canonical_json(candidate_store.manifest))[:20],
                    sequences={
                        key: row.get("sequence", "") for key, row in query_metadata.items()
                    },
                    candidate_metadata=candidate_metadata,
                )
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
