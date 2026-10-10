from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data import prepare_data
from .embeddings import precompute_embeddings
from .full_library import evaluate_full_library_checkpoints, evaluate_full_library_ensemble
from .reporting import generate_report
from .screening import build_candidate_index, evaluate_checkpoint, screen
from .training import train
from .utils import load_yaml, resolve_device


DEFAULT_CONFIG = "configs/v1.yaml"


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML 配置文件")


def _load_config(path: str) -> dict[str, Any]:
    config = load_yaml(path)
    for section in ("data", "backbones", "model", "training", "evaluation", "runtime"):
        if section not in config:
            raise ValueError(f"配置缺少 section: {section}")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rec-rank",
        description="RecRank：输入 DNA，在候选重组酶库中做 cosine dense ranking。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="清洗原始 TSV 并生成稀疏显式监督数据")
    _add_config(prepare_parser)
    prepare_parser.add_argument("--train-file")
    prepare_parser.add_argument("--test-file")
    prepare_parser.add_argument("--output-dir")
    prepare_parser.add_argument("--no-count-check", action="store_true")

    embed_parser = subparsers.add_parser("embed", help="冻结 backbone 并预计算 embedding cache")
    _add_config(embed_parser)
    embed_parser.add_argument("--modality", choices=("protein", "dna"), required=True)
    embed_parser.add_argument("--input", required=True)
    embed_parser.add_argument("--output", required=True)
    embed_parser.add_argument("--model")
    embed_parser.add_argument("--revision")
    embed_parser.add_argument("--batch-size", type=int)
    embed_parser.add_argument("--device")
    embed_parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    embed_parser.add_argument("--shard-size", type=int, default=8192)
    trust_group = embed_parser.add_mutually_exclusive_group()
    trust_group.add_argument("--trust-remote-code", action="store_true", default=None)
    trust_group.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")

    train_parser = subparsers.add_parser(
        "train", help="训练投影头并按配置在每个 epoch 做候选库检索评价"
    )
    _add_config(train_parser)
    train_parser.add_argument("--device")
    train_parser.add_argument("--run-dir")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--epochs", type=int)

    evaluate_parser = subparsers.add_parser("evaluate", help="独立评估一个 checkpoint")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--candidate-embeddings", required=True)
    evaluate_parser.add_argument("--query-embeddings", required=True)
    evaluate_parser.add_argument("--truth", required=True)
    evaluate_parser.add_argument("--query-metadata", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--device", default="cuda")
    evaluate_parser.add_argument("--top-k", type=int, default=10)

    full_evaluate_parser = subparsers.add_parser(
        "evaluate-full-library",
        help="在训练+测试蛋白并集上回顾性评估一个或多个 checkpoint",
    )
    full_evaluate_parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="可重复提供；每个 checkpoint 会写入独立子目录",
    )
    full_evaluate_parser.add_argument(
        "--train-candidate-embeddings", default="embeddings/train_proteins"
    )
    full_evaluate_parser.add_argument(
        "--test-candidate-embeddings", default="embeddings/test_candidates"
    )
    full_evaluate_parser.add_argument("--query-embeddings", default="embeddings/test_dnas")
    full_evaluate_parser.add_argument("--query-rc-embeddings")
    full_evaluate_parser.add_argument("--processed-dir", default="data/processed")
    full_evaluate_parser.add_argument("--output", required=True)
    full_evaluate_parser.add_argument("--device", default="cuda")
    full_evaluate_parser.add_argument("--query-batch-size", type=int, default=64)
    full_evaluate_parser.add_argument("--top-k", type=int, default=10)

    ensemble_parser = subparsers.add_parser(
        "evaluate-full-library-ensemble",
        help="在完整蛋白库上评价多个 checkpoint 的 cosine score 均值 ensemble",
    )
    ensemble_parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="至少重复提供两次；对各 checkpoint 的 cosine score 取算术平均",
    )
    ensemble_parser.add_argument(
        "--train-candidate-embeddings", default="embeddings/train_proteins"
    )
    ensemble_parser.add_argument(
        "--test-candidate-embeddings", default="embeddings/test_candidates"
    )
    ensemble_parser.add_argument("--query-embeddings", default="embeddings/test_dnas")
    ensemble_parser.add_argument("--query-rc-embeddings")
    ensemble_parser.add_argument("--processed-dir", default="data/processed")
    ensemble_parser.add_argument("--output", required=True)
    ensemble_parser.add_argument("--device", default="cuda")
    ensemble_parser.add_argument("--query-batch-size", type=int, default=64)
    ensemble_parser.add_argument("--top-k", type=int, default=10)

    for command in ("export", "build-index"):
        export_parser = subparsers.add_parser(command, help="用 best checkpoint 构建版本化候选蛋白索引")
        export_parser.add_argument("--checkpoint", required=True)
        export_parser.add_argument("--candidate-embeddings", required=True)
        export_parser.add_argument("--output", required=True)
        export_parser.add_argument("--device", default="cuda")

    screen_parser = subparsers.add_parser("screen", help="对一条或一批 DNA 输出 dense TOP10")
    screen_parser.add_argument("--checkpoint", required=True)
    screen_parser.add_argument("--candidate-index", required=True)
    screen_parser.add_argument("--output", required=True)
    screen_parser.add_argument("--device", default="cuda")
    screen_parser.add_argument("--top-k", type=int, default=10)
    query_group = screen_parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--dna")
    query_group.add_argument("--query-tsv")
    query_group.add_argument("--query-embedding-cache")

    report_parser = subparsers.add_parser("report", help="从结构化日志生成静态训练图与报告")
    report_parser.add_argument("--run-dir", required=True)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        config = _load_config(args.config)
        data_cfg = config["data"]
        summary = prepare_data(
            args.train_file or data_cfg["train_file"],
            args.test_file or data_cfg["test_file"],
            args.output_dir or data_cfg["processed_dir"],
        )
        if not args.no_count_check:
            mismatches = {
                key: {"expected": expected, "actual": summary.get(key)}
                for key, expected in data_cfg.get("expected_counts", {}).items()
                if summary.get(key) != expected
            }
            if mismatches:
                raise RuntimeError(
                    "数据准备结果违反已确认不变量；请先检查，不要继续训练：\n"
                    + json.dumps(mismatches, ensure_ascii=False, indent=2)
                )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "embed":
        config = _load_config(args.config)
        backbone_cfg = config["backbones"][args.modality]
        device = resolve_device(args.device or config["runtime"]["device"])
        manifest = precompute_embeddings(
            input_path=args.input,
            output_dir=args.output,
            modality=args.modality,
            model_name=args.model or backbone_cfg["model"],
            batch_size=args.batch_size or int(backbone_cfg["batch_size"]),
            device=device,
            revision=args.revision or backbone_cfg.get("revision"),
            trust_remote_code=(
                bool(backbone_cfg.get("trust_remote_code", False))
                if args.trust_remote_code is None
                else args.trust_remote_code
            ),
            output_dtype=args.dtype or config["backbones"]["embedding_dtype"],
            shard_size=args.shard_size,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    if args.command == "train":
        config = _load_config(args.config)
        if args.seed is not None:
            config["training"]["seed"] = args.seed
        if args.epochs is not None:
            config["training"]["epochs"] = args.epochs
        device = resolve_device(args.device or config["runtime"]["device"])
        run_path = train(config, device, run_dir=args.run_dir, resume=args.resume)
        print(f"训练运行目录: {run_path}")
        return

    if args.command == "evaluate":
        metrics = evaluate_checkpoint(
            args.checkpoint,
            args.candidate_embeddings,
            args.query_embeddings,
            args.truth,
            args.query_metadata,
            args.output,
            resolve_device(args.device),
            args.top_k,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    if args.command == "evaluate-full-library":
        results = evaluate_full_library_checkpoints(
            checkpoint_paths=args.checkpoint,
            train_candidate_cache=args.train_candidate_embeddings,
            test_candidate_cache=args.test_candidate_embeddings,
            query_cache=args.query_embeddings,
            processed_dir=args.processed_dir,
            output_dir=args.output,
            device=resolve_device(args.device),
            query_batch_size=args.query_batch_size,
            top_k=args.top_k,
            query_rc_cache=args.query_rc_embeddings,
        )
        compact = [
            {
                "label": result["label"],
                "checkpoint_epoch": result["checkpoint_epoch"],
                "candidate_count": result["candidate_count"],
                "exact_truth": result["exact_truth"],
                "direction_invariant_truth": result["direction_invariant_truth"],
            }
            for result in results
        ]
        print(json.dumps(compact, ensure_ascii=False, indent=2))
        return

    if args.command == "evaluate-full-library-ensemble":
        result = evaluate_full_library_ensemble(
            checkpoint_paths=args.checkpoint,
            train_candidate_cache=args.train_candidate_embeddings,
            test_candidate_cache=args.test_candidate_embeddings,
            query_cache=args.query_embeddings,
            processed_dir=args.processed_dir,
            output_dir=args.output,
            device=resolve_device(args.device),
            query_batch_size=args.query_batch_size,
            top_k=args.top_k,
            query_rc_cache=args.query_rc_embeddings,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command in {"export", "build-index"}:
        manifest = build_candidate_index(
            args.checkpoint,
            args.candidate_embeddings,
            args.output,
            resolve_device(args.device),
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    if args.command == "screen":
        count = screen(
            args.checkpoint,
            args.candidate_index,
            args.output,
            resolve_device(args.device),
            dna=args.dna,
            query_tsv=args.query_tsv,
            query_embedding_cache=args.query_embedding_cache,
            top_k=args.top_k,
        )
        print(f"已输出 {count} 条候选记录（dense rank <= {args.top_k}）: {args.output}")
        return

    if args.command == "report":
        generated = generate_report(args.run_dir)
        print("已生成：")
        for path in generated:
            print(path)
        return

    raise AssertionError(f"未处理命令: {args.command}")
