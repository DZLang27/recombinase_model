from __future__ import annotations

import csv
import copy
import json
from pathlib import Path

import torch

from rec_rank.data import prepare_data
from rec_rank.embeddings import load_embedding_store, save_embedding_store
from rec_rank.full_library import (
    evaluate_full_library_checkpoints,
    evaluate_full_library_ensemble,
)
from rec_rank.reporting import generate_report
from rec_rank.screening import build_candidate_index, screen
from rec_rank.training import train
from rec_rank.utils import read_tsv


HEADER = [
    "attB", "attP", "recombinases", "attB_str", "attP_str", "recombinases_str",
    "center_common", "resource", "attB_clu_id", "attP_clu_id", "recombinases_clu_id",
    "attB_neg", "attP_neg",
]


def _raw_row(index: int, protein: str, bases: list[str]) -> list[str]:
    return [
        f"b{index}", f"p{index}", f"r{index}", bases[0], bases[1], protein,
        "", "test", f"bc{index}", f"pc{index}", f"rc{index}", bases[2], bases[3],
    ]


def _write(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)


def _cache_from_tsv(path: Path, key_column: str, output: Path, dim: int, seed: int) -> None:
    rows = list(read_tsv(path))
    generator = torch.Generator().manual_seed(seed)
    save_embedding_store(
        output,
        [row[key_column] for row in rows],
        torch.randn(len(rows), dim, generator=generator),
        {"kind": "synthetic_test_backbone", "source": str(path)},
    )


def test_synthetic_training_checkpoint_index_and_screen(tmp_path: Path) -> None:
    train_raw = tmp_path / "train.tsv"
    test_raw = tmp_path / "test.tsv"
    rows = [
        _raw_row(1, "MKTAA", ["AAAA", "CCCC", "GGGG", "TTTT"]),
        _raw_row(2, "MKTVV", ["ACAC", "CACA", "GTGT", "TGTG"]),
        # 同一 DNA 在不同蛋白下拥有显式正、负标注，使 V1.2 可以构造
        # DNA-centric ranking；这些序列的反向互补也都已出现在主 cache。
        _raw_row(3, "MKLGG", ["GGGG", "TTTT", "AAAA", "CCCC"]),
    ]
    _write(train_raw, rows)
    _write(test_raw, rows)
    processed = tmp_path / "processed"
    prepare_data(train_raw, test_raw, processed)

    embeddings = tmp_path / "embeddings"
    _cache_from_tsv(processed / "train_proteins.tsv", "protein_key", embeddings / "train_p", 4, 1)
    _cache_from_tsv(processed / "train_dnas.tsv", "dna_key", embeddings / "train_d", 3, 2)
    _cache_from_tsv(processed / "test_candidates.tsv", "protein_key", embeddings / "test_p", 4, 3)
    _cache_from_tsv(processed / "test_dnas.tsv", "dna_key", embeddings / "test_d", 3, 4)

    config = {
        "data": {"processed_dir": str(processed)},
        "backbones": {
            "protein": {"hidden_dim": 4},
            "dna": {"hidden_dim": 3, "model": "unused", "batch_size": 2, "trust_remote_code": False},
        },
        "model": {
            "projection_hidden_dim": 8,
            "projection_dim": 5,
            "dropout": 0.0,
            "gamma_init": 5.0,
            "gamma_epsilon": 1e-6,
            "gamma_warning_threshold": 100.0,
        },
        "training": {
            "epochs": 2,
            "proteins_per_batch": 2,
            "projection_learning_rate": 1e-2,
            "gamma_learning_rate": 1e-3,
            "projection_weight_decay": 1e-4,
            "gamma_weight_decay": 1e-2,
            "gradient_clip_norm": 1.0,
            "amp": False,
            "seed": 42,
            "num_workers": 0,
            "scheduler": "none",
        },
        "evaluation": {
            "recall_ks": [1, 5, 10],
            "recall_curve_max_k": 5,
            "primary_metric": "recall_at_10",
            "selection_tiebreak": ["recall_at_10", "mrr", "median_rank", "earlier_epoch"],
            "report_novel_dna_subset": True,
        },
        "runtime": {
            "run_root": str(tmp_path / "runs"),
            "protein_embedding_cache": str(embeddings / "train_p"),
            "dna_embedding_cache": str(embeddings / "train_d"),
            "test_candidate_embedding_cache": str(embeddings / "test_p"),
            "test_dna_embedding_cache": str(embeddings / "test_d"),
            "device": "cpu",
        },
    }
    run_dir = train(config, torch.device("cpu"))
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert len(list(csv.DictReader((run_dir / "logs" / "history.csv").open()))) == 3
    best = json.loads((run_dir / "evaluation" / "best_metrics.json").read_text())
    assert 0 <= best["metrics"]["recall_at_10"] <= 1
    generated = generate_report(run_dir)
    assert (run_dir / "figures" / "REPORT.md") in generated

    index_dir = tmp_path / "candidate_index"
    build_candidate_index(
        run_dir / "checkpoints" / "best.pt", embeddings / "test_p", index_dir, torch.device("cpu")
    )
    output = tmp_path / "screen.tsv"
    count = screen(
        run_dir / "checkpoints" / "best.pt",
        index_dir,
        output,
        torch.device("cpu"),
        query_embedding_cache=embeddings / "test_d",
        top_k=1,
    )
    assert count >= len(list(read_tsv(processed / "test_dnas.tsv")))
    assert output.exists()

    # 完整库评价会对训练/测试候选按 protein_key 去重；重复 key 的 backbone
    # embedding 必须一致。这里单独构造一个满足该真实数据不变量的测试 cache。
    train_protein_store = load_embedding_store(embeddings / "train_p")
    test_candidate_keys = [
        row["protein_key"] for row in read_tsv(processed / "test_candidates.tsv")
    ]
    full_eval_test_cache = embeddings / "test_p_full_eval"
    save_embedding_store(
        full_eval_test_cache,
        test_candidate_keys,
        train_protein_store.rows(test_candidate_keys),
        {"kind": "synthetic_test_backbone", "source": "full-library-test"},
    )
    full_output = tmp_path / "full_library_eval"
    full_results = evaluate_full_library_checkpoints(
        [run_dir / "checkpoints" / "best.pt"],
        embeddings / "train_p",
        full_eval_test_cache,
        embeddings / "test_d",
        processed,
        full_output,
        torch.device("cpu"),
        query_batch_size=2,
        top_k=1,
    )
    assert full_results[0]["candidate_count"] == 3
    assert (full_output / "comparison.tsv").exists()
    assert (full_output / "evaluation_scope.json").exists()
    assert (full_output / full_results[0]["label"] / "metrics.json").exists()

    v11_config = copy.deepcopy(config)
    v11_config["training"].update(
        {
            "epochs": 1,
            "ranking_loss_weight": 0.25,
            "ranking_margin": 0.1,
            "reverse_complement_augmentation_fraction": 1.0,
        }
    )
    v11_config["evaluation"].update(
        {
            "candidate_library": "train_test_union",
            "truth_mode": "direction_invariant",
            "reverse_complement_ensemble": True,
            "query_batch_size": 2,
            "primary_metric": "recall_at_10",
            "selection_tiebreak": [
                "recall_at_10",
                "mrr",
                "median_rank",
                "earlier_epoch",
            ],
        }
    )
    v11_config["runtime"].update(
        {
            "run_root": str(tmp_path / "runs_v11"),
            "test_candidate_embedding_cache": str(full_eval_test_cache),
        }
    )
    v11_run = train(v11_config, torch.device("cpu"))
    v11_history = list(csv.DictReader((v11_run / "logs" / "history.csv").open()))
    assert float(v11_history[-1]["train_ranking_triplets"]) > 0
    assert float(v11_history[-1]["train_rc_augmentation_fraction"]) == 1.0
    assert float(v11_history[-1]["train_rc_added_pairs"]) > 0
    v11_best = json.loads((v11_run / "evaluation" / "best_metrics.json").read_text())
    assert v11_best["metrics"]["candidates"] == 3
    assert "exact_recall_at_10" in v11_best["metrics"]
    v11_scope = json.loads(
        (v11_run / "evaluation" / "evaluation_scope.json").read_text()
    )
    assert v11_scope["candidate_count"] == 3

    v12_config = copy.deepcopy(v11_config)
    v12_config["model"]["version"] = "v1.2-frozen"
    v12_config["training"].update(
        {
            "protein_ranking_loss_type": "smooth",
            "protein_ranking_group_balanced": True,
            "dna_ranking_loss_weight": 1.0,
            "dna_ranking_margin": 0.1,
            "dna_ranking_loss_type": "smooth",
            "dna_ranking_queries_per_batch": 8,
            "dna_ranking_use_reverse_complement": True,
            "rc_consistency_loss_weight": 0.1,
        }
    )
    v12_config["runtime"]["run_root"] = str(tmp_path / "runs_v12")
    v12_run = train(v12_config, torch.device("cpu"))
    v12_history = list(csv.DictReader((v12_run / "logs" / "history.csv").open()))
    assert float(v12_history[-1]["train_dna_ranking_groups"]) == 4
    assert float(v12_history[-1]["train_dna_ranking_triplets"]) == 4
    assert float(v12_history[-1]["train_dna_ranking_comparisons"]) == 8
    assert float(v12_history[-1]["train_dna_ranking_loss"]) > 0
    assert float(v12_history[-1]["train_rc_consistency_loss"]) >= 0
    v12_scope = json.loads(
        (v12_run / "evaluation" / "evaluation_scope.json").read_text()
    )
    assert v12_scope["dna_ranking_supervision"]["query_count"] == 4
    assert v12_scope["dna_ranking_supervision"]["unknown_pairs_used_as_negatives"] is False
    generated_v12 = generate_report(v12_run)
    assert (v12_run / "figures" / "dna_ranking_and_rc_consistency.png") in generated_v12

    # V1.3 returns to the exact V1.1 objective and changes only the two
    # independent projection heads. Exercise config -> checkpoint -> reload.
    v13_config = copy.deepcopy(v11_config)
    v13_config["model"].update(
        {
            "version": "v1.3-projection",
            "projection_head_type": "residual",
            "projection_hidden_dim": 6,
            "projection_dim": 6,
            "projection_residual_blocks": 2,
            "projection_expansion_dim": 12,
            "projection_layer_scale_init": 0.1,
        }
    )
    v13_config["runtime"]["run_root"] = str(tmp_path / "runs_v13")
    v13_run = train(v13_config, torch.device("cpu"))
    v13_checkpoint = torch.load(
        v13_run / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert v13_checkpoint["model_architecture"]["projection_head_type"] == "residual"
    assert v13_checkpoint["model_architecture"]["projection_dim"] == 6
    assert len(v13_checkpoint["model_state"]["protein_projection.blocks.0.layer_scale"]) == 6

    ensemble_output = tmp_path / "ensemble_eval"
    ensemble_result = evaluate_full_library_ensemble(
        checkpoint_paths=[
            v11_run / "checkpoints" / "best.pt",
            v13_run / "checkpoints" / "best.pt",
        ],
        train_candidate_cache=embeddings / "train_p",
        test_candidate_cache=full_eval_test_cache,
        query_cache=embeddings / "test_d",
        processed_dir=processed,
        output_dir=ensemble_output,
        device=torch.device("cpu"),
        query_batch_size=2,
        top_k=1,
    )
    assert ensemble_result["checkpoint_count"] == 2
    assert ensemble_result["aggregation"] == (
        "arithmetic_mean_of_per_checkpoint_cosine_scores"
    )
    assert ensemble_result["direction_invariant_truth"]["candidates"] == 3
    assert (ensemble_output / "metrics.json").exists()
    assert (ensemble_output / "query_metrics_direction_invariant.tsv").exists()
    assert (ensemble_output / "dense_top1.tsv").exists()
