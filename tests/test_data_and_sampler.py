from __future__ import annotations

import csv
from pathlib import Path

from rec_rank.data import normalize_dna, prepare_data, stable_key
from rec_rank.sampler import EpochBatchSampler, sample_group


HEADER = [
    "attB",
    "attP",
    "recombinases",
    "attB_str",
    "attP_str",
    "recombinases_str",
    "center_common",
    "resource",
    "attB_clu_id",
    "attP_clu_id",
    "recombinases_clu_id",
    "attB_neg",
    "attP_neg",
]


def _write_raw(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)


def _row(protein: str, att_b: str, att_p: str, neg_b: str, neg_p: str, suffix: str) -> list[str]:
    return [
        f"b{suffix}",
        f"p{suffix}",
        f"r{suffix}",
        att_b,
        att_p,
        protein,
        "",
        "test",
        f"bc{suffix}",
        f"pc{suffix}",
        f"rc{suffix}",
        neg_b,
        neg_p,
    ]


def test_normalize_failed_before_iupac_mapping() -> None:
    assert normalize_dna("FailedFailed") is None
    assert normalize_dna("acguRYSWKMBDHV") == "ACGTNNNNNNNNNN"
    assert normalize_dna("ACGT-X") is None


def test_prepare_removes_only_conflicting_pair(tmp_path: Path) -> None:
    protein = "MKTAA"
    train = tmp_path / "train.tsv"
    test = tmp_path / "test.tsv"
    # AAAA 在第一行是正样本、第二行是负样本；其他监督仍应保留。
    _write_raw(
        train,
        [
            _row(protein, "AAAA", "CCCC", "GGGG", "TTTT", "1"),
            _row(protein, "ACAC", "CACA", "AAAA", "TGTG", "2"),
        ],
    )
    _write_raw(test, [_row(protein, "AAAA", "CCCC", "GGGG", "TTTT", "3")])
    summary = prepare_data(train, test, tmp_path / "processed")
    assert summary["train_conflicting_pairs_removed"] == 1
    assert summary["train_category_memberships"] == 6
    assert summary["test_unique_queries"] == 2
    conflict = (tmp_path / "processed" / "train_conflicts.tsv").read_text()
    assert stable_key("AAAA", "dna") in conflict


def test_sampler_rotates_and_avoids_duplicate_pair() -> None:
    group = {
        "protein_key": "P1",
        "attP_pos": ["D1", "D2"],
        "attB_pos": ["D1"],
        "attP_neg": ["D3"],
        "attB_neg": ["D4"],
    }
    epoch1 = sample_group(group, epoch=1, group_index=0)
    epoch2 = sample_group(group, epoch=2, group_index=0)
    assert len({pair.dna_key for pair in epoch1}) == len(epoch1)
    assert {pair.dna_key for pair in epoch1} != {pair.dna_key for pair in epoch2}
    sampler_a = EpochBatchSampler([group, {**group, "protein_key": "P2"}], 1, seed=42)
    sampler_b = EpochBatchSampler([group, {**group, "protein_key": "P2"}], 1, seed=42)
    assert list(sampler_a.iter_batches(3)) == list(sampler_b.iter_batches(3))
