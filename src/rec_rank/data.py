from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .utils import atomic_json_dump, read_tsv, write_tsv


NORMALIZATION_VERSION = "v1"
DNA_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("attP_str", "attP_pos", 1),
    ("attB_str", "attB_pos", 1),
    ("attP_neg", "attP_neg", 0),
    ("attB_neg", "attB_neg", 0),
)
CATEGORIES = ("attP_pos", "attB_pos", "attP_neg", "attB_neg")
IUPAC_TO_N = str.maketrans({base: "N" for base in "RYSWKMBDHV"})
DNA_ALLOWED = frozenset("ACGTN")
PROTEIN_ALLOWED = frozenset("ABCDEFGHIKLMNPQRSTUVWXYZ*-")


def stable_key(sequence: str, kind: str) -> str:
    prefix = {"protein": "P", "dna": "D"}[kind]
    return f"{prefix}_{hashlib.sha1(sequence.encode('ascii')).hexdigest()[:20]}"


def normalize_protein(value: str | None) -> str | None:
    if value is None:
        return None
    sequence = re.sub(r"\s+", "", value).upper()
    if not sequence or "FAILED" in sequence:
        return None
    if any(base not in PROTEIN_ALLOWED for base in sequence):
        return None
    return sequence


def normalize_dna(value: str | None) -> str | None:
    if value is None:
        return None
    sequence = re.sub(r"\s+", "", value).upper().replace("U", "T")
    # 必须先识别失败哨兵，再映射 IUPAC 字母，否则 FAILED 会被部分改写。
    if not sequence or "FAILED" in sequence:
        return None
    sequence = sequence.translate(IUPAC_TO_N)
    if any(base not in DNA_ALLOWED for base in sequence):
        return None
    return sequence


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


@dataclass(frozen=True)
class Membership:
    protein_key: str
    dna_key: str
    label: int
    category: str
    source_rows: tuple[int, ...]


@dataclass(frozen=True)
class RawTriple:
    """One intact supervision record from the source table."""

    source_row: int
    protein_key: str
    attp_pos_key: str
    attb_pos_key: str
    attp_neg_key: str
    attb_neg_key: str


def _collect_dataset(path: str | Path) -> dict[str, Any]:
    protein_sequences: dict[str, str] = {}
    dna_sequences: dict[str, str] = {}
    memberships: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    protein_clusters: dict[str, set[str]] = defaultdict(set)
    protein_ids: dict[str, set[str]] = defaultdict(set)
    raw_rows = 0
    invalid_slots = 0
    raw_triples: list[RawTriple] = []

    for source_row, row in enumerate(read_tsv(path), start=2):
        raw_rows += 1
        protein_sequence = normalize_protein(row.get("recombinases_str"))
        if protein_sequence is None:
            continue
        protein_key = stable_key(protein_sequence, "protein")
        protein_sequences[protein_key] = protein_sequence
        cluster = (row.get("recombinases_clu_id") or "").strip()
        if cluster:
            protein_clusters[protein_key].add(cluster)
        record_id = (row.get("recombinases") or "").strip()
        if record_id:
            protein_ids[protein_key].add(record_id)

        row_dna_keys: dict[str, str] = {}
        for column, category, label in DNA_COLUMNS:
            dna_sequence = normalize_dna(row.get(column))
            if dna_sequence is None:
                invalid_slots += 1
                continue
            dna_key = stable_key(dna_sequence, "dna")
            dna_sequences[dna_key] = dna_sequence
            row_dna_keys[category] = dna_key
            memberships[(protein_key, dna_key, category)].add(source_row)

        if all(category in row_dna_keys for category in CATEGORIES):
            raw_triples.append(
                RawTriple(
                    source_row=source_row,
                    protein_key=protein_key,
                    attp_pos_key=row_dna_keys["attP_pos"],
                    attb_pos_key=row_dna_keys["attB_pos"],
                    attp_neg_key=row_dna_keys["attP_neg"],
                    attb_neg_key=row_dna_keys["attB_neg"],
                )
            )

    materialized = [
        Membership(
            protein_key=protein_key,
            dna_key=dna_key,
            label=1 if category.endswith("_pos") else 0,
            category=category,
            source_rows=tuple(sorted(source_rows)),
        )
        for (protein_key, dna_key, category), source_rows in memberships.items()
    ]
    materialized.sort(key=lambda item: (item.protein_key, item.category, item.dna_key))
    return {
        "raw_rows": raw_rows,
        "invalid_slots": invalid_slots,
        "protein_sequences": protein_sequences,
        "dna_sequences": dna_sequences,
        "memberships": materialized,
        "raw_triples": raw_triples,
        "protein_clusters": protein_clusters,
        "protein_ids": protein_ids,
    }


def _find_conflicts(memberships: Iterable[Membership]) -> set[tuple[str, str]]:
    labels: dict[tuple[str, str], set[int]] = defaultdict(set)
    for item in memberships:
        labels[(item.protein_key, item.dna_key)].add(item.label)
    return {pair for pair, pair_labels in labels.items() if len(pair_labels) > 1}


def _write_memberships(path: Path, memberships: Iterable[Membership]) -> None:
    rows = (
        {
            "protein_key": item.protein_key,
            "dna_key": item.dna_key,
            "label": item.label,
            "category": item.category,
            "source_rows": ",".join(map(str, item.source_rows)),
        }
        for item in memberships
    )
    write_tsv(path, rows, ["protein_key", "dna_key", "label", "category", "source_rows"])


def _write_reverse_complement_inputs(
    output: Path,
    prefix: str,
    source_sequences: dict[str, str],
    primary_keys: set[str] | None = None,
) -> dict[str, int]:
    """写出方向映射，以及主 cache 中不存在、需要额外编码的反向互补序列。"""
    source_keys = primary_keys if primary_keys is not None else set(source_sequences)
    supplement: dict[str, str] = {}
    mapping_rows = []
    palindrome_count = 0
    for dna_key in sorted(source_sequences):
        rc_sequence = reverse_complement(source_sequences[dna_key])
        rc_key = stable_key(rc_sequence, "dna")
        palindrome_count += int(rc_key == dna_key)
        if rc_key not in source_keys:
            supplement[rc_key] = rc_sequence
        mapping_rows.append(
            {
                "dna_key": dna_key,
                "reverse_complement_dna_key": rc_key,
                "reverse_complement_in_primary_cache": int(rc_key in source_keys),
            }
        )
    write_tsv(
        output / f"{prefix}_rc_map.tsv",
        mapping_rows,
        ["dna_key", "reverse_complement_dna_key", "reverse_complement_in_primary_cache"],
    )
    write_tsv(
        output / f"{prefix}_rc_only.tsv",
        (
            {"dna_key": key, "sequence": supplement[key]}
            for key in sorted(supplement)
        ),
        ["dna_key", "sequence"],
    )
    return {
        f"{prefix}_rc_mapping_count": len(mapping_rows),
        f"{prefix}_rc_only_count": len(supplement),
        f"{prefix}_palindrome_count": palindrome_count,
    }


def _write_dna_ranking_groups(
    path: Path, memberships: Iterable[Membership]
) -> dict[str, int]:
    """写出同一 DNA 下显式正、负蛋白集合；unknown 组合不进入文件。"""
    labels: dict[str, dict[int, set[str]]] = defaultdict(
        lambda: {0: set(), 1: set()}
    )
    for item in memberships:
        labels[item.dna_key][item.label].add(item.protein_key)

    eligible = [dna_key for dna_key, values in labels.items() if values[0] and values[1]]
    positive_pairs = 0
    negative_pairs = 0
    triplets = 0
    with path.open("w", encoding="utf-8") as handle:
        for dna_key in sorted(eligible):
            positives = sorted(labels[dna_key][1])
            negatives = sorted(labels[dna_key][0])
            positive_pairs += len(positives)
            negative_pairs += len(negatives)
            triplet_count = len(positives) * len(negatives)
            triplets += triplet_count
            handle.write(
                json.dumps(
                    {
                        "dna_key": dna_key,
                        "positive_protein_keys": positives,
                        "negative_protein_keys": negatives,
                        "triplet_count": triplet_count,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return {
        "train_dnas_with_positive_proteins": sum(bool(values[1]) for values in labels.values()),
        "train_dnas_with_negative_proteins": sum(bool(values[0]) for values in labels.values()),
        "train_dna_ranking_queries": len(eligible),
        "train_dna_ranking_positive_pairs": positive_pairs,
        "train_dna_ranking_negative_pairs": negative_pairs,
        "train_dna_ranking_triplets": triplets,
    }


def prepare_data(train_path: str | Path, test_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = _collect_dataset(train_path)
    test = _collect_dataset(test_path)

    train_conflicts = _find_conflicts(train["memberships"])
    train_memberships = [
        item for item in train["memberships"] if (item.protein_key, item.dna_key) not in train_conflicts
    ]
    supervised_train_proteins = {item.protein_key for item in train_memberships}
    supervised_train_dnas = {item.dna_key for item in train_memberships}
    train_triples = [
        triple
        for triple in train["raw_triples"]
        if all(
            (triple.protein_key, dna_key) not in train_conflicts
            for dna_key in (
                triple.attp_pos_key,
                triple.attb_pos_key,
                triple.attp_neg_key,
                triple.attb_neg_key,
            )
        )
    ]

    write_tsv(
        output / "train_proteins.tsv",
        (
            {
                "protein_key": key,
                "sequence": train["protein_sequences"][key],
                "record_ids": ";".join(sorted(train["protein_ids"].get(key, set()))),
                "cluster_ids": ";".join(sorted(train["protein_clusters"].get(key, set()))),
            }
            for key in sorted(supervised_train_proteins)
        ),
        ["protein_key", "sequence", "record_ids", "cluster_ids"],
    )
    write_tsv(
        output / "train_dnas.tsv",
        ({"dna_key": key, "sequence": train["dna_sequences"][key]} for key in sorted(supervised_train_dnas)),
        ["dna_key", "sequence"],
    )
    train_rc_summary = _write_reverse_complement_inputs(
        output,
        "train_dnas",
        {key: train["dna_sequences"][key] for key in supervised_train_dnas},
    )
    _write_memberships(output / "train_pairs.tsv", train_memberships)

    conflict_rows = []
    for protein_key, dna_key in sorted(train_conflicts):
        conflict_memberships = [
            item
            for item in train["memberships"]
            if item.protein_key == protein_key and item.dna_key == dna_key
        ]
        conflict_rows.append(
            {
                "protein_key": protein_key,
                "dna_key": dna_key,
                "sequence": train["dna_sequences"][dna_key],
                "categories": ",".join(item.category for item in conflict_memberships),
                "source_rows": ",".join(
                    str(row) for item in conflict_memberships for row in item.source_rows
                ),
            }
        )
    write_tsv(
        output / "train_conflicts.tsv",
        conflict_rows,
        ["protein_key", "dna_key", "sequence", "categories", "source_rows"],
    )

    grouped: dict[str, dict[str, set[str]]] = {
        key: {category: set() for category in CATEGORIES} for key in supervised_train_proteins
    }
    for item in train_memberships:
        grouped[item.protein_key][item.category].add(item.dna_key)
    groups_path = output / "train_groups.jsonl"
    with groups_path.open("w", encoding="utf-8") as handle:
        for protein_key in sorted(grouped):
            record: dict[str, Any] = {"protein_key": protein_key}
            record.update({category: sorted(grouped[protein_key][category]) for category in CATEGORIES})
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    triples_path = output / "train_triples.jsonl"
    with triples_path.open("w", encoding="utf-8") as handle:
        for triple in train_triples:
            handle.write(
                json.dumps(
                    {
                        "source_row": triple.source_row,
                        "protein_key": triple.protein_key,
                        "attp_pos_key": triple.attp_pos_key,
                        "attb_pos_key": triple.attb_pos_key,
                        "attp_neg_key": triple.attp_neg_key,
                        "attb_neg_key": triple.attb_neg_key,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    dna_ranking_summary = _write_dna_ranking_groups(
        output / "train_dna_ranking_groups.jsonl", train_memberships
    )

    # 测试候选空间由测试文件中的全部 unique protein sequence 构成。
    test_candidate_keys = set(test["protein_sequences"])
    write_tsv(
        output / "test_candidates.tsv",
        (
            {
                "protein_key": key,
                "sequence": test["protein_sequences"][key],
                "record_ids": ";".join(sorted(test["protein_ids"].get(key, set()))),
                "cluster_ids": ";".join(sorted(test["protein_clusters"].get(key, set()))),
            }
            for key in sorted(test_candidate_keys)
        ),
        ["protein_key", "sequence", "record_ids", "cluster_ids"],
    )

    test_conflicts = _find_conflicts(test["memberships"])
    test_memberships = [
        item for item in test["memberships"] if (item.protein_key, item.dna_key) not in test_conflicts
    ]
    test_labeled_dnas = {item.dna_key for item in test_memberships}
    write_tsv(
        output / "test_dnas.tsv",
        ({"dna_key": key, "sequence": test["dna_sequences"][key]} for key in sorted(test_labeled_dnas)),
        ["dna_key", "sequence"],
    )
    _write_memberships(output / "test_pairs.tsv", test_memberships)

    truth: dict[str, set[str]] = defaultdict(set)
    for item in test_memberships:
        if item.label == 1:
            truth[item.dna_key].add(item.protein_key)

    train_positive_sequences = {
        train["dna_sequences"][item.dna_key] for item in train_memberships if item.label == 1
    }
    train_positive_both_directions = train_positive_sequences | {
        reverse_complement(sequence) for sequence in train_positive_sequences
    }
    query_rows = []
    truth_rows = []
    for dna_key in sorted(truth):
        sequence = test["dna_sequences"][dna_key]
        positive_keys = sorted(truth[dna_key])
        is_exact_seen = sequence in train_positive_sequences
        is_direction_seen = sequence in train_positive_both_directions
        query_rows.append(
            {
                "dna_key": dna_key,
                "sequence": sequence,
                "positive_count": len(positive_keys),
                "exact_seen_in_train_positive": int(is_exact_seen),
                "direction_seen_in_train_positive": int(is_direction_seen),
                "is_novel_dna": int(not is_direction_seen),
            }
        )
        truth_rows.append(
            {"dna_key": dna_key, "positive_protein_keys": ";".join(positive_keys)}
        )
    write_tsv(
        output / "test_queries.tsv",
        query_rows,
        [
            "dna_key",
            "sequence",
            "positive_count",
            "exact_seen_in_train_positive",
            "direction_seen_in_train_positive",
            "is_novel_dna",
        ],
    )
    write_tsv(output / "test_truth.tsv", truth_rows, ["dna_key", "positive_protein_keys"])
    test_query_rc_summary = _write_reverse_complement_inputs(
        output,
        "test_queries",
        {row["dna_key"]: row["sequence"] for row in query_rows},
        primary_keys=test_labeled_dnas,
    )

    train_unique_pairs = {(item.protein_key, item.dna_key) for item in train_memberships}
    test_unique_pairs = {(item.protein_key, item.dna_key) for item in test_memberships}
    train_positive_pairs = {
        (item.protein_key, item.dna_key) for item in train_memberships if item.label == 1
    }
    train_negative_pairs = train_unique_pairs - train_positive_pairs
    test_truth_pairs = {(protein_key, dna_key) for dna_key, keys in truth.items() for protein_key in keys}
    multi_positive_queries = sum(len(keys) > 1 for keys in truth.values())
    overlap_proteins = test_candidate_keys & supervised_train_proteins
    summary = {
        "normalization_version": NORMALIZATION_VERSION,
        "train_raw_rows": train["raw_rows"],
        "train_invalid_dna_slots": train["invalid_slots"],
        "train_unique_proteins_raw": len(train["protein_sequences"]),
        "train_unique_proteins": len(supervised_train_proteins),
        "train_unique_dnas": len(supervised_train_dnas),
        "train_category_memberships": len(train_memberships),
        "train_unique_labeled_pairs": len(train_unique_pairs),
        "train_positive_pairs": len(train_positive_pairs),
        "train_negative_pairs": len(train_negative_pairs),
        "train_conflicting_pairs_removed": len(train_conflicts),
        "train_complete_triples_raw": len(train["raw_triples"]),
        "train_supervised_triples": len(train_triples),
        "train_triple_unique_proteins": len(
            {triple.protein_key for triple in train_triples}
        ),
        "train_triples_removed_by_pair_conflict": (
            len(train["raw_triples"]) - len(train_triples)
        ),
        **dna_ranking_summary,
        "test_raw_rows": test["raw_rows"],
        "test_unique_candidates": len(test_candidate_keys),
        "test_unique_queries": len(truth),
        "test_truth_pairs": len(test_truth_pairs),
        "test_multi_positive_queries": multi_positive_queries,
        "test_unique_labeled_pairs": len(test_unique_pairs),
        "test_conflicting_pairs_removed": len(test_conflicts),
        "test_train_exact_protein_overlap": len(overlap_proteins),
        "test_query_exact_positive_overlap": sum(
            row["exact_seen_in_train_positive"] for row in query_rows
        ),
        "test_query_direction_independent_overlap": sum(
            row["direction_seen_in_train_positive"] for row in query_rows
        ),
        **train_rc_summary,
        **test_query_rc_summary,
        "test_novel_dna_queries": sum(row["is_novel_dna"] for row in query_rows),
    }
    atomic_json_dump(output / "summary.json", summary)
    return summary


def load_groups(path: str | Path) -> list[dict[str, Any]]:
    groups = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if "protein_key" not in value:
                raise ValueError(f"{path}:{line_number} 缺少 protein_key")
            for category in CATEGORIES:
                value.setdefault(category, [])
            groups.append(value)
    return groups
