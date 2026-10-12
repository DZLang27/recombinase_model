from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .data import CATEGORIES


@dataclass(frozen=True)
class SampledPair:
    protein_key: str
    dna_key: str
    label: int
    category: str
    group_index: int


@dataclass(frozen=True)
class TripleSample:
    source_row: int
    protein_key: str
    attp_pos_key: str
    attb_pos_key: str
    attp_neg_key: str
    attb_neg_key: str
    group_index: int = -1

    @property
    def dna_keys(self) -> tuple[str, str, str, str]:
        return (
            self.attp_pos_key,
            self.attb_pos_key,
            self.attp_neg_key,
            self.attb_neg_key,
        )


def load_triples(path: str | Path) -> list[TripleSample]:
    triples: list[TripleSample] = []
    required = (
        "source_row",
        "protein_key",
        "attp_pos_key",
        "attb_pos_key",
        "attp_neg_key",
        "attb_neg_key",
    )
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(f"{path}:{line_number} 缺少字段: {missing}")
            triples.append(
                TripleSample(
                    source_row=int(row["source_row"]),
                    protein_key=row["protein_key"],
                    attp_pos_key=row["attp_pos_key"],
                    attb_pos_key=row["attb_pos_key"],
                    attp_neg_key=row["attp_neg_key"],
                    attb_neg_key=row["attb_neg_key"],
                )
            )
    return triples


def flatten_triple_batch(triples: list[TripleSample]) -> list[SampledPair]:
    pairs: list[SampledPair] = []
    for triple in triples:
        pairs.extend(
            [
                SampledPair(
                    triple.protein_key,
                    triple.attp_pos_key,
                    1,
                    "attP_pos",
                    triple.group_index,
                ),
                SampledPair(
                    triple.protein_key,
                    triple.attb_pos_key,
                    1,
                    "attB_pos",
                    triple.group_index,
                ),
                SampledPair(
                    triple.protein_key,
                    triple.attp_neg_key,
                    0,
                    "attP_neg",
                    triple.group_index,
                ),
                SampledPair(
                    triple.protein_key,
                    triple.attb_neg_key,
                    0,
                    "attB_neg",
                    triple.group_index,
                ),
            ]
        )
    return pairs


class TripleEpochBatchSampler:
    """Sample intact source triples while keeping at most one row per protein/batch."""

    def __init__(self, triples: list[TripleSample], proteins_per_batch: int, seed: int) -> None:
        if proteins_per_batch <= 0:
            raise ValueError("proteins_per_batch 必须大于 0")
        grouped: dict[str, list[TripleSample]] = {}
        for triple in triples:
            grouped.setdefault(triple.protein_key, []).append(triple)
        self.protein_keys = sorted(grouped)
        self.triples_by_protein = {
            key: sorted(grouped[key], key=lambda value: value.source_row)
            for key in self.protein_keys
        }
        self.proteins_per_batch = proteins_per_batch
        self.seed = seed

    def epoch_triples(self, epoch: int) -> list[TripleSample]:
        selected: list[TripleSample] = []
        for group_index, protein_key in enumerate(self.protein_keys):
            candidates = self.triples_by_protein[protein_key]
            value = candidates[(epoch - 1) % len(candidates)]
            selected.append(
                TripleSample(
                    source_row=value.source_row,
                    protein_key=value.protein_key,
                    attp_pos_key=value.attp_pos_key,
                    attb_pos_key=value.attb_pos_key,
                    attp_neg_key=value.attp_neg_key,
                    attb_neg_key=value.attb_neg_key,
                    group_index=group_index,
                )
            )
        return selected

    def iter_triple_batches(self, epoch: int) -> Iterator[list[TripleSample]]:
        selected = self.epoch_triples(epoch)
        random.Random(self.seed + epoch).shuffle(selected)
        for start in range(0, len(selected), self.proteins_per_batch):
            yield selected[start : start + self.proteins_per_batch]

    def iter_batches(self, epoch: int) -> Iterator[list[SampledPair]]:
        for triples in self.iter_triple_batches(epoch):
            yield flatten_triple_batch(triples)

    def epoch_pairs(self, epoch: int) -> list[SampledPair]:
        return flatten_triple_batch(self.epoch_triples(epoch))


@dataclass(frozen=True)
class DnaRankingGroup:
    dna_key: str
    positive_protein_keys: tuple[str, ...]
    negative_protein_keys: tuple[str, ...]

    @property
    def triplet_count(self) -> int:
        return len(self.positive_protein_keys) * len(self.negative_protein_keys)


def load_dna_ranking_groups(path: str | Path) -> list[DnaRankingGroup]:
    groups: list[DnaRankingGroup] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            positives = tuple(row.get("positive_protein_keys", []))
            negatives = tuple(row.get("negative_protein_keys", []))
            if not positives or not negatives:
                raise ValueError(f"{path}:{line_number} 缺少显式正蛋白或负蛋白")
            overlap = set(positives) & set(negatives)
            if overlap:
                raise ValueError(
                    f"{path}:{line_number} 同一 DNA 的正负蛋白冲突: {sorted(overlap)[:3]}"
                )
            group = DnaRankingGroup(
                dna_key=row["dna_key"],
                positive_protein_keys=positives,
                negative_protein_keys=negatives,
            )
            expected = row.get("triplet_count")
            if expected is not None and int(expected) != group.triplet_count:
                raise ValueError(f"{path}:{line_number} triplet_count 不一致")
            groups.append(group)
    return groups


class DnaRankingBatchSampler:
    def __init__(self, groups: list[DnaRankingGroup], queries_per_batch: int, seed: int) -> None:
        if queries_per_batch <= 0:
            raise ValueError("queries_per_batch 必须大于 0")
        self.groups = groups
        self.queries_per_batch = queries_per_batch
        self.seed = seed

    def iter_batches(self, epoch: int) -> Iterator[list[DnaRankingGroup]]:
        indices = list(range(len(self.groups)))
        random.Random(self.seed + 1_000_003 + epoch).shuffle(indices)
        for start in range(0, len(indices), self.queries_per_batch):
            yield [self.groups[index] for index in indices[start : start + self.queries_per_batch]]


def sample_group(group: dict[str, Any], epoch: int, group_index: int) -> list[SampledPair]:
    """每类最多取一条，按 epoch 确定性轮换，并避免同 epoch 重复 pair。"""
    used_dnas: set[str] = set()
    sampled: list[SampledPair] = []
    for category in CATEGORIES:
        candidates = sorted(set(group.get(category, [])))
        if not candidates:
            continue
        start = epoch % len(candidates)
        selected = None
        for offset in range(len(candidates)):
            candidate = candidates[(start + offset) % len(candidates)]
            if candidate not in used_dnas:
                selected = candidate
                break
        if selected is None:
            continue
        used_dnas.add(selected)
        sampled.append(
            SampledPair(
                protein_key=group["protein_key"],
                dna_key=selected,
                label=1 if category.endswith("_pos") else 0,
                category=category,
                group_index=group_index,
            )
        )
    return sampled


class EpochBatchSampler:
    def __init__(self, groups: list[dict[str, Any]], proteins_per_batch: int, seed: int) -> None:
        if proteins_per_batch <= 0:
            raise ValueError("proteins_per_batch 必须大于 0")
        self.groups = groups
        self.proteins_per_batch = proteins_per_batch
        self.seed = seed

    def iter_batches(self, epoch: int) -> Iterator[list[SampledPair]]:
        indices = list(range(len(self.groups)))
        random.Random(self.seed + epoch).shuffle(indices)
        for start in range(0, len(indices), self.proteins_per_batch):
            batch: list[SampledPair] = []
            for group_index in indices[start : start + self.proteins_per_batch]:
                batch.extend(sample_group(self.groups[group_index], epoch, group_index))
            if batch:
                yield batch

    def epoch_pairs(self, epoch: int) -> list[SampledPair]:
        return [pair for batch in self.iter_batches(epoch) for pair in batch]


def cumulative_pair_coverage(groups: list[dict[str, Any]], through_epoch: int) -> tuple[int, int, float]:
    all_pairs = {
        (group["protein_key"], dna_key)
        for group in groups
        for category in CATEGORIES
        for dna_key in group.get(category, [])
    }
    seen: set[tuple[str, str]] = set()
    for epoch in range(1, through_epoch + 1):
        for group_index, group in enumerate(groups):
            for pair in sample_group(group, epoch, group_index):
                seen.add((pair.protein_key, pair.dna_key))
    total = len(all_pairs)
    return len(seen), total, len(seen) / total if total else 0.0
