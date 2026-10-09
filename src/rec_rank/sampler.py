from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterator

from .data import CATEGORIES


@dataclass(frozen=True)
class SampledPair:
    protein_key: str
    dna_key: str
    label: int
    category: str
    group_index: int


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

