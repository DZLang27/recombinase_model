from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .embeddings import EmbeddingStore
from .model import RecRankModel


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device) -> RecRankModel:
    model = RecRankModel(**checkpoint["model_architecture"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device)


def project_store(
    model: RecRankModel,
    store: EmbeddingStore,
    keys: list[str],
    modality: str,
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    if modality not in {"protein", "dna"}:
        raise ValueError("modality 必须是 protein 或 dna")
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(keys), batch_size):
            batch_keys = keys[start : start + batch_size]
            values = store.rows(batch_keys, device=device).float()
            projected = (
                model.project_proteins(values)
                if modality == "protein"
                else model.project_dnas(values)
            )
            outputs.append(projected.float().cpu())
    if not outputs:
        raise ValueError("不能投影空的 key 集合")
    return torch.cat(outputs, dim=0)
