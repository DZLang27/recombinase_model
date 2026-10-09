from __future__ import annotations

import json
from importlib.metadata import version as package_version
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from .data import NORMALIZATION_VERSION
from .utils import atomic_json_dump, atomic_torch_save, read_tsv, sha256_file, sha256_text, utc_now_iso


@dataclass
class EmbeddingStore:
    keys: list[str]
    embeddings: torch.Tensor
    manifest: dict[str, Any]

    def __post_init__(self) -> None:
        if len(self.keys) != self.embeddings.shape[0]:
            raise ValueError("embedding key 数量与 tensor 行数不一致")
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("embedding store 中存在重复 key")
        if not torch.isfinite(self.embeddings).all():
            raise ValueError("embedding store 包含 NaN/Inf")
        self.key_to_index = {key: index for index, key in enumerate(self.keys)}

    def rows(self, keys: list[str], device: torch.device | str | None = None) -> torch.Tensor:
        try:
            indices = [self.key_to_index[key] for key in keys]
        except KeyError as exc:
            raise KeyError(f"embedding store 缺少 key: {exc.args[0]}") from exc
        values = self.embeddings[indices]
        return values.to(device, non_blocking=True) if device is not None else values

    def contains_all(self, keys: list[str]) -> bool:
        return all(key in self.key_to_index for key in keys)


def load_embedding_store(path: str | Path) -> EmbeddingStore:
    directory = Path(path)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 embedding manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"embedding cache 尚未完成: {directory}")
    keys: list[str] = []
    tensors: list[torch.Tensor] = []
    for shard in manifest.get("shards", []):
        shard_path = directory / shard["file"]
        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        keys.extend(payload["keys"])
        tensors.append(payload["embeddings"])
    if not tensors:
        raise ValueError(f"embedding cache 没有 shard: {directory}")
    embeddings = torch.cat(tensors, dim=0)
    if sha256_text("\n".join(keys)) != manifest.get("ordered_keys_sha256"):
        raise ValueError(f"embedding cache key fingerprint 不匹配: {directory}")
    if list(embeddings.shape) != manifest.get("shape"):
        raise ValueError(f"embedding cache shape 与 manifest 不一致: {directory}")
    return EmbeddingStore(keys=keys, embeddings=embeddings, manifest=manifest)


def _load_sequences(path: str | Path, modality: str) -> tuple[str, list[str], list[str]]:
    expected_key = "protein_key" if modality == "protein" else "dna_key"
    rows = list(read_tsv(path))
    if not rows:
        raise ValueError(f"输入 sequence TSV 为空: {path}")
    if expected_key not in rows[0] or "sequence" not in rows[0]:
        raise ValueError(f"{path} 必须包含 {expected_key} 和 sequence")
    keys = [row[expected_key] for row in rows]
    sequences = [row["sequence"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path} 存在重复 {expected_key}")
    return expected_key, keys, sequences


def _special_token_mean(last_hidden: torch.Tensor, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
    attention = encoded["attention_mask"].bool()
    special = encoded.get("special_tokens_mask")
    valid = attention if special is None else attention & ~special.bool()
    fallback_rows = valid.sum(dim=1) == 0
    if fallback_rows.any():
        valid[fallback_rows] = attention[fallback_rows]
    weights = valid.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in mapping:
        raise ValueError(f"不支持的 embedding dtype: {name}")
    return mapping[name]


class BackboneEncoder:
    """训练预计算与在线筛选共用的唯一 backbone 编码实现。"""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        revision: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("缺少 transformers;请在 rec_rank 环境安装 requirements.txt") from exc
        load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if revision:
            load_kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        config = AutoConfig.from_pretrained(model_name, **load_kwargs)
        auto_map = getattr(config, "auto_map", {}) or {}
        # Nucleotide Transformer v2 只为 AutoModelForMaskedLM 注册了自定义
        # modeling_esm。若使用 AutoModel，Transformers 5.x 会错误落到内置
        # EsmModel，并因配置缺少新字段（例如 rope_theta）而失败。
        if (
            trust_remote_code
            and "AutoModel" not in auto_map
            and "AutoModelForMaskedLM" in auto_map
        ):
            masked_lm_model = AutoModelForMaskedLM.from_pretrained(
                model_name, config=config, **load_kwargs
            )
            self.model = masked_lm_model.base_model.to(device)
        else:
            self.model = AutoModel.from_pretrained(
                model_name, config=config, **load_kwargs
            ).to(device)
        self.model.eval()
        self.device = device
        self.resolved_revision = getattr(config, "_commit_hash", None) or revision
        self.tokenizer_resolved_revision = (
            self.tokenizer.init_kwargs.get("_commit_hash") or revision
        )

    def encode(self, sequences: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        # NT-v2 的自定义 tokenizer 在部分 Transformers 4.x 版本中返回的
        # special_tokens_mask 会比 input_ids 多一位。直接依据 special token
        # IDs 构造同形状 mask，语义相同且对两个 backbone 都稳定。
        special_mask = torch.zeros_like(encoded["input_ids"], dtype=torch.bool)
        for token_id in self.tokenizer.all_special_ids:
            special_mask |= encoded["input_ids"] == token_id
        encoded["special_tokens_mask"] = special_mask
        model_inputs = {
            key: value for key, value in encoded.items() if key != "special_tokens_mask"
        }
        amp_enabled = self.device.type == "cuda"
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=amp_enabled
        ):
            output_values = self.model(**model_inputs)
            pooled = _special_token_mean(output_values.last_hidden_state, encoded)
        return pooled.float().cpu()


def _iter_batches(keys: list[str], sequences: list[str], batch_size: int, start: int) -> Iterator[tuple[list[str], list[str]]]:
    for offset in range(start, len(keys), batch_size):
        yield keys[offset : offset + batch_size], sequences[offset : offset + batch_size]


def precompute_embeddings(
    input_path: str | Path,
    output_dir: str | Path,
    modality: str,
    model_name: str,
    batch_size: int,
    device: torch.device,
    revision: str | None = None,
    trust_remote_code: bool = False,
    output_dtype: str = "float32",
    shard_size: int = 8192,
) -> dict[str, Any]:
    if modality not in {"protein", "dna"}:
        raise ValueError("modality 必须是 protein 或 dna")
    if batch_size <= 0 or shard_size <= 0:
        raise ValueError("batch_size 和 shard_size 必须大于 0")
    _, keys, sequences = _load_sequences(input_path, modality)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ordered_keys_hash = sha256_text("\n".join(keys))
    requested = {
        "format_version": 1,
        "normalization_version": NORMALIZATION_VERSION,
        "input_sha256": sha256_file(input_path),
        "input_path": str(Path(input_path).resolve()),
        "ordered_keys_sha256": ordered_keys_hash,
        "count": len(keys),
        "modality": modality,
        "model": model_name,
        "requested_revision": revision,
        "trust_remote_code": trust_remote_code,
        "pooling": "special_token_mean",
        "dtype": output_dtype,
        "torch_version": torch.__version__,
        "transformers_version": package_version("transformers"),
    }
    manifest_path = output / "manifest.json"
    shards: list[dict[str, Any]] = []
    completed_count = 0
    existing: dict[str, Any] | None = None
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = [key for key, value in requested.items() if existing.get(key) != value]
        if mismatches:
            raise ValueError(f"拒绝复用不匹配的 embedding cache;冲突字段: {', '.join(mismatches)}")
        if existing.get("status") == "complete":
            load_embedding_store(output)
            return existing
        shards = list(existing.get("shards", []))
        completed_count = sum(int(shard["rows"]) for shard in shards)
        for shard in shards:
            if not (output / shard["file"]).exists():
                raise FileNotFoundError(f"manifest 引用的 shard 不存在: {shard['file']}")

    encoder = BackboneEncoder(
        model_name=model_name,
        device=device,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    resolved_revision = encoder.resolved_revision
    tokenizer_resolved_revision = encoder.tokenizer_resolved_revision
    if existing is not None and existing.get("resolved_revision") not in {None, resolved_revision}:
        raise ValueError("部分 embedding cache 的已解析模型 revision 与当前加载结果不一致")
    if existing is not None and existing.get("tokenizer_resolved_revision") not in {
        None,
        tokenizer_resolved_revision,
    }:
        raise ValueError("部分 embedding cache 的 tokenizer revision 与当前加载结果不一致")
    output_torch_dtype = _dtype_from_name(output_dtype)
    current_keys: list[str] = []
    current_tensors: list[torch.Tensor] = []
    shard_index = len(shards)
    started = utc_now_iso()

    in_progress = dict(requested)
    in_progress.update(
        {
            "status": "in_progress",
            "created_at": started,
            "resolved_revision": resolved_revision,
            "tokenizer_resolved_revision": tokenizer_resolved_revision,
            "shards": shards,
            "completed_count": completed_count,
        }
    )
    atomic_json_dump(manifest_path, in_progress)

    def flush_shard() -> None:
        nonlocal shard_index, current_keys, current_tensors, completed_count
        if not current_keys:
            return
        tensor = torch.cat(current_tensors, dim=0).to(output_torch_dtype).contiguous()
        if not torch.isfinite(tensor).all():
            raise FloatingPointError("backbone embedding 出现 NaN/Inf")
        file_name = f"shard_{shard_index:05d}.pt"
        atomic_torch_save(output / file_name, {"keys": current_keys, "embeddings": tensor})
        shard_info = {"file": file_name, "rows": len(current_keys), "shape": list(tensor.shape)}
        shards.append(shard_info)
        completed_count += len(current_keys)
        shard_index += 1
        current_keys = []
        current_tensors = []
        in_progress["shards"] = shards
        in_progress["completed_count"] = completed_count
        atomic_json_dump(manifest_path, in_progress)

    for batch_keys, batch_sequences in _iter_batches(keys, sequences, batch_size, completed_count):
        pooled = encoder.encode(batch_sequences)
        current_keys.extend(batch_keys)
        current_tensors.append(pooled)
        if len(current_keys) >= shard_size:
            flush_shard()
    flush_shard()

    shape = [completed_count, shards[0]["shape"][1]] if shards else [0, 0]
    complete = dict(in_progress)
    complete.update(
        {
            "status": "complete",
            "completed_at": utc_now_iso(),
            "completed_count": completed_count,
            "shape": shape,
            "shards": shards,
            "normalized": False,
            "nan_inf_check": "passed",
        }
    )
    if completed_count != len(keys):
        raise RuntimeError(f"embedding 数量不完整: {completed_count}/{len(keys)}")
    atomic_json_dump(manifest_path, complete)
    return complete


def save_embedding_store(
    output_dir: str | Path,
    keys: list[str],
    embeddings: torch.Tensor,
    manifest_fields: dict[str, Any],
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if len(keys) != embeddings.shape[0] or len(keys) != len(set(keys)):
        raise ValueError("index key/tensor 数量不一致或 key 重复")
    if not torch.isfinite(embeddings).all():
        raise FloatingPointError("index embedding 出现 NaN/Inf")
    payload = {"keys": keys, "embeddings": embeddings.detach().cpu().contiguous()}
    atomic_torch_save(output / "shard_00000.pt", payload)
    manifest = {
        "format_version": 1,
        "status": "complete",
        "created_at": utc_now_iso(),
        "count": len(keys),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype).removeprefix("torch."),
        "ordered_keys_sha256": sha256_text("\n".join(keys)),
        "normalized": True,
        "nan_inf_check": "passed",
        "shards": [{"file": "shard_00000.pt", "rows": len(keys), "shape": list(embeddings.shape)}],
        **manifest_fields,
    }
    atomic_json_dump(output / "manifest.json", manifest)
    return manifest
