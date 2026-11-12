"""Retrieval-engineering config: fixed model revisions and Milvus endpoint mode."""
from pathlib import Path
import re


def require_revision(revision):
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Model revision must be a full 40-character commit SHA")
    return revision


def milvus_mode(uri):
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError("Milvus uri is required")
    lowered = uri.strip().lower()
    if lowered.startswith(("http://", "https://", "tcp://")):
        return "server"
    if lowered.endswith(".db"):
        return "lite"
    raise ValueError("Milvus uri must be an http(s)/tcp server address or a Milvus Lite .db file")


def load_config(path):
    import yaml
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Retrieval engineering config must be an object")
    require_revision(value["baseline"]["revision"])
    require_revision(value["embedding"]["revision"])
    require_revision(value["reranker"]["revision"])
    milvus_mode(value["milvus"]["uri"])
    prefix = value["embedding"].get("query_prefix", "")
    if not isinstance(prefix, str) or prefix and not prefix.endswith(" "):
        raise ValueError("BGE query prefix must be empty or end with a space")
    _validate_finetune(value.get("finetune"))
    return value


def _validate_finetune(spec):
    if spec is None:
        return
    if not isinstance(spec, dict):
        raise ValueError("finetune config must be an object")
    epochs = spec.get("epochs")
    if type(epochs) is not int or not 1 <= epochs <= 3:
        raise ValueError("finetune.epochs must be an integer from 1 to 3")
    if type(spec.get("batch_size")) is not int or spec["batch_size"] < 2:
        raise ValueError("finetune.batch_size must be an integer of at least 2")
    if type(spec.get("seed")) is not int:
        raise ValueError("finetune.seed must be an integer")
    if spec.get("hard_negatives") != 1:
        raise ValueError("finetune.hard_negatives must be 1")
    if spec.get("loss") != "MultipleNegativesRankingLoss":
        raise ValueError("finetune.loss must be MultipleNegativesRankingLoss")
    if spec.get("reranker") != "skip":
        raise ValueError("finetune.reranker must be skip")
    fraction = spec.get("monitor_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < float(fraction) < 0.5:
        raise ValueError("finetune.monitor_fraction must be between 0 and 0.5")
    for key in ("learning_rate", "warmup_ratio", "weight_decay", "loss_scale"):
        value = spec.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0:
            raise ValueError(f"finetune.{key} must be a non-negative number")
    if type(spec.get("max_length")) is not int or spec["max_length"] < 8:
        raise ValueError("finetune.max_length must be an integer of at least 8")
