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
    return value
