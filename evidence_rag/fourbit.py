"""4-bit comparison settings for an 8GB GPU.

B0 and B1 share one NF4 base. Training saves the adapter and loads it later;
nothing here merges LoRA back into a 16-bit model.
"""
from pathlib import Path

from .data import RELATIONS
from .evaluation import diagnostic_metrics, export_official, joint_metric, official_metrics
from .model import MODEL_ID, MODEL_REVISION
from .pipeline import predict_view
from .serialization import encode_supervised
from .workflow import run_retrieved, validate_cache


def nf4_options(precision):
    """BitsAndBytes settings shared by B0 inference and B1 QLoRA."""
    if precision not in ("float16", "bfloat16"):
        raise ValueError("NF4 compute dtype must be float16 or bfloat16")
    return {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": precision}


def _require_double_quant(cfg):
    if cfg.get("quantization") != "nf4":
        raise ValueError("B0 and B1 must both use nf4")
    if cfg.get("precision") not in ("float16", "bfloat16"):
        raise ValueError("4-bit compute dtype must be float16 or bfloat16")
    if cfg.get("bnb_4bit_quant_type", "nf4") != "nf4" or cfg.get("bnb_4bit_use_double_quant", True) is not True:
        raise ValueError("8GB path requires nf4 with double quantization")
    if cfg.get("merge_adapter"):
        raise ValueError("Refusing to merge LoRA into the base model")


def apply_adapter_override(cfg, adapter, group):
    """Point B1 at a saved adapter without changing the shared 4-bit base."""
    if not adapter:
        return cfg
    if group == "B0":
        raise ValueError("B0 must use the frozen 4-bit base without an adapter")
    return {**cfg, "adapter": adapter}


def require_8gb_inference(cfg, group):
    if group not in ("B0", "B1"):
        raise ValueError("8GB inference group must be B0 or B1")
    _require_double_quant(cfg)
    if cfg.get("revision") != MODEL_REVISION or cfg.get("model_id") != MODEL_ID:
        raise ValueError("8GB path keeps the pinned SelfRAG-7B revision")
    if cfg.get("max_length") != 2048:
        raise ValueError("4-bit inference keeps the 2048 context used by the data contract")
    if group == "B0" and cfg.get("adapter") is not None:
        raise ValueError("B0 must use the frozen 4-bit base without an adapter")
    if group == "B1" and not cfg.get("adapter"):
        raise ValueError("B1 inference must load the saved adapter on the same 4-bit base")
    return cfg


def require_8gb_training(cfg):
    if cfg.get("run") != "B1":
        raise ValueError("8GB QLoRA profile is for B1")
    _require_double_quant(cfg)
    if not cfg.get("gradient_checkpointing"):
        raise ValueError("NF4 QLoRA requires gradient checkpointing")
    if cfg.get("optimizer") != "paged_adamw_8bit":
        raise ValueError("8GB QLoRA uses the paged 8-bit optimizer")
    if cfg.get("save_adapter_only") is False:
        raise ValueError("Refusing to merge LoRA into the base model")
    accumulation = cfg.get("gradient_accumulation")
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError("gradient_accumulation must be a positive integer")
    if cfg.get("max_length") != 2048:
        raise ValueError("8GB B1 profile uses max_length 2048 and does not truncate")
    if cfg.get("device") != "cuda":
        raise ValueError("8GB QLoRA training runs on cuda")
    modules = (cfg.get("lora") or {}).get("target_modules", [])
    if "q_proj" not in modules or "v_proj" not in modules:
        raise ValueError("LoRA targets must include q_proj and v_proj")
    return cfg


def build_optimizer(name, parameters, learning_rate):
    import torch
    if name in (None, "adamw"):
        return torch.optim.AdamW(parameters, lr=learning_rate)
    if name == "paged_adamw_8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError("paged_adamw_8bit requires bitsandbytes with CUDA kernels") from exc
        return bnb.optim.PagedAdamW8bit(parameters, lr=learning_rate)
    raise ValueError("Unsupported optimizer: " + str(name))


def cuda_memory_snapshot():
    import torch
    if not torch.cuda.is_available():
        return {"cuda": False, "peak_allocated_bytes": None, "peak_reserved_bytes": None, "device_name": None}
    return {"cuda": True, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "device_name": torch.cuda.get_device_name(0)}


def reset_peak_memory():
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def audit_lengths(views, tokenizer, max_length):
    """Measure supervised sequences. Overlong rows are reported and never truncated."""
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if not views:
        raise ValueError("Length audit needs at least one view")
    rows = []
    for view in views:
        tokens = len(encode_supervised(view, tokenizer, 10**9)["input_ids"])
        rows.append({"view_id": view.view_id, "tokens": tokens, "overlong": tokens > max_length})
    overlong = [{"view_id": row["view_id"], "tokens": row["tokens"]} for row in rows if row["overlong"]]
    return {"max_length": max_length, "n": len(rows), "longest": max(row["tokens"] for row in rows), "overlong": overlong}


def load_tokenizer(cache_dir=None, local_files_only=True):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True,
                                         cache_dir=cache_dir, local_files_only=local_files_only)


def citation_audit(results, corpus):
    """Count citation-valid rows. Invalid and failed rows stay in the denominator."""
    valid = 0
    for result in results:
        document = corpus[result["doc_id"]]
        ids = result.get("original_sentence_ids") or []
        relation = result.get("relation_candidate")
        ok = result.get("status") in ("accepted", "review_required") and relation in RELATIONS
        ok = ok and isinstance(ids, list) and len(ids) == len(set(ids))
        ok = ok and all(type(i) is int and 0 <= i < len(document.abstract) for i in ids)
        ok = ok and ((relation == "NOINFO") == (len(ids) == 0))
        sentences = result.get("evidence_sentences")
        if ok and sentences is not None:
            ok = sentences == [document.abstract[i] for i in ids]
        valid += int(bool(ok))
    n = len(results)
    return {"n": n, "valid": valid, "invalid": n - valid}


def subset_claims(claims, limit):
    ordered = [claims[key] for key in sorted(claims)]
    if not ordered:
        raise ValueError("Target claim set is empty")
    if limit is None:
        return ordered
    if type(limit) is not int or limit < 1:
        raise ValueError("Smoke limit must be a positive integer")
    if limit > len(ordered):
        raise ValueError("Smoke limit exceeds the available claims")
    return ordered[:limit]


def run_experiment(dataset, claims, verifier, output, *, group, config, hits=None, cited=False, limit=None,
                   views=None, quadruples=None, frozen_manifest=None):
    """Score one group with the existing official, diagnostic and citation checks."""
    require_8gb_inference(config, group)
    if cited == (hits is not None):
        raise ValueError("Pass retrieved hits or cited originals, not both and not neither")
    chosen = subset_claims(claims, limit)
    chosen_map = {claim.claim_id: claim for claim in chosen}
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Experiment output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    frozen_sha = None
    if frozen_manifest:
        from .common import file_hash
        from .protocol import verify_frozen
        verify_frozen(frozen_manifest)
        frozen_sha = file_hash(frozen_manifest)
    if cited:
        from .views import full_view
        built = [full_view(claim, dataset.corpus[did], labeled=True)
                 for claim in chosen for did in claim.cited_doc_ids]
        predictions = [predict_view(view, verifier) for view in built]
        diagnostics = diagnostic_metrics(built, predictions)
        condition = "cited"
    else:
        filtered = {**hits, "hits": {str(claim.claim_id): hits["hits"][str(claim.claim_id)] for claim in chosen}}
        validate_cache(filtered, dataset, chosen_map, filtered["split"], filtered["partition"])
        predictions = run_retrieved(dataset, chosen_map, filtered, verifier)
        built = views
        diagnostics = diagnostic_metrics(views, predictions) if views is not None else None
        condition = "retrieved"
    official_rows = export_official(predictions, chosen_map, dataset.corpus)
    official = official_metrics(official_rows, chosen_map, dataset.corpus)
    joint = None
    if quadruples is not None:
        if built is None:
            raise ValueError("Joint metric requires reviewed views")
        joint = joint_metric(built, predictions, quadruples)["joint"]
    payload = {"group": group, "condition": condition, "smoke": limit is not None, "smoke_limit": limit,
               "claim_ids": [claim.claim_id for claim in chosen], "quantization": config["quantization"],
               "precision": config["precision"], "adapter": config.get("adapter"),
               "model_revision": config.get("revision"), "peak_vram": cuda_memory_snapshot(),
               "per_example": [{"view_id": row["view_id"], "claim_id": row["claim_id"], "doc_id": row["doc_id"],
                                "status": row["status"], "elapsed_seconds": row["elapsed_seconds"]}
                               for row in predictions],
               "official": official, "diagnostics": diagnostics, "joint": joint,
               "citation": citation_audit(predictions, dataset.corpus), "config": config,
               "frozen_manifest_sha256": frozen_sha,
               "note": "Metrics come from this run's predictions. A missing joint means no reviewed quadruples were passed."}
    from .common import write_json, write_jsonl
    write_jsonl(output / "predictions.jsonl", predictions)
    write_jsonl(output / "official.jsonl", official_rows)
    write_json(output / "results.json", payload)
    return payload
