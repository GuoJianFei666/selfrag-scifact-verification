"""Shared evaluation paths use retrieved documents, never cited gold as retrieval."""
from pathlib import Path

from .common import digest, file_hash, read_json, write_json, write_jsonl
from .evaluation import diagnostic_metrics, export_official, joint_metric, official_metrics, retrieval_metrics
from .pipeline import predict_view
from .views import full_view


def validate_cache(cache, dataset, claims, split, partition):
    if cache["condition"] != "retrieved" or cache["corpus_hash"] != dataset.corpus_hash or cache["split"] != split or cache["partition"] != partition:
        raise ValueError("Retrieval cache condition, corpus or split mismatch")
    if not claims or type(cache["top_k"]) is not int or cache["top_k"] < 1:
        raise ValueError("Retrieval target and top_k must be nonempty")
    metrics = retrieval_metrics(cache["hits"], claims)
    for hits in cache["hits"].values():
        if not hits or len(hits) > cache["top_k"] or any(h["doc_id"] not in dataset.corpus for h in hits):
            raise ValueError("Retrieval cache contains empty, excessive or unknown documents")
    return metrics


def run_retrieved(dataset, claims, cache, verifier):
    return [predict_view(full_view(claim, dataset.corpus[hit["doc_id"]]), verifier)
            for cid, claim in sorted(claims.items()) for hit in cache["hits"][str(cid)]]


def checkpoint_evaluation(dataset, claims, cache, verifier, adapter, output, views=None, quadruples=None):
    """One checkpoint, fixed internal model_select hits and optional reviewed views."""
    retrieval = validate_cache(cache, dataset, claims, "train", "model_select")
    if views and any(v.split != "model_select" or v.verification != "VERIFIED" or v.claim_id not in claims for v in views):
        raise ValueError("Checkpoint diagnostics require reviewed model_select views")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Checkpoint evaluation output must be empty")
    results = run_retrieved(dataset, claims, cache, verifier)
    official = export_official(results, claims, dataset.corpus)
    metrics = official_metrics(official, claims, dataset.corpus)
    manifest = read_json(Path(adapter) / "run_manifest.json")
    record = {"split": "model_select", "condition": "retrieved", "path": str(Path(adapter).resolve()),
              "baseline": manifest["run"], "step": manifest["optimizer_steps"], "learning_rate": manifest["learning_rate"],
              "abstract_rationalized": metrics["abstract_rationalized"]["f1"], "joint": None,
              "forward_calls": manifest["forward_calls"], "official": metrics, "retrieval": retrieval,
              "checkpoint_manifest_sha256": file_hash(Path(adapter) / "run_manifest.json"),
              "cache_sha256": digest(cache), "model_version": verifier.contract}
    if views:
        diagnostic = [predict_view(v, verifier) for v in views]
        record["diagnostics"] = diagnostic_metrics(views, diagnostic)
        if quadruples is not None:
            record["joint"] = joint_metric(views, diagnostic, quadruples)["joint"]
        write_jsonl(output / "diagnostic_predictions.jsonl", diagnostic)
    write_jsonl(output / "predictions.jsonl", results)
    write_jsonl(output / "official_predictions.jsonl", official)
    record["inference_forward_calls"] = sum((r.get("cost") or {}).get("forward_calls", 0) for r in results)
    record["elapsed_seconds"] = sum(r["elapsed_seconds"] for r in results)
    write_json(output / "selection_record.json", record)
    return record


def write_receipt(output, frozen_path, outputs):
    record = {"frozen_manifest_sha256": file_hash(frozen_path),
              "outputs": {str(Path(p).resolve()): file_hash(p) for p in outputs}}
    write_json(str(output) + ".receipt.json", record)


def verify_receipt(prediction, frozen_path, primary_output):
    receipt = read_json(str(primary_output) + ".receipt.json")
    if receipt["frozen_manifest_sha256"] != file_hash(frozen_path) or receipt["outputs"].get(str(Path(prediction).resolve())) != file_hash(prediction):
        raise ValueError("Prediction or its frozen-run receipt changed")
    for path, expected in receipt["outputs"].items():
        if file_hash(path) != expected:
            raise ValueError("Another output in the frozen-run receipt changed")
