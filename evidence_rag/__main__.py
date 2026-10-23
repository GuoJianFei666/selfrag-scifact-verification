"""The CLI implements the blueprint phases with one explicit command surface."""
import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import random

from .common import digest, file_hash, read_json, read_jsonl, write_json, write_jsonl
from .data import Dataset, audit_sources, download_archive, import_archive, validate_split
from .evaluation import diagnostic_metrics, export_official, joint_metric, official_metrics, retrieval_metrics
from .model import MODEL_ID, MODEL_REVISION, SelfRagVerifier, cold_reload_check, load_release
from .pipeline import Engine, predict_view
from .protocol import calibrate, freeze_files, require_bindings, require_final, select_checkpoint, verify_frozen
from .retrieval import ContrieverEncoder, ExactIndex
from .training import prepare_training, train_lora, validate_same_exposure
from .views import apply_reviews, full_view, load_views, prepare_evaluation, propose_views
from .workflow import checkpoint_evaluation, run_retrieved, validate_cache, verify_receipt, write_receipt


def config(path):
    import yaml
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Configuration must be an object")
    return value


def select_claims(data, split, partition=None, splits=None):
    claims = data.claims[split]
    if partition:
        if partition not in ("fit", "model_select", "calibration"):
            raise ValueError("Unknown internal partition")
        if split != "train" or not splits:
            raise ValueError("Internal partitions require official train and a split manifest")
        manifest = read_json(splits)
        validate_split(manifest)
        if file_hash(data.directory / "manifest.json") != manifest["dataset_manifest_sha256"]:
            raise ValueError("Split manifest refers to another dataset")
        claims = {cid: c for cid, c in claims.items() if manifest["assignments"][str(cid)] == partition}
    if not claims:
        raise ValueError("Target claim set is empty")
    return claims


def open_index(path, data, device="cpu", cache=None):
    spec = read_json(Path(path) / "manifest.json")["encoder"]
    encoder = ContrieverEncoder(spec["model_id"], spec["revision"], device=device,
                                max_length=spec["max_length"], normalize=spec["normalize"], cache_dir=cache)
    return ExactIndex(path, data.corpus, encoder)


def verifier_from_config(path):
    return verifier_from_settings(config(path))


def verifier_from_settings(cfg):
    if cfg.get("model_id", MODEL_ID) != MODEL_ID or cfg.get("revision", MODEL_REVISION) != MODEL_REVISION:
        raise ValueError("Only the frozen SelfRAG release is accepted")
    model, tokenizer, contract = load_release(cfg.get("device", "cpu"), cfg.get("precision", "float32"),
                                              cfg.get("adapter"), cfg.get("cache_dir"), cfg.get("local_files_only", True),
                                              cfg.get("quantization", "none"))
    return SelfRagVerifier(model, tokenizer, contract, cfg.get("max_length", 2048), cfg.get("max_new_tokens", 96),
                           cfg.get("temperature", 1.0), cfg.get("threshold", 0.0))


def main():
    parser = argparse.ArgumentParser(prog="evidence-rag")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("import-data"); p.add_argument("--archive", required=True); p.add_argument("--output", required=True); p.add_argument("--download", action="store_true")
    p = sub.add_parser("audit-sources"); p.add_argument("--data", required=True); p.add_argument("--sources", required=True); p.add_argument("--audit", required=True); p.add_argument("--splits", required=True); p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("build-views"); p.add_argument("--data", required=True); p.add_argument("--splits", required=True); p.add_argument("--partition", choices=("fit", "model_select", "calibration", "dev"), required=True); p.add_argument("--output", required=True); p.add_argument("--frozen-method"); p.add_argument("--original-only", action="store_true")
    p = sub.add_parser("review-queue"); p.add_argument("--views", required=True); p.add_argument("--output", required=True); p.add_argument("--round", type=int, choices=(1, 2), required=True); p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("apply-reviews"); p.add_argument("--views", required=True); p.add_argument("--reviews", nargs="+", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("prepare-evaluation"); p.add_argument("--views", required=True); p.add_argument("--output", required=True); p.add_argument("--frozen-method")
    p = sub.add_parser("build-index"); p.add_argument("--data", required=True); p.add_argument("--output", required=True); p.add_argument("--revision", required=True); p.add_argument("--device", default="cpu"); p.add_argument("--cache"); p.add_argument("--batch-size", type=int, default=16); p.add_argument("--max-length", type=int, default=512)
    p = sub.add_parser("cache-hits"); p.add_argument("--data", required=True); p.add_argument("--index", required=True); p.add_argument("--split", choices=("train", "dev", "test"), required=True); p.add_argument("--partition"); p.add_argument("--splits"); p.add_argument("--output", required=True); p.add_argument("--top-k", type=int, default=5); p.add_argument("--device", default="cpu"); p.add_argument("--cache")
    p = sub.add_parser("model-contract"); p.add_argument("--config", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("fetch-model"); p.add_argument("--cache", required=True); p.add_argument("--include-weights", action="store_true")
    p = sub.add_parser("cold-reload"); p.add_argument("--config", required=True); p.add_argument("--probe", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("train"); p.add_argument("--config", required=True); p.add_argument("--views", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("prepare-training"); p.add_argument("--views", required=True); p.add_argument("--run", choices=("B1", "B2", "B3"), required=True); p.add_argument("--output", required=True); p.add_argument("--audit", required=True)
    p = sub.add_parser("compare-exposure"); p.add_argument("--b2", required=True); p.add_argument("--b3", required=True)
    p = sub.add_parser("predict-views"); p.add_argument("--views", required=True); p.add_argument("--config", required=True); p.add_argument("--output", required=True); p.add_argument("--frozen-manifest"); p.add_argument("--data"); p.add_argument("--splits")
    p = sub.add_parser("score-checkpoint"); p.add_argument("--data", required=True); p.add_argument("--splits", required=True); p.add_argument("--hits", required=True); p.add_argument("--config", required=True); p.add_argument("--adapter", required=True); p.add_argument("--views"); p.add_argument("--quadruples"); p.add_argument("--output", required=True)
    p = sub.add_parser("calibration-rows"); p.add_argument("--views", required=True); p.add_argument("--predictions", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("run"); p.add_argument("--data", required=True); p.add_argument("--hits", required=True); p.add_argument("--split", choices=("train", "dev", "test"), required=True); p.add_argument("--partition"); p.add_argument("--splits"); p.add_argument("--config", required=True); p.add_argument("--output", required=True); p.add_argument("--frozen-manifest")
    p = sub.add_parser("evaluate"); p.add_argument("--data", required=True); p.add_argument("--predictions", required=True); p.add_argument("--split", choices=("train", "dev"), required=True); p.add_argument("--partition"); p.add_argument("--splits"); p.add_argument("--output", required=True); p.add_argument("--frozen-manifest")
    p = sub.add_parser("diagnostics"); p.add_argument("--views", required=True); p.add_argument("--predictions", required=True); p.add_argument("--quadruples"); p.add_argument("--output", required=True); p.add_argument("--frozen-manifest"); p.add_argument("--data"); p.add_argument("--splits")
    p = sub.add_parser("calibrate"); p.add_argument("--rows", required=True); p.add_argument("--output", required=True); p.add_argument("--threshold", type=float, default=0.0)
    p = sub.add_parser("select-checkpoint"); p.add_argument("--records", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("select-optimizer"); p.add_argument("--records", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("freeze"); p.add_argument("--phase", choices=("method", "views", "final_run"), required=True); p.add_argument("--spec", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("verify-frozen"); p.add_argument("path")
    p = sub.add_parser("serve"); p.add_argument("--data", required=True); p.add_argument("--index"); p.add_argument("--config"); p.add_argument("--reports", default="var/reports"); p.add_argument("--port", type=int, default=8787); p.add_argument("--cache")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    action = args.command
    if action == "import-data":
        if args.download:
            download_archive(args.archive)
        value = import_archive(args.archive, args.output, {"corpus": 5183, "train": 809, "dev": 300, "test": 300})
    elif action == "audit-sources":
        value = audit_sources(Dataset(args.data), args.sources, args.audit, args.splits, args.seed)
    elif action == "build-views":
        data, splits = Dataset(args.data), read_json(args.splits)
        if args.partition == "dev":
            if not args.frozen_method:
                raise ValueError("Dev view construction requires a frozen method protocol")
            require_bindings(args.frozen_method, "method", {"dataset_manifest": Path(args.data) / "manifest.json", "split_manifest": args.splits})
            claims = data.claims["dev"]
        else:
            claims = select_claims(data, "train", args.partition, args.splits)
        views = []
        for claim in sorted(claims.values(), key=lambda c: c.claim_id):
            for did in claim.cited_doc_ids:
                group = splits["groups"].get(str(claim.claim_id))
                if group is None and args.partition != "dev":
                    raise ValueError("Internal training view lacks a source group")
                full = full_view(claim, data.corpus[did], args.partition, group or "UNRESOLVED_SOURCE", True)
                views.extend([full] + ([] if args.original_only else propose_views(full)))
        write_jsonl(args.output, [v.asdict() for v in views])
        value = {"views": len(views), "verified": sum(v.verification == "VERIFIED" for v in views)}
    elif action == "review-queue":
        rows = [{"view_id": v.view_id, "content_hash": digest(v.asdict()), "claim": v.claim, "title": v.title,
                 "sentences": list(v.sentences), "round": args.round, "decision": None,
                 "relation": None, "reviewer": "", "note": ""} for v in load_views(args.views) if v.verification == "UNVERIFIED"]
        random.Random(args.seed + args.round * 104729).shuffle(rows)
        write_jsonl(args.output, rows)
        value = {"review_items": len(rows), "round": args.round}
    elif action == "apply-reviews":
        views = apply_reviews(load_views(args.views), [r for path in args.reviews for r in read_jsonl(path)])
        write_jsonl(args.output, [v.asdict() for v in views])
        value = {"verified": sum(v.verification == "VERIFIED" for v in views), "total": len(views)}
    elif action == "prepare-evaluation":
        views = load_views(args.views)
        if any(v.split == "dev" for v in views):
            if not args.frozen_method:
                raise ValueError("Dev evaluation views require a frozen method")
            require_bindings(args.frozen_method, "method", {})
        value = prepare_evaluation(args.views, args.output)
        if args.frozen_method:
            root = Path(args.output)
            freeze_files(root / "view_protocol.json", {"method": args.frozen_method, "views": root / "views.jsonl",
                         "quadruples": root / "quadruples.json", "exclusion_audit": root / "exclusion_audit.json"},
                         {"partition": value["partition"], "view_rule_version": "deletion-v1"}, "views")
    elif action == "build-index":
        data = Dataset(args.data)
        encoder = ContrieverEncoder("facebook/contriever", args.revision, args.device, args.max_length, cache_dir=args.cache)
        manifest = ExactIndex.build(args.output, list(data.corpus.values()), encoder, args.batch_size)
        value = {"documents": len(manifest["doc_ids"]), "dimension": manifest["dimension"], "manifest": str(Path(args.output) / "manifest.json")}
    elif action == "cache-hits":
        data = Dataset(args.data); index = open_index(args.index, data, args.device, args.cache)
        claims = select_claims(data, args.split, args.partition, args.splits)
        ordered = sorted(claims.items())
        hits = {str(cid): rows for (cid, _), rows in zip(ordered, index.search_many([c.text for _, c in ordered], args.top_k))}
        cache_record = {"condition": "retrieved", "split": args.split, "partition": args.partition,
                 "corpus_hash": data.corpus_hash, "retrieval": index.manifest, "top_k": args.top_k, "hits": hits}
        write_json(args.output, cache_record)
        value = {"claims": len(claims), "retrieval": retrieval_metrics(hits, claims), "output": args.output}
    elif action == "model-contract":
        value = verifier_from_config(args.config).contract; write_json(args.output, value)
    elif action == "fetch-model":
        from huggingface_hub import snapshot_download
        patterns = ["*.json", "*.model"]
        if args.include_weights:
            patterns += ["*.safetensors", "pytorch_model*.bin"]
        path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=args.cache, allow_patterns=patterns)
        value = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weights_requested": args.include_weights, "snapshot": path}
    elif action == "cold-reload":
        value = cold_reload_check(verifier_from_config(args.config), args.probe, args.output)
    elif action == "train":
        value = train_lora(config(args.config), args.views, args.output)
    elif action == "prepare-training":
        value = prepare_training(args.views, args.run, args.output, args.audit)
    elif action == "compare-exposure":
        validate_same_exposure(read_json(args.b2), read_json(args.b3)); value = {"same_sft_exposure": True}
    elif action == "predict-views":
        views = load_views(args.views)
        frozen = None
        if any(v.split == "dev" for v in views):
            frozen = require_final(args.frozen_manifest, args.data, args.splits, config=args.config,
                                   prediction=args.output, diagnostic_views=args.views)
            if Path(args.output).resolve() != Path(frozen["parameters"]["diagnostic_prediction_path"]).resolve():
                raise ValueError("Diagnostic output path mismatch")
        model = verifier_from_config(args.config)
        results = [predict_view(v, model) for v in views]
        write_jsonl(args.output, results); value = {"predictions": len(results)}
        if frozen:
            write_receipt(args.output, args.frozen_manifest, [args.output])
    elif action == "score-checkpoint":
        data = Dataset(args.data); claims = select_claims(data, "train", "model_select", args.splits)
        model = verifier_from_settings({**config(args.config), "adapter": args.adapter})
        value = checkpoint_evaluation(data, claims, read_json(args.hits), model, args.adapter, args.output,
                                      load_views(args.views) if args.views else None,
                                      read_json(args.quadruples) if args.quadruples else None)
    elif action == "calibration-rows":
        from .data import RELATIONS
        views = load_views(args.views); raw_predictions = read_jsonl(args.predictions)
        predictions = {p["view_id"]: p for p in raw_predictions}
        if len(predictions) != len(raw_predictions) or set(predictions) != {v.view_id for v in views} or any(v.split != "calibration" or v.kind != "Full" or v.verification != "VERIFIED" for v in views):
            raise ValueError("Calibration requires complete original cited calibration views")
        if any(predictions[v.view_id]["claim_id"] != v.claim_id or predictions[v.view_id]["doc_id"] != v.doc_id for v in views):
            raise ValueError("Calibration prediction claim/document mismatch")
        rows = [{"split": "calibration", "condition": "cited", "view_kind": "Full", "view_id": v.view_id,
                 "label_scores": predictions[v.view_id]["label_scores"], "label_index": RELATIONS.index(v.relation),
                 "model_version": predictions[v.view_id]["model_version"], "views_file_sha256": file_hash(args.views),
                 "predictions_file_sha256": file_hash(args.predictions)} for v in views]
        write_jsonl(args.output, rows); value = {"rows": len(rows)}
    elif action == "run":
        data = Dataset(args.data); claims = select_claims(data, args.split, args.partition, args.splits)
        frozen = require_final(args.frozen_manifest, args.data, args.splits, args.config, args.hits, args.output) if args.split == "dev" else None
        if frozen and Path(args.output).resolve() != Path(frozen["parameters"]["prediction_path"]).resolve():
            raise ValueError("Retrieved output path mismatch")
        cache = read_json(args.hits)
        validate_cache(cache, data, claims, args.split, args.partition)
        model = verifier_from_config(args.config)
        results = run_retrieved(data, claims, cache, model)
        write_jsonl(args.output, results)
        official_path = str(args.output) + ".official.jsonl"
        if frozen and Path(official_path).resolve() != Path(frozen["parameters"]["official_prediction_path"]).resolve():
            raise ValueError("Official output path mismatch")
        write_jsonl(official_path, export_official(results, claims, data.corpus))
        if frozen:
            write_receipt(args.output, args.frozen_manifest, [args.output, official_path])
        value = {"claims": len(claims), "pairs": len(results)}
    elif action == "evaluate":
        if args.split == "dev":
            frozen = require_final(args.frozen_manifest, args.data, args.splits, prediction=args.predictions)
            verify_receipt(args.predictions, args.frozen_manifest, frozen["parameters"]["prediction_path"])
        data = Dataset(args.data); claims = select_claims(data, args.split, args.partition, args.splits)
        value = official_metrics(read_jsonl(args.predictions), claims, data.corpus); write_json(args.output, value)
    elif action == "diagnostics":
        views, predictions = load_views(args.views), read_jsonl(args.predictions)
        if any(v.split == "dev" for v in views):
            frozen = require_final(args.frozen_manifest, args.data, args.splits, prediction=args.predictions, diagnostic_views=args.views)
            verify_receipt(args.predictions, args.frozen_manifest, frozen["parameters"]["diagnostic_prediction_path"])
            view_protocol = verify_frozen(frozen["files"]["view_protocol"]["path"])
            if args.quadruples:
                require_bindings(frozen["files"]["view_protocol"]["path"], "views", {"quadruples": args.quadruples})
        value = diagnostic_metrics(views, predictions)
        if args.quadruples:
            value["quadruples"] = joint_metric(views, predictions, read_json(args.quadruples))
        write_json(args.output, value)
    elif action == "calibrate":
        value = calibrate(read_jsonl(args.rows), args.output, args.threshold)
    elif action == "select-checkpoint":
        value = select_checkpoint(read_json(args.records), args.output)
    elif action == "select-optimizer":
        records = read_json(args.records)
        if not records or any(r["baseline"] != "B1" or r["split"] != "model_select" for r in records):
            raise ValueError("Common optimizer selection requires B1 model_select results")
        best = min(records, key=lambda r: (-r["abstract_rationalized"], -(r["joint"] if r["joint"] is not None else -1), r["forward_calls"], r["learning_rate"]))
        value = {"selected_on": "model_select", "baseline": "B1", "learning_rate": best["learning_rate"], "candidate_records_hash": digest(records)}
        write_json(args.output, value)
    elif action == "freeze":
        spec = read_json(args.spec); value = freeze_files(args.output, spec["files"], spec["parameters"], args.phase)
    elif action == "verify-frozen":
        value = verify_frozen(args.path)
    elif action == "serve":
        import uvicorn
        from .app import create_app
        data = Dataset(args.data)
        index = open_index(args.index, data, cache=args.cache) if args.index else None
        model = verifier_from_config(args.config) if args.config else None
        uvicorn.run(create_app(Engine(data, index, model, args.reports)), host="127.0.0.1", port=args.port)
        return
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
