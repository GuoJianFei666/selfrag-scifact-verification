"""Retrieval experiment commands. One entry point per comparison, plus `compare` for the non-LLM suite."""
import os

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import argparse
import logging

from ..common import read_json, utcnow, write_json
from .config import load_config
from .finetune import run_finetune
from .llm import gated_transform, llm_settings
from .plan import EXPERIMENTS
from .report import merge_dev_report
from .run import Comparison, publish

CHUNKINGS = {
    "whole": "whole_abstract",
    "whole_abstract": "whole_abstract",
    "sentence": "sentence",
    "sliding": "sliding",
    "semantic": "semantic",
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m evidence_rag.retrieval_engineering")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", required=True)
    common.add_argument("--config", default="configs/retrieval_engineering.yaml")
    common.add_argument("--cache", default="var/model-cache")
    common.add_argument("--work", default="var/retrieval-runs")
    common.add_argument("--split", default="dev", choices=("dev", "train"))
    common.add_argument("--resume", action="store_true")
    common.add_argument("--output")
    common.add_argument("--markdown")

    sub.add_parser("baseline", parents=[common], help="(a) Contriever / FAISS exact")
    bge = sub.add_parser("bge", parents=[common], help="(b) BGE in Milvus via LlamaIndex")
    bge.add_argument("--chunking", default="whole", choices=tuple(CHUNKINGS))
    sub.add_parser("rerank", parents=[common], help="(c) BGE retrieval plus bge-reranker")
    sub.add_parser("rewrite", parents=[common], help="(d) query rewrite and HyDE; skips without an API key")
    chunking = sub.add_parser("chunking", parents=[common], help="(e) whole, sentence, sliding, and semantic chunks")
    chunking.add_argument("--variants", default="whole,sentence,sliding,semantic")
    sub.add_parser("compare", parents=[common], help="Run (a), (b), (c) and (e). Record (d) as not yet run.")
    finetune = sub.add_parser("finetune", parents=[common], help="Fine-tune bge-small on SciFact train")
    finetune.add_argument("--model-dir", default="var/retrieval-runs/finetuned-bge")
    evaluated = sub.add_parser("eval-finetuned", parents=[common], help="Score the fine-tuned encoder on SciFact dev")
    evaluated.add_argument("--model", required=True)
    evaluated.add_argument("--record", required=True)
    evaluated.add_argument("--results", default="results/retrieval-dev.json")
    # compare can run without --data being forgotten; it is required by the parent.
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "finetune":
        if args.split != "train":
            raise ValueError("finetune only reads the train split; pass --split train")
        record = run_finetune(args.data, load_config(args.config), args.cache, args.model_dir)
        output = args.output or "var/retrieval-runs/finetune-record.json"
        write_json(output, record)
        logging.info("wrote %s", output)
        return record
    if args.command == "eval-finetuned":
        if args.split != "dev":
            raise ValueError("eval-finetuned scores the dev split")
        return _eval_finetuned(args)
    if args.command == "rewrite":
        report = _rewrite(args)
    else:
        comparison = Comparison(args.data, args.config, args.cache, args.work, args.split, args.resume)
        report = _run(comparison, args)
    output = args.output or _default_output(args.command)
    markdown = args.markdown
    if args.command == "compare" and markdown is None:
        markdown = "results/retrieval-dev.md"
    if args.command == "compare" and args.output is None:
        output = "results/retrieval-dev.json"
    publish(report, output, markdown)
    logging.info("wrote %s", output)
    return report


def _eval_finetuned(args):
    comparison = Comparison(args.data, args.config, args.cache, args.work, "dev", True)
    rows = comparison.eval_finetuned(args.model)
    existing_path = args.output or args.results
    merged = merge_dev_report(read_json(existing_path), rows, read_json(args.record), utcnow())
    markdown = args.markdown
    if markdown is None and existing_path == "results/retrieval-dev.json":
        markdown = "results/retrieval-dev.md"
    publish(merged, existing_path, markdown)
    logging.info("wrote %s", existing_path)
    return merged


def _run(comparison, args):
    if args.command == "baseline":
        rows = [comparison.baseline()]
    elif args.command == "bge":
        rows = [comparison.bge(CHUNKINGS[args.chunking])]
    elif args.command == "rerank":
        rows = [comparison.rerank()]
    elif args.command == "chunking":
        rows = comparison.chunking(_variants(args.variants))
    elif args.command == "compare":
        return comparison.compare()
    else:
        raise ValueError(args.command)
    return comparison.report(rows)


def _rewrite(args):
    settings = llm_settings(load_config(args.config)["llm"], os.environ)
    if not settings["enabled"]:
        rows = []
        for experiment_id in ("bge_query_rewrite", "bge_hyde"):
            label = next(spec["label"] for spec in EXPERIMENTS if spec["id"] == experiment_id)
            rows.append(gated_transform(settings, experiment_id, label, lambda: None))
        return {
            "task": "scifact-document-retrieval",
            "generated_at": utcnow(),
            "experiments": rows,
            "metric_notes": ["Query rewrite and HyDE did not run because the OpenAI-compatible endpoint is not configured."],
        }
    comparison = Comparison(args.data, args.config, args.cache, args.work, args.split, args.resume)
    return comparison.report(comparison.rewrite(os.environ))


def _variants(value):
    kinds = []
    for item in value.split(","):
        key = item.strip()
        if key not in CHUNKINGS:
            raise ValueError(f"Unknown chunking variant: {key}")
        kinds.append(CHUNKINGS[key])
    if not kinds:
        raise ValueError("At least one chunking variant is required")
    return kinds


def _default_output(command):
    return f"var/retrieval-runs/{command}.json"


if __name__ == "__main__":
    main()
