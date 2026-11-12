"""Train-split pair construction. These tests do not load a retrieval model."""
import pytest

from evidence_rag.data import Claim, Document, Evidence
from evidence_rag.retrieval_engineering.config import load_config
from evidence_rag.retrieval_engineering.finetune_data import (
    build_examples, pack_batches, split_train_claims,
)
from evidence_rag.retrieval_engineering.report import merge_dev_report, render_markdown

PREFIX = "Represent this sentence for searching relevant passages: "


def _doc(doc_id, *sentences):
    return Document(doc_id, f"Title {doc_id}", tuple(sentences))


def _claim(claim_id, text, evidence, split="train"):
    return Claim(claim_id, text, tuple(evidence), evidence, split, split != "test")


def test_positive_window_prefers_the_span_covering_more_rationale_sentences():
    document = _doc(1, "alpha", "kinase binds", "inhibitor blocks", "delta")
    evidence = Evidence("SUPPORT", ((1, 2),))
    from evidence_rag.retrieval_engineering.finetune_data import best_positive_window
    chosen = best_positive_window(document, evidence, 2, 1)
    assert chosen["sentence_start"] == 1
    assert chosen["sentence_end"] == 3
    assert chosen["text"] == "Title 1\nkinase binds inhibitor blocks"


def test_examples_use_one_non_gold_hard_negative_and_prefix_only_the_claim():
    corpus = {
        1: _doc(1, "alpha", "kinase binds", "inhibitor blocks"),
        2: _doc(2, "weather", "forecast"),
        3: _doc(3, "kinase assay", "control"),
    }
    claim = _claim(7, "kinase inhibitor", {1: Evidence("SUPPORT", ((1,),))})
    rows = build_examples([claim], corpus, {7: [1, 3, 2]}, 2, 1, PREFIX)
    assert len(rows) == 1
    assert rows[0]["negative_doc_id"] == 3
    assert rows[0]["positive_doc_id"] == 1
    assert rows[0]["anchor"] == PREFIX + "kinase inhibitor"
    assert not rows[0]["positive"].startswith("Represent this sentence")
    assert "kinase assay" in rows[0]["negative"]


def test_dev_claims_are_rejected():
    claim = _claim(7, "kinase inhibitor", {1: Evidence("SUPPORT", ((0,),))}, split="dev")
    with pytest.raises(ValueError, match="train"):
        build_examples([claim], {1: _doc(1, "alpha")}, {7: [1]}, 2, 1, PREFIX)
    with pytest.raises(ValueError, match="train"):
        split_train_claims([claim], 0.1, 42)


def test_monitor_split_is_deterministic_and_leaves_every_id_in_one_side():
    claims = [_claim(i, f"claim {i}", {i: Evidence("SUPPORT", ((0,),))}) for i in range(10)]
    first_fit, first_monitor = split_train_claims(claims, 0.2, 42)
    second_fit, second_monitor = split_train_claims(claims, 0.2, 42)
    assert [claim.claim_id for claim in first_fit] == [claim.claim_id for claim in second_fit]
    assert [claim.claim_id for claim in first_monitor] == [claim.claim_id for claim in second_monitor]
    fit_ids = {claim.claim_id for claim in first_fit}
    monitor_ids = {claim.claim_id for claim in first_monitor}
    assert fit_ids.isdisjoint(monitor_ids)
    assert fit_ids | monitor_ids == set(range(10))
    assert len(monitor_ids) == 2


def test_batches_keep_a_claim_unique_and_cover_every_row():
    examples = [
        {"claim_id": 1, "anchor": "a", "positive": "p1", "negative": "n"},
        {"claim_id": 1, "anchor": "a", "positive": "p2", "negative": "n"},
        {"claim_id": 2, "anchor": "b", "positive": "p3", "negative": "n"},
        {"claim_id": 3, "anchor": "c", "positive": "p4", "negative": "n"},
    ]
    batches = pack_batches(examples, batch_size=2, seed=7, epoch=0)
    seen = []
    for batch in batches:
        assert len({row["claim_id"] for row in batch}) == len(batch)
        seen.extend((row["claim_id"], row["positive"]) for row in batch)
    assert sorted(seen) == [(1, "p1"), (1, "p2"), (2, "p3"), (3, "p4")]


def test_merge_keeps_existing_metrics_and_only_accepts_finetuned_rows():
    existing = {
        "split": "dev",
        "generated_at": "2026-11-12T08:09:14+00:00",
        "experiments": [
            {"id": "bge_milvus_whole", "label": "base", "status": "measured",
             "recall": {"1": 0.5, "3": 0.5, "5": 0.8997340425531916, "10": 0.5, "20": 0.5},
             "mrr_at_10": 0.25, "ndcg_at_10": 0.125,
             "latency_seconds": {"mean": 0.2}, "index_build_seconds": 1.0},
            {"id": "bge_query_rewrite", "label": "rewrite", "status": "not_run", "reason": "not yet run"},
        ],
        "models": {},
    }
    row = {
        "id": "bge_finetuned_whole", "label": "finetuned", "status": "measured",
        "recall": {"1": 0.1, "3": 0.2, "5": 0.3, "10": 0.4, "20": 0.5},
        "mrr_at_10": 0.2, "ndcg_at_10": 0.3,
        "latency_seconds": {"mean": 0.2}, "index_build_seconds": 3.0,
    }
    record = {"base_model_id": "BAAI/bge-small-en-v1.5", "base_revision": "abc", "seed": 42, "status": "completed"}
    merged = merge_dev_report(existing, [row], record, "2026-11-12T12:00:00+00:00")
    assert merged["experiments"][0]["recall"]["5"] == 0.8997340425531916
    assert merged["experiments"][1]["status"] == "not_run"
    assert merged["experiments"][2]["id"] == "bge_finetuned_whole"
    assert merged["generated_at"] == "2026-11-12T08:09:14+00:00"
    assert merged["finetune"]["seed"] == 42
    text = render_markdown(merged)
    assert "0.8997" in text
    assert "0.3000" in text
    assert "随机种子：42" in text
    with pytest.raises(ValueError, match="Refusing"):
        merge_dev_report(existing, [{"id": "bge_milvus_whole", "status": "measured"}], record, "t")


def test_finetune_config_and_commands_reject_the_wrong_split():
    spec = load_config("configs/retrieval_engineering.yaml")["finetune"]
    assert spec["epochs"] == 2
    assert spec["seed"] == 42
    assert spec["reranker"] == "skip"
    assert spec["loss"] == "MultipleNegativesRankingLoss"
    from evidence_rag.retrieval_engineering.__main__ import main
    with pytest.raises(ValueError, match="train split"):
        main(["finetune", "--data", "unused", "--split", "dev"])
    with pytest.raises(ValueError, match="dev split"):
        main(["eval-finetuned", "--data", "unused", "--split", "train", "--model", "m", "--record", "r"])
