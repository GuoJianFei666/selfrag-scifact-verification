"""CPU tests for retrieval-engineering plumbing. No model downloads."""
import json
import math
from pathlib import Path

import pytest

from evidence_rag.retrieval_engineering.aggregate import aggregate_documents
from evidence_rag.retrieval_engineering.chunking import (
    semantic_spans,
    sentence_chunks,
    sliding_chunks,
    whole_abstract_chunks,
)
from evidence_rag.retrieval_engineering.config import milvus_mode, require_revision
from evidence_rag.retrieval_engineering.llm import chat_complete, hyde_passage, llm_settings, rewrite_query
from evidence_rag.retrieval_engineering.metrics import latency_summary, macro_metrics
from evidence_rag.retrieval_engineering.plan import comparison_rows
from evidence_rag.retrieval_engineering.report import render_markdown
from evidence_rag.retrieval_engineering.rerank import order_by_scores


class Doc:
    def __init__(self, doc_id, title, abstract):
        self.doc_id = doc_id
        self.title = title
        self.abstract = tuple(abstract)


def test_macro_recall_mrr_and_ndcg_match_hand_calculation():
    # Query A gold {10, 20}, rank [10, 30, 20]. Query B gold {20}, rank [10, 30].
    rankings = [([10, 30, 20, 40], {10, 20}), ([10, 30], {20})]
    scored = macro_metrics(rankings, ks=(1, 3, 5, 10, 20), mrr_k=10, ndcg_k=10)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    ndcg_a = (1 / math.log2(2) + 1 / math.log2(4)) / ideal
    assert scored["queries"] == 2
    assert scored["recall"]["1"] == pytest.approx(0.25)
    assert scored["recall"]["3"] == pytest.approx(0.5)
    assert scored["recall"]["5"] == pytest.approx(0.5)
    assert scored["mrr_at_10"] == pytest.approx(0.5)
    assert scored["ndcg_at_10"] == pytest.approx(ndcg_a / 2)
    assert scored["gold_documents"] == 3


def test_short_ranking_does_not_invent_hits_and_duplicates_fail():
    scored = macro_metrics([([7], {7, 8})], ks=(1, 5), mrr_k=10, ndcg_k=10)
    assert scored["recall"]["1"] == pytest.approx(0.5)
    assert scored["recall"]["5"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="duplicate"):
        macro_metrics([([1, 1], {1})])
    with pytest.raises(ValueError, match="eligible"):
        macro_metrics([])


def test_score_claims_times_every_claim_and_averages_only_gold():
    from evidence_rag.retrieval_engineering.metrics import score_claims

    class Claim:
        def __init__(self, claim_id, text, evidence):
            self.claim_id, self.text, self.evidence = claim_id, text, evidence

    claims = {2: Claim(2, "two", {}), 1: Claim(1, "one", {10: object()})}
    calls = []

    def search(text):
        calls.append(text)
        doc_id = 10 if text == "one" else 99
        return [{"doc_id": doc_id, "rank": 1, "score": 1.0}]

    scored = score_claims(claims, search, ranking_depth=5, ks=(1, 5), warmup=True)
    assert calls == ["one", "one", "two"]
    assert scored["eligible_claims"] == 1
    assert scored["queries_timed"] == 2
    assert scored["latency_seconds"]["n"] == 2
    assert scored["recall"]["1"] == pytest.approx(1.0)
    assert scored["gold_documents"] == 1


def test_latency_summary_reports_mean_and_median():
    summary = latency_summary([1.0, 3.0, 2.0, 4.0])
    assert summary == {"mean": 2.5, "median": 2.5, "n": 4}


def test_chunkers_keep_document_identity_and_original_sentence_spans():
    document = Doc(7, "Title", ["One.", "Two.", "Three."])
    whole = whole_abstract_chunks(document)
    assert whole == [{
        "doc_id": 7, "chunk_index": 0, "sentence_start": 0, "sentence_end": 3,
        "text": "Title\nOne. Two. Three.",
    }]
    sentences = sentence_chunks(document)
    assert [row["text"] for row in sentences] == ["Title\nOne.", "Title\nTwo.", "Title\nThree."]
    assert [(row["sentence_start"], row["sentence_end"]) for row in sentences] == [(0, 1), (1, 2), (2, 3)]
    sliding = sliding_chunks(document, window=2, stride=1)
    assert [row["text"] for row in sliding] == ["Title\nOne. Two.", "Title\nTwo. Three."]
    assert sliding_chunks(Doc(1, "Solo", ["Only."]), window=2, stride=1)[0]["text"] == "Solo\nOnly."
    with pytest.raises(ValueError):
        sliding_chunks(document, window=0, stride=1)


def test_semantic_chunker_splits_on_low_similarity_and_max_length():
    spans = semantic_spans([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], threshold=0.5, max_sentences=3)
    assert spans == [(0, 2), (2, 3)]
    capped = semantic_spans([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], threshold=0.5, max_sentences=2)
    assert capped == [(0, 2), (2, 3)]
    with pytest.raises(ValueError):
        semantic_spans([], threshold=0.5, max_sentences=2)


def test_document_aggregation_keeps_best_score_and_breaks_ties_by_id():
    hits = [
        {"doc_id": 3, "score": 0.2, "chunk_index": 0},
        {"doc_id": 2, "score": 0.4, "chunk_index": 1},
        {"doc_id": 3, "score": 0.9, "chunk_index": 4},
        {"doc_id": 5, "score": 0.4, "chunk_index": 0},
    ]
    ranked = aggregate_documents(hits, depth=2)
    assert [row["doc_id"] for row in ranked] == [3, 2]
    assert ranked[0]["score"] == pytest.approx(0.9)
    assert [row["rank"] for row in ranked] == [1, 2]


def test_reranker_reorders_candidates_without_dropping_identity():
    ranked = order_by_scores([30, 10, 20], [0.1, 0.4, 0.4], depth=2)
    assert [(row["doc_id"], row["rank"]) for row in ranked] == [(10, 1), (20, 2)]
    with pytest.raises(ValueError, match="scores"):
        order_by_scores([1, 2], [0.3], depth=2)


def test_llm_stays_off_without_a_key_and_transforms_use_the_injected_writer():
    settings = llm_settings(
        {"api_key_env": "RETRIEVAL_LLM_API_KEY", "base_url_env": "RETRIEVAL_LLM_BASE_URL",
         "model_env": "RETRIEVAL_LLM_MODEL", "default_base_url": "https://api.openai.com/v1"},
        {},
    )
    assert settings["enabled"] is False
    assert "RETRIEVAL_LLM_API_KEY" in settings["reason"]
    seen = []

    def complete(prompt):
        seen.append(prompt)
        return "  rewritten query  " if len(seen) == 1 else "hypothetical abstract"

    assert rewrite_query("Statins reduce events.", complete) == "rewritten query"
    assert hyde_passage("Statins reduce events.", complete) == "hypothetical abstract"
    assert "Statins reduce events." in seen[0]
    assert "Statins reduce events." in seen[1]
    from evidence_rag.retrieval_engineering.llm import gated_transform
    sentinel = []
    skipped = gated_transform(settings, "bge_query_rewrite", "Query rewrite", lambda: sentinel.append("called"))
    assert skipped["status"] == "skipped"
    assert skipped["reason"] == settings["reason"]
    assert sentinel == []
    enabled = {**settings, "enabled": True}
    assert gated_transform(enabled, "bge_hyde", "HyDE", lambda: {"status": "measured"})["status"] == "measured"


def test_chat_complete_posts_openai_compatible_json_without_echoing_the_key():
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "only the query"}}]}).encode()

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return Response()

    text = chat_complete("https://example.test/v1/", "secret-token", "demo-model", "Rewrite this", opener, timeout=7)
    assert text == "only the query"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["auth"] == "Bearer secret-token"
    assert captured["body"]["model"] == "demo-model"
    assert captured["body"]["messages"][0]["content"] == "Rewrite this"
    assert captured["timeout"] == 7


def test_milvus_uri_distinguishes_lite_file_from_server():
    assert milvus_mode("var/milvus/scifact.db") == "lite"
    assert milvus_mode("http://localhost:19530") == "server"
    assert milvus_mode("https://milvus.example:19530") == "server"
    with pytest.raises(ValueError):
        milvus_mode("localhost:19530")
    assert require_revision("5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
    with pytest.raises(ValueError):
        require_revision("main")


def test_comparison_plan_marks_llm_steps_not_run_and_keeps_cpu_steps():
    rows = comparison_rows(include_llm=False)
    by_id = {row["id"]: row for row in rows}
    assert by_id["contriever_faiss"]["status"] == "pending"
    assert by_id["bge_milvus_whole"]["status"] == "pending"
    assert by_id["bge_milvus_rerank"]["status"] == "pending"
    assert by_id["bge_milvus_sentence"]["status"] == "pending"
    assert by_id["bge_milvus_sliding"]["status"] == "pending"
    assert by_id["bge_query_rewrite"]["status"] == "not_run"
    assert by_id["bge_hyde"]["status"] == "not_run"
    assert by_id["bge_query_rewrite"]["reason"] == "not yet run"
    assert by_id["bge_hyde"]["reason"] == "not yet run"


def test_markdown_table_is_generated_from_recorded_numbers_only():
    report = {
        "generated_at": "2026-11-12T00:00:00+00:00",
        "hardware": {"cpu_model": "test-cpu", "cpu_cores": 4, "ram_gib": 15.0, "gpu": "none"},
        "experiments": [
            {"id": "bge_milvus_whole", "label": "BGE / Milvus", "status": "measured",
             "recall": {"1": 0.5, "3": 0.5, "5": 0.5, "10": 0.5, "20": 0.5},
             "mrr_at_10": 0.25, "ndcg_at_10": 0.125,
             "latency_seconds": {"mean": 0.2, "median": 0.1, "n": 4},
             "index_build_seconds": 12.5},
            {"id": "bge_query_rewrite", "label": "Query rewrite", "status": "not_run", "reason": "not yet run"},
        ],
    }
    text = render_markdown(report)
    assert "0.5000" in text
    assert "0.2500" in text
    assert "0.1250" in text
    assert "0.200" in text
    assert "12.500" in text
    assert "not yet run" in text
    assert "0.0000" not in text
    broken = {"experiments": [{"id": "x", "label": "X", "status": "measured", "recall": {"1": 0.5}}]}
    with pytest.raises(ValueError, match="measured"):
        render_markdown(broken)


def test_bge_query_prefix_is_not_applied_to_passages():
    import numpy as np
    from evidence_rag.retrieval_engineering.models import BgeEmbedder

    seen = []

    class FakeEncoder:
        def encode(self, texts, **kwargs):
            seen.extend(texts)
            return np.ones((len(texts), 4), dtype=np.float32)

    embedder = BgeEmbedder(FakeEncoder(), "Represent this sentence for searching relevant passages: ", 4, 8)
    embedder.get_query_embedding("claim text")
    embedder.get_text_embedding("claim text")
    assert seen[0] == "Represent this sentence for searching relevant passages: claim text"
    assert seen[1] == "claim text"


def test_rewrite_command_skips_without_calling_the_endpoint(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_LLM_API_KEY", raising=False)
    monkeypatch.delenv("RETRIEVAL_LLM_MODEL", raising=False)
    from evidence_rag.retrieval_engineering.__main__ import _rewrite

    class Args:
        config = "configs/retrieval_engineering.yaml"

    report = _rewrite(Args())
    assert [row["id"] for row in report["experiments"]] == ["bge_query_rewrite", "bge_hyde"]
    assert {row["status"] for row in report["experiments"]} == {"skipped"}
    text = render_markdown(report)
    assert text.count("not yet run") >= 2


def test_milvus_lite_roundtrip_uses_llamaindex_and_fake_embeddings(tmp_path):
    from evidence_rag.retrieval_engineering.stores import HashEmbedder, MilvusChunkIndex

    embedder = HashEmbedder(dimension=8)
    nodes = [
        {"doc_id": 100, "chunk_index": 0, "text": "alpha title\nalpha abstract"},
        {"doc_id": 200, "chunk_index": 0, "text": "beta title\nbeta abstract"},
        {"doc_id": 100, "chunk_index": 1, "text": "alpha title\nalpha second sentence"},
    ]
    index = MilvusChunkIndex.build(
        tmp_path / "tiny.db", nodes, embedder, collection="tiny_scifact", overwrite=True,
    )
    hits = index.search("alpha abstract", chunk_k=3)
    assert {hit["doc_id"] for hit in hits} <= {100, 200}
    assert hits
    assert all(isinstance(hit["doc_id"], int) and isinstance(hit["score"], float) for hit in hits)
    ranked = aggregate_documents(hits, depth=2)
    assert ranked[0]["doc_id"] == 100
    index.close()
    reopened = MilvusChunkIndex.open(tmp_path / "tiny.db", embedder, "tiny_scifact")
    assert reopened.count == 3
    again = aggregate_documents(reopened.search("alpha abstract", chunk_k=3), depth=2)
    assert again[0]["doc_id"] == 100
    reopened.close()
    other = MilvusChunkIndex.build(
        tmp_path / "other.db", nodes[:2], embedder, collection="other_scifact", overwrite=True,
    )
    assert other.search("beta abstract", chunk_k=1)[0]["doc_id"] == 200
    other.close()
