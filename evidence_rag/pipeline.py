"""Bind retrieved documents to structured model results and persist full reports."""
from dataclasses import replace
import logging
from pathlib import Path
from threading import Lock
import time
from uuid import uuid4

from .common import read_json, utcnow, write_json
from .data import Claim
from .views import full_view

LOG = logging.getLogger(__name__)


def predict_view(view, verifier):
    start = time.monotonic()
    try:
        result = verifier.predict(view)
    except RuntimeError as exc:
        LOG.exception("Model execution failed for %s", view.view_id)
        result = {"status": "execution_failed", "relation_candidate": None, "review_reason": str(exc),
                  "raw_output": None, "confidence": None, "original_sentence_ids": [], "evidence_sentences": [],
                  "cost": getattr(verifier, "last_cost", None)}
    result.update(view_id=view.view_id, claim_id=view.claim_id, doc_id=view.doc_id,
                  elapsed_seconds=time.monotonic() - start)
    return result


class Engine:
    def __init__(self, dataset, index=None, verifier=None, reports="var/reports"):
        self.dataset, self.index, self.verifier = dataset, index, verifier
        self.reports = Path(reports)
        self.lock = Lock()

    def health(self):
        return {"retrieval_ready": self.index is not None, "generation_ready": self.verifier is not None,
                "corpus_documents": len(self.dataset.corpus), "corpus_hash": self.dataset.corpus_hash,
                "retrieval": self.index.manifest.get("encoder") if self.index else None,
                "model": self.verifier.contract if self.verifier else None,
                "scope": "English abstracts; one claim-document relation at a time"}

    def search(self, text, top_k, document_ids=None):
        if self.index is None:
            raise RuntimeError("Retrieval index is not configured")
        with self.lock:
            hits = self.index.search(text, top_k, document_ids)
        return [{**hit, "title": self.dataset.corpus[hit["doc_id"]].title,
                 "sentences": list(self.dataset.corpus[hit["doc_id"]].abstract)} for hit in hits]

    def verify(self, text, top_k=5, document_ids=None):
        if self.verifier is None:
            raise RuntimeError("SelfRAG model weights are not loaded; no verification prediction was produced")
        hits = self.search(text, top_k, document_ids)
        if not hits:
            raise ValueError("Document scope contains no retrievable abstracts")
        request_id = uuid4().hex
        claim = Claim(0, text, (), {}, "request", False)
        results = []
        with self.lock:
            for hit in hits:
                document = self.dataset.corpus[hit["doc_id"]]
                view = full_view(claim, document)
                result = predict_view(view, self.verifier)
                result.update(title=document.title, retrieval_rank=hit["rank"], retrieval_score=hit["score"],
                              source={"doc_id": document.doc_id, "corpus_hash": self.dataset.corpus_hash,
                                      "document_hash": document.content_hash})
                results.append(result)
        report = {"request_id": request_id, "claim": text, "created_at": utcnow(), "top_k": top_k,
                  "document_scope": document_ids, "retrieval_version": self.index.manifest,
                  "model_version": self.verifier.contract, "results": results}
        write_json(self.reports / f"{request_id}.json", report)
        return report

    def report(self, request_id):
        if len(request_id) != 32 or any(c not in "0123456789abcdef" for c in request_id):
            raise ValueError("Invalid report ID")
        return read_json(self.reports / f"{request_id}.json")
