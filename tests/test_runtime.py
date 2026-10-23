from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient
import numpy as np
import pytest

from evidence_rag.app import create_app
from evidence_rag.common import read_json, write_json
from evidence_rag.pipeline import Engine, predict_view
from evidence_rag.protocol import calibrate, freeze_files, require_bindings, require_final, select_checkpoint, verify_frozen
from evidence_rag.retrieval import ExactIndex


class SyntheticEncoder:
    spec = {"model_id": "engineering-test/fixed-vectors", "revision": "synthetic"}
    def encode(self, texts, batch_size=16):
        return np.array([[1, 1] for _ in texts], dtype=np.float32)


def test_real_faiss_index_mapping_ties_and_scope(dataset, tmp_path):
    encoder = SyntheticEncoder()
    ExactIndex.build(tmp_path / "index", list(dataset.corpus.values())[::-1], encoder)
    index = ExactIndex(tmp_path / "index", dataset.corpus, encoder)
    assert [r["doc_id"] for r in index.search("synthetic", 2)] == [100, 200]
    assert [r["doc_id"] for r in index.search("synthetic", 2, [200])] == [200]
    changed = dict(dataset.corpus)
    changed[100] = replace(changed[100], title="Changed")
    with pytest.raises(ValueError, match="changed"):
        ExactIndex(tmp_path / "index", changed, encoder)


def test_freeze_detects_file_and_parameter_mutation(tmp_path):
    item = tmp_path / "input.json"; item.write_text("{}")
    path = tmp_path / "protocol.json"
    freeze_files(path, {"input": item}, {"temperature": 1.0}, "method")
    assert verify_frozen(path)["phase"] == "method"
    item.write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        verify_frozen(path)
    with pytest.raises(FileExistsError):
        freeze_files(path, {}, {}, "method")


def test_calibration_uses_only_held_out_cited_full(tmp_path):
    rows = [{"split": "calibration", "condition": "cited", "view_kind": "Full", "label_scores": [3., 1., 0.], "label_index": 0,
             "model_version": {"model_id": "engineering-test/fixed-scores"}}]
    record = calibrate(rows, tmp_path / "calibration.json", 0.7)
    assert record["nll_after"] <= record["nll_before"]
    with pytest.raises(ValueError):
        calibrate([{**rows[0], "split": "dev"}], tmp_path / "bad.json")


def test_checkpoint_selection_is_internal_and_prespecified(tmp_path):
    records = [{"split": "model_select", "abstract_rationalized": 0.5, "joint": None,
                "forward_calls": 100, "step": step, "path": str(step)} for step in (2, 1)]
    assert select_checkpoint(records, tmp_path / "chosen.json")["step"] == 1
    with pytest.raises(ValueError):
        select_checkpoint([{**records[0], "split": "dev"}], tmp_path / "bad.json")


def test_missing_weights_no_fake_verification_and_http_validation(dataset, tmp_path):
    engine = Engine(dataset, reports=tmp_path / "reports")
    with TestClient(create_app(engine)) as client:
        assert client.get("/").status_code == 200
        health = client.get("/api/health").json()
        assert not health["generation_ready"]
        assert client.post("/api/verify", json={"claim": "a claim"}).status_code == 503
        assert client.post("/api/search", json={"claim": "a claim"}).status_code == 503
        assert client.post("/api/search", json={"claim": "", "top_k": True}).status_code == 422
        assert client.post("/api/verify", json={"claim": "a claim"}, headers={"Origin": "https://other.example"}).status_code == 403
        assert client.get("/api/reports/not-a-report").status_code == 404


def test_engineering_fixture_binds_source_and_exports_report(dataset, full, tmp_path):
    class Index:
        manifest = {"test_fixture": True}
        def search(self, *args): return [{"doc_id": 100, "rank": 1, "score": 0.2}]
    class Verifier:
        contract = {"model_id": "engineering-test/fixed-output", "not_a_model_result": True}
        def predict(self, view):
            return {"relation_candidate": "SUPPORT", "status": "accepted", "view_sentence_ids": [1],
                    "original_sentence_ids": [1], "evidence_sentences": [view.sentences[1]], "confidence": 0.9}
    engine = Engine(dataset, Index(), Verifier(), tmp_path / "reports")
    report = engine.verify("synthetic request", 1)
    assert report["results"][0]["source"]["doc_id"] == 100
    assert engine.report(report["request_id"]) == report
    other = engine.verify("a different synthetic request", 1)
    assert other["results"][0]["view_id"] != report["results"][0]["view_id"]


def test_cli_help_is_executable():
    result = subprocess.run([sys.executable, "-m", "evidence_rag", "--help"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    assert "build-index" in result.stdout and "train" in result.stdout


def test_source_split_requires_complete_ids_and_indivisible_groups(dataset, split_manifest):
    from evidence_rag.data import validate_split
    manifest = read_json(split_manifest)
    validate_split(manifest)
    with pytest.raises(ValueError, match="all official"):
        validate_split({**manifest, "assignments": {}})
    with pytest.raises(ValueError, match="Unresolved source"):
        validate_split({**manifest, "groups": {}})


def test_checkpoint_workflow_uses_actual_retrieved_documents(dataset, tmp_path):
    from evidence_rag.workflow import checkpoint_evaluation
    class Verifier:
        contract = {"model_id": "engineering-test/noinfo-fixture"}
        def predict(self, view):
            return {"status": "accepted", "relation_candidate": "NOINFO", "confidence": 0.8,
                    "original_sentence_ids": [], "view_sentence_ids": [], "evidence_sentences": []}
    adapter = tmp_path / "adapter"
    write_json(adapter / "run_manifest.json", {"run": "B1", "optimizer_steps": 1, "forward_calls": 2, "learning_rate": 0.001})
    cache = {"condition": "retrieved", "corpus_hash": dataset.corpus_hash, "split": "train", "partition": "model_select",
             "top_k": 1, "hits": {"1": [{"doc_id": 200, "rank": 1, "score": 0.3}]}}
    output = tmp_path / "evaluation"
    record = checkpoint_evaluation(dataset, dataset.claims["train"], cache, Verifier(), adapter, output)
    from evidence_rag.common import read_jsonl
    assert [r["doc_id"] for r in read_jsonl(output / "predictions.jsonl")] == [200]
    assert record["abstract_rationalized"] == 0 and record["retrieval"]["document_recall"] == 0
    with pytest.raises(ValueError, match="condition"):
        checkpoint_evaluation(dataset, dataset.claims["train"], {**cache, "condition": "cited"}, Verifier(), adapter, tmp_path / "bad")


def test_unrelated_frozen_manifest_and_empty_partition_rejected(dataset, tmp_path):
    from evidence_rag.__main__ import select_claims
    unrelated = tmp_path / "unrelated.json"
    freeze_files(unrelated, {}, {}, "final_run")
    with pytest.raises(ValueError, match="required named inputs"):
        require_final(unrelated, dataset.directory, tmp_path / "missing.json")
    with pytest.raises(ValueError, match="Unknown internal"):
        select_claims(dataset, "train", "model-select", tmp_path / "splits.json")


def test_frozen_bindings_and_prediction_receipt(tmp_path):
    from evidence_rag.workflow import verify_receipt, write_receipt
    files = {}
    for role in ("dataset_manifest", "split_manifest", "retrieval_config", "training_config", "selection_rule"):
        item = tmp_path / (role + ".json"); item.write_text("{}")
        files[role] = item
    frozen = tmp_path / "method.json"
    freeze_files(frozen, files, {}, "method")
    require_bindings(frozen, "method", {"split_manifest": files["split_manifest"]})
    changed = tmp_path / "changed.json"; changed.write_text("[]")
    with pytest.raises(ValueError, match="Actual input"):
        require_bindings(frozen, "method", {"split_manifest": changed})
    output = tmp_path / "predictions.jsonl"; output.write_text("{}\n")
    write_receipt(output, frozen, [output]); verify_receipt(output, frozen, output)
    output.write_text("[]\n")
    with pytest.raises(ValueError, match="receipt changed"):
        verify_receipt(output, frozen, output)


def test_complete_frozen_final_protocol_binds_dev_and_calibration(dataset, split_manifest, tmp_path):
    from evidence_rag.model import MODEL_REVISION
    method_inputs = {"dataset_manifest": dataset.directory / "manifest.json", "split_manifest": split_manifest}
    for role in ("retrieval_config", "training_config", "selection_rule"):
        path = tmp_path / (role + ".json"); write_json(path, {"synthetic_fixture": True}); method_inputs[role] = path
    method = tmp_path / "method.json"; freeze_files(method, method_inputs, {}, "method")
    view_inputs = {"method": method}
    for role in ("views", "quadruples", "exclusion_audit"):
        path = tmp_path / (role + ".json"); write_json(path, []); view_inputs[role] = path
    view_protocol = tmp_path / "views-protocol.json"; freeze_files(view_protocol, view_inputs, {"partition": "dev"}, "views")
    cfg = {"revision": MODEL_REVISION, "precision": "float32", "quantization": "none", "temperature": 1., "threshold": 0., "adapter": None}
    cfg_path = tmp_path / "model.json"; write_json(cfg_path, cfg)
    cal = tmp_path / "calibration.json"
    write_json(cal, {"fit_on": "calibration_original_cited", "temperature": 1., "threshold": 0.,
                    "model_version": {"revision": MODEL_REVISION, "precision": "float32", "quantization": "none", "adapter": None}})
    hits = tmp_path / "hits.json"; write_json(hits, {"synthetic_fixture": True})
    choice = tmp_path / "choice.json"; write_json(choice, {"baseline": "B0", "selected_on": "frozen_release"})
    files = {"method": method, "view_protocol": view_protocol, "dataset_manifest": dataset.directory / "manifest.json",
             "split_manifest": split_manifest, "model_config": cfg_path, "retrieval_hits": hits, "calibration": cal, "checkpoint_selection": choice}
    params = {"baseline": "B0", "prediction_path": str(tmp_path / "predictions.jsonl"),
              "official_prediction_path": str(tmp_path / "official.jsonl"), "diagnostic_prediction_path": str(tmp_path / "diagnostic.jsonl")}
    final = tmp_path / "final.json"; freeze_files(final, files, params, "final_run")
    require_final(final, dataset.directory, split_manifest, config=cfg_path, hits=hits, prediction=params["prediction_path"])
    wrong_cfg = tmp_path / "wrong-config.json"; write_json(wrong_cfg, {**cfg, "threshold": 0.7})
    with pytest.raises(ValueError, match="Actual input"):
        require_final(final, dataset.directory, split_manifest, config=wrong_cfg)
