"""CPU tests for the 8GB QLoRA and 4-bit inference path. No 7B weights."""
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from evidence_rag.common import file_hash, write_json
from evidence_rag.model import MODEL_ID, MODEL_REVISION


ROOT = Path(__file__).resolve().parents[1]


def load_config(name):
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def inference_config(adapter=None):
    return {"model_id": MODEL_ID, "revision": MODEL_REVISION, "device": "cuda", "precision": "bfloat16",
            "quantization": "nf4", "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
            "max_length": 2048, "max_new_tokens": 96, "temperature": 1.0, "threshold": 0.0,
            "adapter": adapter, "merge_adapter": False}


class FlakyVerifier:
    contract = {"model_id": "engineering-test/fourbit-fixture", "quantization": "nf4"}

    def predict(self, view):
        if view.doc_id == 200:
            raise RuntimeError("synthetic execution failure")
        return {"status": "accepted", "relation_candidate": "NOINFO", "confidence": 0.4,
                "original_sentence_ids": [], "view_sentence_ids": [], "evidence_sentences": []}


def test_published_8gb_configs_share_one_nf4_base():
    from evidence_rag.fourbit import nf4_options, require_8gb_inference, require_8gb_training
    b0, train, infer = load_config("B0-4bit.yaml"), load_config("B1-4bit.yaml"), load_config("B1-4bit-infer.yaml")
    require_8gb_inference(b0, "B0")
    require_8gb_inference(infer, "B1")
    require_8gb_training(train)
    assert b0["adapter"] is None and infer["adapter"]
    assert b0["revision"] == infer["revision"] == MODEL_REVISION
    assert b0["model_id"] == infer["model_id"] == MODEL_ID
    shared = ("precision", "quantization", "bnb_4bit_quant_type", "bnb_4bit_use_double_quant",
              "max_length", "max_new_tokens", "temperature", "threshold")
    assert all(b0[key] == infer[key] for key in shared)
    assert train["quantization"] == "nf4" and train["bnb_4bit_use_double_quant"] is True
    assert train["precision"] == b0["precision"] == "bfloat16"
    assert nf4_options("bfloat16") == {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                       "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "bfloat16"}
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        nf4_options("float32")
    with pytest.raises(ValueError, match="merge"):
        require_8gb_training({**train, "merge_adapter": True})
    with pytest.raises(ValueError, match="checkpoint"):
        require_8gb_training({**train, "gradient_checkpointing": False})


def test_adapter_override_keeps_the_nf4_base():
    from evidence_rag.fourbit import apply_adapter_override, require_8gb_inference
    cfg = apply_adapter_override(inference_config(), "var/B1-smoke", "B1")
    require_8gb_inference(cfg, "B1")
    assert cfg["adapter"] == "var/B1-smoke" and cfg["quantization"] == "nf4" and cfg["precision"] == "bfloat16"
    with pytest.raises(ValueError, match="without an adapter"):
        apply_adapter_override(inference_config(), "var/B1-smoke", "B0")


def test_nf4_load_rejects_cpu_before_download(monkeypatch):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("downloaded")))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("downloaded")))
    from evidence_rag.model import load_release
    with pytest.raises(ValueError, match="NF4"):
        load_release(device="cpu", precision="bfloat16", quantization="nf4", local_files_only=True)


def test_cpu_snapshot_does_not_invent_vram():
    from evidence_rag.fourbit import cuda_memory_snapshot
    if torch.cuda.is_available():
        pytest.skip("CPU snapshot contract")
    snap = cuda_memory_snapshot()
    assert snap == {"cuda": False, "peak_allocated_bytes": None, "peak_reserved_bytes": None, "device_name": None}


def test_lengths_are_reported_without_truncation(full, tokenizer):
    from evidence_rag.fourbit import audit_lengths
    report = audit_lengths([full], tokenizer, 8)
    assert report["n"] == 1 and report["longest"] > 8
    assert report["overlong"] == [{"view_id": full.view_id, "tokens": report["longest"]}]


def test_paged_optimizer_names_bitsandbytes_when_import_fails(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "bitsandbytes" or name.startswith("bitsandbytes."):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    from evidence_rag.fourbit import build_optimizer
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = build_optimizer("adamw", [parameter], 1e-3)
    assert type(optimizer).__name__ == "AdamW"
    with pytest.raises(RuntimeError, match="bitsandbytes"):
        build_optimizer("paged_adamw_8bit", [parameter], 1e-3)
    with pytest.raises(ValueError, match="optimizer"):
        build_optimizer("sgd", [parameter], 1e-3)


def test_qlora_step_checkpoints_and_saves_adapter_only(full, tokenizer, split_manifest, tmp_path, monkeypatch):
    from evidence_rag.common import read_json
    from evidence_rag.training import train_lora
    base = __import__("tests.test_models_training", fromlist=["tiny_model"]).tiny_model(tokenizer)
    contract = {"model_id": "engineering-test/tiny-random-llama", "revision": "synthetic", "template_version": "scifact-json-v1"}
    monkeypatch.setattr("evidence_rag.training.load_release", lambda *args, **kwargs: (base, tokenizer, contract))
    views = tmp_path / "views.jsonl"
    from evidence_rag.common import write_jsonl
    write_jsonl(views, [full.asdict()])
    output = tmp_path / "adapter"
    record = train_lora({"run": "B1", "learning_rate": 1e-3, "epochs": 1, "gradient_accumulation": 1,
                         "split_manifest": str(split_manifest), "device": "cpu", "max_length": 2048,
                         "precision": "float32", "quantization": "none", "gradient_checkpointing": True,
                         "optimizer": "adamw", "save_adapter_only": True,
                         "lora": {"r": 2, "lora_alpha": 4, "lora_dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}},
                        views, output, limit=1)
    assert record["gradient_checkpointing"] is True and record["smoke_limit"] == 1
    assert record["changed_adapter_parameters"] and record["supervised_tokens"] > 0
    assert (output / "adapter_config.json").is_file()
    assert not (output / "pytorch_model.bin").exists() and not (output / "model.safetensors").exists()
    assert read_json(output / "run_manifest.json")["peak_vram"]["peak_allocated_bytes"] is None
    with pytest.raises(ValueError, match="merge"):
        train_lora({"run": "B1", "learning_rate": 1e-3, "epochs": 1, "merge_adapter": True,
                    "split_manifest": str(split_manifest)}, views, tmp_path / "merged")


def test_cited_smoke_keeps_failures_in_the_denominator(dataset, tmp_path):
    from evidence_rag.common import read_json, read_jsonl
    from evidence_rag.fourbit import run_experiment
    output = tmp_path / "smoke"
    result = run_experiment(dataset, dataset.claims["train"], FlakyVerifier(), output, group="B0",
                            config=inference_config(), cited=True, limit=1)
    assert result["smoke"] is True and result["condition"] == "cited" and result["claim_ids"] == [1]
    assert result["diagnostics"]["n"] == 2 and result["diagnostics"]["failure_rate"] == 0.5
    assert result["citation"] == {"n": 2, "valid": 1, "invalid": 1}
    assert result["official"]["abstract_label_only"]["relevant"] == 1
    assert result["official"]["abstract_label_only"]["correct"] == 0
    assert result["joint"] is None
    assert result["peak_vram"]["peak_allocated_bytes"] is None
    assert {row["status"] for row in result["per_example"]} == {"accepted", "execution_failed"}
    assert all(row["elapsed_seconds"] >= 0 for row in result["per_example"])
    saved = read_json(output / "results.json")
    assert saved["per_example"] == result["per_example"]
    assert len(read_jsonl(output / "predictions.jsonl")) == 2
    assert read_jsonl(output / "official.jsonl")[0]["evidence"] == {}
    with pytest.raises(ValueError, match="Smoke limit"):
        run_experiment(dataset, dataset.claims["train"], FlakyVerifier(), tmp_path / "too-many", group="B0",
                       config=inference_config(), cited=True, limit=9)


def test_retrieved_experiment_does_not_swap_in_cited_gold(dataset, tmp_path):
    from evidence_rag.fourbit import run_experiment
    cache = {"condition": "retrieved", "corpus_hash": dataset.corpus_hash, "split": "train", "partition": None,
             "top_k": 1, "hits": {"1": [{"doc_id": 200, "rank": 1, "score": 0.3}]}}
    result = run_experiment(dataset, dataset.claims["train"], FlakyVerifier(), tmp_path / "retrieved", group="B1",
                            config=inference_config("var/B1"), hits=cache)
    assert result["condition"] == "retrieved" and result["smoke"] is False
    assert [row["doc_id"] for row in result["per_example"]] == [200]
    assert result["diagnostics"] is None and result["joint"] is None
    assert result["citation"]["n"] == 1 and result["citation"]["invalid"] == 1
    assert result["official"]["abstract_label_only"]["relevant"] == 1
    with pytest.raises(ValueError, match="hits or cited"):
        run_experiment(dataset, dataset.claims["train"], FlakyVerifier(), tmp_path / "both", group="B0",
                       config=inference_config(), hits=cache, cited=True)


def test_final_protocol_rejects_smoke_adapter(dataset, split_manifest, tmp_path):
    from evidence_rag.protocol import freeze_files, require_final
    method_inputs = {"dataset_manifest": dataset.directory / "manifest.json", "split_manifest": split_manifest}
    for role in ("retrieval_config", "training_config", "selection_rule"):
        path = tmp_path / (role + ".json")
        write_json(path, {"synthetic_fixture": True})
        method_inputs[role] = path
    method = tmp_path / "method.json"
    freeze_files(method, method_inputs, {}, "method")
    view_inputs = {"method": method}
    for role in ("views", "quadruples", "exclusion_audit"):
        path = tmp_path / (role + ".json")
        write_json(path, [])
        view_inputs[role] = path
    view_protocol = tmp_path / "views-protocol.json"
    freeze_files(view_protocol, view_inputs, {"partition": "dev"}, "views")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    manifest = {"run": "B1", "smoke_limit": 2, "training_views_sha256": "synthetic"}
    write_json(adapter / "run_manifest.json", manifest)
    write_json(adapter / "adapter_config.json", {"peft_type": "LORA"})
    (adapter / "adapter_model.safetensors").write_bytes(b"not-a-real-weight")
    selection = {"selected_on": "model_select", "selected": {"path": str(adapter.resolve()), "baseline": "B1",
                 "checkpoint_manifest_sha256": file_hash(adapter / "run_manifest.json")}}
    choice = tmp_path / "choice.json"
    write_json(choice, selection)
    cfg = {"revision": MODEL_REVISION, "precision": "bfloat16", "quantization": "nf4", "temperature": 1.0,
           "threshold": 0.0, "adapter": str(adapter)}
    cfg_path = tmp_path / "model.json"
    write_json(cfg_path, cfg)
    calibration = tmp_path / "calibration.json"
    write_json(calibration, {"fit_on": "calibration_original_cited", "temperature": 1.0, "threshold": 0.0,
                             "model_version": {"revision": MODEL_REVISION, "precision": "bfloat16", "quantization": "nf4",
                                               "adapter": None}})
    hits = tmp_path / "hits.json"
    write_json(hits, {"synthetic_fixture": True})
    files = {"method": method, "view_protocol": view_protocol, "dataset_manifest": dataset.directory / "manifest.json",
             "split_manifest": split_manifest, "model_config": cfg_path, "retrieval_hits": hits,
             "calibration": calibration, "checkpoint_selection": choice,
             "adapter_manifest": adapter / "run_manifest.json", "adapter_config": adapter / "adapter_config.json",
             "adapter_weights": adapter / "adapter_model.safetensors"}
    params = {"baseline": "B1", "prediction_path": str(tmp_path / "predictions.jsonl"),
              "official_prediction_path": str(tmp_path / "official.jsonl"),
              "diagnostic_prediction_path": str(tmp_path / "diagnostic.jsonl")}
    final = tmp_path / "final.json"
    freeze_files(final, files, params, "final_run")
    with pytest.raises(ValueError, match="Smoke adapter"):
        require_final(final, dataset.directory, split_manifest, config=cfg_path, hits=hits,
                      prediction=params["prediction_path"])


def test_cli_lists_8gb_commands():
    result = subprocess.run([sys.executable, "-m", "evidence_rag", "--help"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    assert "experiment" in result.stdout and "audit-lengths" in result.stdout
