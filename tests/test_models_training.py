from dataclasses import replace
import math
from pathlib import Path

import pytest
import torch

from evidence_rag.common import read_json, write_json, write_jsonl
from evidence_rag.model import SelfRagVerifier, validate_output
from evidence_rag.serialization import ContextLimitError, collate, encode_supervised, prompt, relation_candidates, relation_scores
from evidence_rag.training import build_training_plan, pair_losses, prepare_training, train_lora, validate_same_exposure
from evidence_rag.views import delete_view


def tiny_model(tokenizer):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.set_num_threads(2)
    torch.manual_seed(7)
    return LlamaForCausalLM(LlamaConfig(vocab_size=len(tokenizer), hidden_size=24, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=False)).eval()


def test_sft_masks_input_and_padding_and_includes_eos(full, tokenizer):
    sample = encode_supervised(full, tokenizer)
    target = tokenizer.decode([i for i in sample["labels"] if i != -100])
    assert target == '{"relation":"CONTRADICT","sentence_ids":[11]}<eos>'
    assert sample["labels"][0] == -100
    short = {key: value[:-2] for key, value in sample.items()}
    batch = collate([sample, short], tokenizer.pad_token_id)
    assert batch["labels"][1, -1] == -100 and batch["attention_mask"][1, -1] == 0
    with pytest.raises(ContextLimitError):
        encode_supervised(full, tokenizer, 16)


def test_prompt_excludes_labels_intervention_and_source_metadata(full):
    alternate = replace(full, kind="Break", relation="NOINFO", rationales=(), split="dev", source_group="hidden-source")
    assert prompt(full) == prompt(alternate)
    assert "hidden-source" not in prompt(alternate)


def test_full_multitoken_label_likelihood_and_shift(full, tokenizer):
    model = tiny_model(tokenizer)
    sequences, shared = relation_candidates(full, tokenizer)
    scores = relation_scores(model, sequences, shared, tokenizer.pad_token_id)
    manual = []
    for sequence in sequences:
        ids = torch.tensor([sequence])
        logp = model(input_ids=ids).logits.float().log_softmax(-1)
        manual.append(sum(logp[0, j - 1, sequence[j]] for j in range(shared, len(sequence))))
    assert torch.allclose(scores, torch.stack(manual), atol=1e-4)
    assert all(len(sequence) - shared > 1 for sequence in sequences)


def test_pair_losses_are_stable_and_have_gradient():
    full = torch.tensor([2., 1., -1.], requires_grad=True)
    keep = torch.tensor([1.5, 1.1, -0.5], requires_grad=True)
    broken = torch.tensor([-1., -2., 2.], requires_grad=True)
    directional, js = pair_losses(full, keep, broken)
    (directional + js).backward()
    assert directional > 0 and js >= 0
    assert all(v.grad is not None and v.grad.abs().sum() > 0 for v in (full, keep, broken))
    _, same = pair_losses(full, full)
    _, reverse = pair_losses(keep, full)
    assert same.item() == pytest.approx(0, abs=1e-7) and reverse.item() == pytest.approx(js.item())


def test_b2_and_b3_have_identical_sft_sequence(full):
    views = [full, replace(delete_view(full, {0, 11}, "Break"), verification="VERIFIED")]
    _, b2, _ = build_training_plan(views, "B2", 42, 2)
    _, b3, _ = build_training_plan(views, "B3", 42, 2)
    assert b2 == b3
    with pytest.raises(ValueError, match="fit"):
        build_training_plan([replace(full, split="dev")], "B1", 42, 1)


@pytest.mark.parametrize("raw", ['{"relation":"NOINFO","sentence_ids":[0]}',
                                '{"relation":"SUPPORT","sentence_ids":[]}',
                                '{"relation":"SUPPORT","sentence_ids":[true]}',
                                '{"relation":"SUPPORT","sentence_ids":[99]}',
                                '{"relation":"SUPPORT","sentence_ids":[0,0]}',
                                '{"relation":"SUPPORT","relation":"NOINFO","sentence_ids":[]}'])
def test_invalid_model_output_is_not_noinfo(full, raw):
    with pytest.raises(ValueError):
        validate_output(raw, full)


def test_real_tiny_lora_update_and_cold_reload(full, tokenizer, split_manifest, tmp_path, monkeypatch):
    from peft import PeftModel
    from transformers import LlamaForCausalLM, AutoTokenizer
    base = tiny_model(tokenizer)
    base_path = tmp_path / "tiny-base"
    base.save_pretrained(base_path)
    frozen = {id(p): p.detach().clone() for p in base.parameters()}
    contract = {"model_id": "engineering-test/tiny-random-llama", "revision": "synthetic", "template_version": "scifact-json-v1"}
    monkeypatch.setattr("evidence_rag.training.load_release", lambda *a, **k: (base, tokenizer, contract))
    views_path = tmp_path / "views.jsonl"
    write_jsonl(views_path, [full.asdict()])
    cfg = {"run": "B1", "learning_rate": 1e-3, "epochs": 2, "gradient_accumulation": 1,
           "split_manifest": str(split_manifest),
           "device": "cpu", "max_length": 2048, "precision": "float32",
           "lora": {"r": 2, "lora_alpha": 4, "lora_dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}}
    output = tmp_path / "adapter"
    record = train_lora(cfg, views_path, output)
    assert record["changed_adapter_parameters"] and record["model_id"].startswith("engineering-test/")
    assert len(record["checkpoints"]) == 2 and record["optimizer_steps"] == 2
    early = read_json(output / "checkpoint-epoch-1" / "run_manifest.json")
    assert early["optimizer_steps"] == 1 and early["supervised_tokens"] * 2 == record["supervised_tokens"]
    assert early["sft_sequence_sha256"] != record["sft_sequence_sha256"]
    assert record["reload_check"]["max_absolute_difference"] == pytest.approx(0, abs=1e-6)
    assert all(torch.equal(frozen[id(p)], p.detach()) for p in base.parameters() if id(p) in frozen)
    reloaded = PeftModel.from_pretrained(LlamaForCausalLM.from_pretrained(base_path), output).eval()
    saved_tokenizer = AutoTokenizer.from_pretrained(output, use_fast=True)
    sample = encode_supervised(full, saved_tokenizer)
    ids = torch.tensor([sample["input_ids"]])
    with torch.inference_mode():
        assert torch.allclose(base(input_ids=ids).logits, reloaded(input_ids=ids).logits, atol=1e-5)
    mismatch = dict(record, supervised_tokens=record["supervised_tokens"] + 1)
    with pytest.raises(ValueError, match="differs"):
        validate_same_exposure(record, mismatch)


def test_explicit_training_preparation_and_orphan_rejection(full, tmp_path):
    path = tmp_path / "candidates.jsonl"
    broken = delete_view(full, {0, 11}, "Break")
    write_jsonl(path, [full.asdict(), broken.asdict()])
    audit = prepare_training(path, "B1", tmp_path / "b1.jsonl", tmp_path / "audit.json")
    assert audit["included"] == 1 and audit["excluded"][0]["reason"] == "B1_uses_original_full_only"
    with pytest.raises(ValueError, match="reviewed interventions"):
        prepare_training(path, "B2", tmp_path / "bad.jsonl", tmp_path / "bad-audit.json")
    with pytest.raises(ValueError, match="Full parent"):
        build_training_plan([replace(broken, verification="VERIFIED")], "B2", 42, 1)


def test_real_tiny_b2_b3_training_and_equal_exposure(full, tokenizer, split_manifest, tmp_path, monkeypatch):
    contract = {"model_id": "engineering-test/tiny-random-llama", "revision": "synthetic", "template_version": "scifact-json-v1"}
    monkeypatch.setattr("evidence_rag.training.load_release", lambda *a, **k: (tiny_model(tokenizer), tokenizer, contract))
    views_path = tmp_path / "reviewed.jsonl"
    broken = replace(delete_view(full, {0, 11}, "Break"), verification="VERIFIED", review_note="synthetic engineering fixture")
    write_jsonl(views_path, [full.asdict(), broken.asdict()])
    common = tmp_path / "common.json"
    write_json(common, {"selected_on": "model_select", "baseline": "B1", "learning_rate": 1e-3})
    cfg = {"learning_rate": 1e-3, "epochs": 1, "gradient_accumulation": 2, "device": "cpu",
           "split_manifest": str(split_manifest),
           "common_optimizer": str(common), "lora": {"r": 2, "lora_alpha": 4, "lora_dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}}
    b2 = train_lora({**cfg, "run": "B2"}, views_path, tmp_path / "b2")
    b3 = train_lora({**cfg, "run": "B3", "b2_manifest": str(tmp_path / "b2" / "run_manifest.json")}, views_path, tmp_path / "b3")
    validate_same_exposure(b2, b3)
    assert b3["b2_exposure_verified"] and b3["forward_calls"] > b2["forward_calls"]
    assert b3["pair_input_tokens"] > 0 and b2["pair_input_tokens"] == 0
    assert b3["changed_adapter_parameters"]
    with pytest.raises(ValueError, match="differs"):
        train_lora({**cfg, "run": "B3", "epochs": 2, "b2_manifest": str(tmp_path / "b2" / "run_manifest.json")}, views_path, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_positive_generation_continues_exact_scored_prefix(full, tokenizer, monkeypatch):
    view = delete_view(full, {0}, "Keep")
    model = tiny_model(tokenizer)
    selected = relation_candidates(view, tokenizer)[0][0]
    suffix = tokenizer.encode('[10]}', add_special_tokens=False) + [tokenizer.eos_token_id]
    def generate(**kwargs):
        assert kwargs["input_ids"].tolist() == [selected]
        return torch.tensor([selected + suffix])
    monkeypatch.setattr(model, "generate", generate)
    monkeypatch.setattr("evidence_rag.model.relation_scores", lambda *a: torch.tensor([4., 1., 0.]))
    verifier = SelfRagVerifier(model, tokenizer, {"model_id": "engineering-test/controlled-generation"})
    result = verifier.predict(view)
    assert result["status"] == "accepted" and result["view_sentence_ids"] == [10]
    assert result["original_sentence_ids"] == [11] and result["evidence_sentences"] == [full.sentences[11]]
    assert result["generated_token_ids"] == suffix and result["decoded_continuation"].endswith('<eos>')
