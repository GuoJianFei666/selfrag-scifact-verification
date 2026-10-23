"""SelfRAG release loading and constrained task adaptation."""
import math
from pathlib import Path

from .common import digest, file_hash, parse_json, read_json, write_json
from .data import RELATIONS
from .serialization import TEMPLATE_VERSION, relation_candidates, relation_scores

MODEL_ID = "selfrag/selfrag_llama2_7b"
MODEL_REVISION = "190261383b0779ff66d2f95a73c7ad267d94b820"
REQUIRED_TOKENS = ("[Retrieval]", "[No Retrieval]", "[Relevant]", "[Irrelevant]", "[Fully supported]",
                   "[Partially supported]", "[No support / Contradictory]", "<paragraph>", "</paragraph>")


def validate_contract(model, tokenizer):
    config = model.config
    if config.vocab_size != 32016 or len(tokenizer) != 32016 or config.max_position_embeddings != 4096:
        raise ValueError("Loaded model/tokenizer does not match the frozen SelfRAG release")
    if model.get_input_embeddings().weight.shape[0] != 32016 or model.get_output_embeddings().weight.shape[0] != 32016:
        raise ValueError("Embedding or output vocabulary mismatch")
    if config.tie_word_embeddings:
        raise ValueError("SelfRAG input/output embeddings must not be tied")
    token_ids = []
    for token in REQUIRED_TOKENS:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1 or ids[0] == tokenizer.unk_token_id:
            raise ValueError(f"Missing single-token reflection marker: {token}")
        token_ids.append(ids[0])
    if len(token_ids) != len(set(token_ids)) or tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Control/pad token mappings are invalid")
    model.config.pad_token_id = tokenizer.pad_token_id
    return {"model_id": MODEL_ID, "revision": MODEL_REVISION, "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
            "control_tokens": dict(zip(REQUIRED_TOKENS, token_ids)), "template_version": TEMPLATE_VERSION}


def load_release(device="cpu", precision="float32", adapter=None, cache_dir=None, local_files_only=False, quantization="none"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if precision not in ("float32", "bfloat16", "float16"):
        raise ValueError("Unsupported precision")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True,
                                              cache_dir=cache_dir, local_files_only=local_files_only)
    extra = {}
    if quantization == "nf4":
        from transformers import BitsAndBytesConfig
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("NF4 training requires a supported CUDA/bitsandbytes environment")
        extra = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=getattr(torch, precision)),
                 "device_map": {"": device}}
    elif quantization != "none":
        raise ValueError("quantization must be none or nf4")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                torch_dtype=getattr(torch, precision), cache_dir=cache_dir,
                                                local_files_only=local_files_only, trust_remote_code=False, **extra)
    contract = validate_contract(model, tokenizer)
    contract.update(precision=precision, quantization=quantization, device=str(device))
    contract["adapter"] = None
    if adapter:
        from peft import PeftModel
        record = read_json(Path(adapter) / "run_manifest.json")
        if record["model_id"] != MODEL_ID or record["model_revision"] != MODEL_REVISION or record["template_version"] != TEMPLATE_VERSION:
            raise ValueError("Adapter base or template does not match SelfRAG")
        model = PeftModel.from_pretrained(model, str(adapter))
        contract["adapter"] = {"manifest_sha256": file_hash(Path(adapter) / "run_manifest.json"),
                               "weights": {p.name: file_hash(p) for p in sorted(Path(adapter).glob("adapter_model.*"))},
                               "run": record["run"], "training_views_sha256": record["training_views_sha256"]}
    return (model if quantization != "none" else model.to(device)), tokenizer, contract


def validate_output(raw, view, selected_relation=None):
    value = parse_json(raw)
    if not isinstance(value, dict) or set(value) != {"relation", "sentence_ids"}:
        raise ValueError("Model output must contain exactly relation and sentence_ids")
    relation, ids = value["relation"], value["sentence_ids"]
    if relation not in RELATIONS or (selected_relation and relation != selected_relation):
        raise ValueError("Invalid or changed relation")
    if not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < len(view.sentences) for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Sentence IDs must be unique and visible")
    if (relation == "NOINFO") != (len(ids) == 0):
        raise ValueError("NOINFO requires no evidence; other relations require evidence")
    return value


def cold_reload_check(verifier, probe_path, output):
    """Run in a fresh CLI process after loading the frozen base and saved adapter."""
    import torch
    probe = read_json(probe_path)
    if probe["model_revision"] != verifier.contract["revision"] or any(probe[k] != verifier.contract[k] for k in ("precision", "quantization")):
        raise ValueError("Cold reload probe and loaded model conditions differ")
    device = verifier.model.get_input_embeddings().weight.device
    ids = torch.tensor(probe["input_ids"], dtype=torch.long, device=device)
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] > verifier.max_length:
        raise ValueError("Invalid fixed reload probe")
    with torch.inference_mode():
        actual = verifier.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[:, -1].float().cpu()
    expected = torch.tensor(probe["last_token_logits"], dtype=torch.float32)
    tolerance = 1e-4 if probe["precision"] == "float32" else 2e-2
    if expected.shape != actual.shape or not torch.allclose(expected, actual, atol=tolerance, rtol=tolerance):
        raise ValueError("Cold reload fixed-input logits differ")
    record = {"fresh_process": True, "probe_sha256": file_hash(probe_path), "model_version": verifier.contract,
              "max_absolute_difference": float((expected - actual).abs().max()), "atol": tolerance, "rtol": tolerance}
    write_json(output, record)
    return record


class SelfRagVerifier:
    def __init__(self, model, tokenizer, contract, max_length=2048, max_new_tokens=96, temperature=1.0, threshold=0.0):
        if not math.isfinite(temperature) or temperature <= 0 or not 0 <= threshold <= 1:
            raise ValueError("Invalid calibration parameters")
        self.model, self.tokenizer, self.contract = model.eval(), tokenizer, contract
        self.max_length, self.max_new_tokens = max_length, max_new_tokens
        self.temperature, self.threshold = temperature, threshold

    def predict(self, view):
        cost = {"forward_calls": 0, "sequence_forward_calls": 0, "input_tokens": 0}
        module = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        def count_forward(module, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is not None:
                cost["forward_calls"] += 1
                cost["sequence_forward_calls"] += int(ids.shape[0])
                mask = kwargs.get("attention_mask")
                cost["input_tokens"] += int(mask.sum()) if mask is not None and mask.shape == ids.shape else int(ids.numel())
        hook = module.register_forward_pre_hook(count_forward, with_kwargs=True)
        try:
            result = self._predict(view)
            result["cost"] = cost
            return result
        finally:
            hook.remove()
            self.last_cost = cost

    def _predict(self, view):
        import torch
        sequences, shared = relation_candidates(view, self.tokenizer, self.max_length)
        with torch.inference_mode():
            scores = relation_scores(self.model, sequences, shared, self.tokenizer.pad_token_id)
            probs = torch.softmax(scores / self.temperature, dim=-1)
            index = int(probs.argmax())
            selected = RELATIONS[index]
            generated_ids, decoded_continuation = [], None
            if selected == "NOINFO":
                raw = '{"relation":"NOINFO","sentence_ids":[]}'
            else:
                prefix = sequences[index]
                if len(prefix) + self.max_new_tokens > min(self.max_length, 4096):
                    from .serialization import ContextLimitError
                    raise ContextLimitError("Input plus evidence generation exceeds frozen context limit")
                device = self.model.get_input_embeddings().weight.device
                ids = torch.tensor([prefix], dtype=torch.long, device=device)
                generated = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                                max_new_tokens=self.max_new_tokens, do_sample=False,
                                                pad_token_id=self.tokenizer.pad_token_id,
                                                eos_token_id=self.tokenizer.eos_token_id)
                suffix = self.tokenizer.decode(generated[0, len(prefix):], skip_special_tokens=False)
                generated_ids = generated[0, len(prefix):].tolist()
                decoded_continuation = suffix
                eos = self.tokenizer.eos_token
                if eos and suffix.endswith(eos):
                    suffix = suffix[:-len(eos)]
                raw = '{"relation":"' + selected + '","sentence_ids":' + suffix.strip()
        result = {"relation_candidate": selected, "label_scores": scores.tolist(),
                  "probabilities": dict(zip(RELATIONS, probs.tolist())), "confidence": float(probs[index]),
                  "raw_output": raw, "model_version": self.contract, "status": "invalid_output",
                  "relation_candidate_token_ids": sequences[index], "generated_token_ids": generated_ids,
                  "decoded_continuation": decoded_continuation,
                  "output_origin": "relation_scoring" if selected == "NOINFO" else "evidence_generation",
                  "original_sentence_ids": [], "evidence_sentences": [], "review_reason": None}
        try:
            parsed = validate_output(raw, view, selected)
        except ValueError as exc:
            result["review_reason"] = str(exc)
            return result
        local = parsed["sentence_ids"]
        result.update(view_sentence_ids=local, original_sentence_ids=[view.original_ids[i] for i in local],
                      evidence_sentences=[view.sentences[i] for i in local],
                      status="accepted" if result["confidence"] >= self.threshold else "review_required",
                      review_reason=None if result["confidence"] >= self.threshold else "below_calibration_threshold")
        return result
