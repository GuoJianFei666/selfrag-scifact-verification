"""Token-masked LoRA SFT with optional differentiable paired objectives."""
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time

from .common import digest, file_hash, read_json, utcnow, write_json, write_jsonl
from .model import MODEL_ID, MODEL_REVISION, load_release
from .data import validate_split
from .serialization import TEMPLATE_VERSION, collate, encode_supervised, relation_candidates, relation_scores
from .views import load_views, validate_intervention


def prepare_training(views_path, run, output, audit_path):
    """Create an explicit eligible set; never promote an unreviewed view."""
    if run not in ("B1", "B2", "B3"):
        raise ValueError("Training run must be B1/B2/B3")
    views = load_views(views_path)
    if any(v.split != "fit" for v in views):
        raise ValueError("Training preparation accepts only fit views")
    fulls = {(v.claim_id, v.doc_id): v for v in views if v.kind == "Full"}
    selected, excluded = [], []
    for view in views:
        reason = None
        if run == "B1" and view.kind != "Full":
            reason = "B1_uses_original_full_only"
        elif view.verification != "VERIFIED":
            reason = "semantic_review_" + view.verification.lower()
        elif view.kind != "Full":
            parent = fulls.get((view.claim_id, view.doc_id))
            if parent is None or parent.verification != "VERIFIED":
                raise ValueError("Reviewed intervention lacks a verified Full parent")
            validate_intervention(parent, view)
        if reason:
            excluded.append({"view_id": view.view_id, "kind": view.kind, "reason": reason})
        else:
            selected.append(view)
    if not selected or (run != "B1" and not any(v.kind != "Full" for v in selected)):
        raise ValueError("No eligible training set; B2/B3 require reviewed interventions")
    if any(Path(p).exists() for p in (output, audit_path)):
        raise FileExistsError("Prepared data and exclusion audit must be new files")
    write_jsonl(output, [v.asdict() for v in selected])
    record = {"run": run, "input_sha256": file_hash(views_path), "output_sha256": file_hash(output),
              "included": len(selected), "excluded": excluded}
    write_json(audit_path, record)
    return record


def pair_losses(full, keep=None, broken=None, margin=1.0):
    import torch
    import torch.nn.functional as F
    zero = full.sum() * 0
    directional, consistency = zero, zero
    if broken is not None:
        sufficient_full = torch.logsumexp(full[:2], dim=0) - full[2]
        sufficient_broken = torch.logsumexp(broken[:2], dim=0) - broken[2]
        directional = F.softplus(margin - (sufficient_full - sufficient_broken))
    if keep is not None:
        log_p, log_q = F.log_softmax(full.float(), dim=-1), F.log_softmax(keep.float(), dim=-1)
        log_m = torch.logaddexp(log_p, log_q) - math.log(2)
        consistency = 0.5 * ((log_p.exp() * (log_p - log_m)).sum() + (log_q.exp() * (log_q - log_m)).sum())
    return directional, consistency


def build_training_plan(views, run, seed, epochs):
    if run not in ("B1", "B2", "B3") or type(epochs) is not int or epochs < 1:
        raise ValueError("Train run must be B1/B2/B3 with positive integer epochs")
    if any(v.split != "fit" or v.verification != "VERIFIED" for v in views):
        raise ValueError("Training accepts only verified fit views")
    eligible = [v for v in views if run != "B1" or v.kind == "Full"]
    if not eligible:
        raise ValueError("No eligible training views")
    if run in ("B2", "B3") and not any(v.kind != "Full" for v in eligible):
        raise ValueError("B2/B3 require reviewed interventions")
    by_id = {v.view_id: v for v in eligible}
    if len(by_id) != len(eligible):
        raise ValueError("Duplicate SFT view IDs")
    sequence = []
    rng = random.Random(seed)
    for epoch in range(epochs):
        ids = sorted(by_id)
        rng.shuffle(ids)
        sequence.extend(ids)
    groups = defaultdict(dict)
    for view in eligible:
        key = (view.claim_id, view.doc_id)
        if view.kind in groups[key]:
            raise ValueError("First experiment uses one deterministic view per condition")
        groups[key][view.kind] = view
    pairs = {}
    for key, variants in groups.items():
        if "Full" not in variants:
            raise ValueError("Every training intervention requires its Full parent")
        if "Full" in variants:
            full = variants["Full"]
            for kind in ("Keep", "Break", "Neutral"):
                if kind in variants:
                    validate_intervention(full, variants[kind])
            pairs[full.view_id] = {kind: view.view_id for kind, view in variants.items() if kind in ("Full", "Keep", "Break")}
    return by_id, sequence, pairs


def validate_same_exposure(first, second):
    fields = ("training_views_sha256", "sft_sequence_sha256", "supervised_tokens", "optimizer_steps",
              "seed", "learning_rate", "gradient_accumulation", "epochs", "lora", "model_id", "model_revision", "precision", "quantization",
              "tokenizer_sha256", "template_version", "split_manifest_sha256")
    differences = [key for key in fields if first[key] != second[key]]
    if differences:
        raise ValueError("B2/B3 exposure or initialization differs: " + ", ".join(differences))


def train_lora(config, views_path, output, resume_adapter=None):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    defaults = {"seed": 42, "gradient_accumulation": 16, "precision": "float32", "quantization": "none",
                "lora": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
                "max_length": 2048, "margin": 1.0, "lambda_dir": 0.1, "lambda_keep": 0.1, "device": "cpu"}
    config = {**defaults, **config}
    if config["precision"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("Unsupported training precision")
    if not 0 < config["learning_rate"] <= 0.1 or config["lambda_dir"] < 0 or config["lambda_keep"] < 0:
        raise ValueError("Invalid optimizer or pair-loss weights")
    if not config.get("split_manifest"):
        raise ValueError("Training requires a complete source split manifest")
    split_manifest = read_json(config["split_manifest"])
    validate_split(split_manifest)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Training output must be empty")
    if config["run"] in ("B2", "B3"):
        selected = read_json(config["common_optimizer"])
        if selected["selected_on"] != "model_select" or selected["baseline"] != "B1":
            raise ValueError("B2/B3 must use the B1-selected common optimizer")
        if selected["learning_rate"] != config["learning_rate"]:
            raise ValueError("Learning rate differs from the frozen common optimizer")
    seed = config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    views = load_views(views_path)
    if config.get("split_manifest"):
        if any(split_manifest["assignments"].get(str(v.claim_id)) != "fit" or split_manifest["groups"].get(str(v.claim_id)) != v.source_group for v in views):
            raise ValueError("Training view split or source group differs from frozen split manifest")
    by_id, sequence, pairs = build_training_plan(views, config["run"], seed, config["epochs"])
    if config["run"] == "B3" and not any("Keep" in pair or "Break" in pair for pair in pairs.values()):
        raise ValueError("B3 has no reviewed valid pairs")
    b2 = None
    if config["run"] == "B3":
        if not config.get("b2_manifest"):
            raise ValueError("B3 requires a B2 run manifest for same-exposure verification")
        b2 = read_json(config["b2_manifest"])
        planned = {"training_views_sha256": file_hash(views_path), "sft_sequence_sha256": digest(sequence),
                   "split_manifest_sha256": file_hash(config["split_manifest"]) if config.get("split_manifest") else None,
                   **{k: config[k] for k in ("seed", "learning_rate", "gradient_accumulation", "epochs", "lora", "precision", "quantization")}}
        if b2["run"] != "B2" or any(b2[k] != v for k, v in planned.items()):
            raise ValueError("B3 planned configuration or exposure differs from B2")
    if resume_adapter:
        raise ValueError("Each comparison starts from the frozen base; resume is not enabled")
    model, tokenizer, contract = load_release(config["device"], config["precision"], cache_dir=config.get("cache_dir"),
                                               local_files_only=config.get("local_files_only", True),
                                               quantization=config.get("quantization", "none"))
    if config.get("quantization", "none") != "none":
        model = prepare_model_for_kbit_training(model)
    lora = config.get("lora", {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"]})
    model = get_peft_model(model, LoraConfig(**lora, task_type="CAUSAL_LM", bias="none"))
    encoded = {vid: encode_supervised(view, tokenizer, config["max_length"]) for vid, view in by_id.items()}
    accumulation = config.get("gradient_accumulation", 16)
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("Only LoRA parameters may be trainable")
    before = {name: p.detach().cpu().clone() for name, p in trainable.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=config["learning_rate"])
    model.train()
    model.config.use_cache = False
    output.mkdir(parents=True, exist_ok=True)
    forwards, tokens, steps, sft_input_tokens, pair_input_tokens = 0, 0, 0, 0, 0
    start = time.monotonic()
    logs = []
    lambda_dir = config.get("lambda_dir", 0.1) if config["run"] == "B3" else 0.0
    lambda_keep = config.get("lambda_keep", 0.1) if config["run"] == "B3" else 0.0
    device = model.get_input_embeddings().weight.device
    optimizer.zero_grad()
    chunks = []
    epoch_size = len(by_id)
    for epoch in range(config["epochs"]):
        for start_at in range(epoch * epoch_size, (epoch + 1) * epoch_size, accumulation):
            chunks.append((epoch + 1, sequence[start_at:min(start_at + accumulation, (epoch + 1) * epoch_size)]))
    checkpoint_records = []
    for chunk_index, (epoch, batch_ids) in enumerate(chunks):
        loss_value = 0.0
        batch_tokens = sum(sum(t != -100 for t in encoded[vid]["labels"]) for vid in batch_ids)
        valid_pairs = [vid for vid in batch_ids if vid in pairs and
                       ((lambda_dir and "Break" in pairs[vid]) or (lambda_keep and "Keep" in pairs[vid]))]
        for vid in batch_ids:
            batch = {k: v.to(device) for k, v in collate([encoded[vid]], tokenizer.pad_token_id).items()}
            token_count = sum(t != -100 for t in encoded[vid]["labels"])
            loss = model(**batch).loss * token_count / batch_tokens
            forwards += 1
            tokens += token_count
            sft_input_tokens += len(encoded[vid]["input_ids"])
            pair = pairs.get(vid)
            if vid in valid_pairs:
                scores = {}
                needed = ["Full"] + (["Break"] if lambda_dir and "Break" in pair else []) + (["Keep"] if lambda_keep and "Keep" in pair else [])
                devices = [device.index or 0] if device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed + steps * 997 + chunk_index)
                    for kind in needed:
                        seqs, shared = relation_candidates(by_id[pair[kind]], tokenizer, config["max_length"])
                        scores[kind] = relation_scores(model, seqs, shared, tokenizer.pad_token_id)
                        forwards += 1
                        pair_input_tokens += sum(map(len, seqs))
                directional, consistency = pair_losses(scores["Full"], scores.get("Keep"), scores.get("Break"), config.get("margin", 1.0))
                loss = loss + (lambda_dir * directional + lambda_keep * consistency) / len(valid_pairs)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            loss_value += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        steps += 1
        logs.append({"step": steps, "loss": loss_value, "supervised_tokens": tokens, "forward_calls": forwards})
        if chunk_index + 1 == len(chunks) or chunks[chunk_index + 1][0] != epoch:
            checkpoint = output / f"checkpoint-epoch-{epoch}"
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            model.eval()
            probe_ids = torch.tensor([encoded[sequence[0]]["input_ids"]], device=device)
            with torch.inference_mode():
                probe_logits = model(input_ids=probe_ids, attention_mask=torch.ones_like(probe_ids)).logits[:, -1].float().cpu()
            write_json(checkpoint / "reload_probe.json", {"input_ids": probe_ids.cpu().tolist(),
                       "last_token_logits": probe_logits.tolist(), "precision": config["precision"],
                       "quantization": config["quantization"], "model_revision": contract["revision"]})
            model.train()
            checkpoint_records.append({"path": str(checkpoint.resolve()), "epoch": epoch, "step": steps,
                                       "optimizer_steps": steps, "supervised_tokens": tokens, "forward_calls": forwards,
                                       "sft_input_tokens": sft_input_tokens, "pair_input_tokens": pair_input_tokens,
                                       "sft_sequence_sha256": digest(sequence[:epoch * epoch_size]),
                                       "validation_forward_calls": epoch,
                                       "epochs": epoch})
    changed = [name for name, parameter in trainable.items() if not torch.equal(before[name], parameter.detach().cpu())]
    if not changed:
        raise ValueError("No adapter parameter changed")
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    record = {"created_at": utcnow(), "run": config["run"], "model_id": contract["model_id"], "model_revision": contract["revision"],
              "template_version": TEMPLATE_VERSION, "contract": contract, "training_views_sha256": file_hash(views_path),
              "sft_sequence_sha256": digest(sequence), "supervised_tokens": tokens, "optimizer_steps": steps,
              "sft_input_tokens": sft_input_tokens, "pair_input_tokens": pair_input_tokens,
              "split_manifest_sha256": file_hash(config["split_manifest"]) if config.get("split_manifest") else None,
              "forward_calls": forwards, "elapsed_seconds": time.monotonic() - start,
              "validation_forward_calls": len(checkpoint_records) + 2,
              "changed_adapter_parameters": changed, "trainable_parameter_names": sorted(trainable),
              "config": config, **{key: config.get(key, "none") for key in
                                  ("seed", "learning_rate", "gradient_accumulation", "epochs", "lora", "precision", "quantization")}}
    record["tokenizer_sha256"] = digest({"vocabulary": tokenizer.get_vocab(), "special_tokens": tokenizer.special_tokens_map})
    record["checkpoints"] = checkpoint_records
    for checkpoint in checkpoint_records:
        checkpoint_manifest = {**record, **{k: v for k, v in checkpoint.items() if k not in ("path", "step", "epoch")},
                               "planned_sft_sequence_sha256": record["sft_sequence_sha256"], "checkpoint": checkpoint}
        if config["run"] == "B3":
            counterpart = next(c for c in b2["checkpoints"] if c["epoch"] == checkpoint["epoch"])
            validate_same_exposure(read_json(Path(counterpart["path"]) / "run_manifest.json"), checkpoint_manifest)
            checkpoint_manifest["b2_exposure_verified"] = True
        write_json(Path(checkpoint["path"]) / "run_manifest.json", checkpoint_manifest)
    write_json(output / "run_manifest.json", record)
    write_json(output / "training_log.json", logs)
    # Reload the saved adapter on the same base in memory, then compare fixed-input logits.
    probe = {k: v.to(device) for k, v in collate([encoded[sequence[0]]], tokenizer.pad_token_id).items() if k != "labels"}
    model.eval()
    with torch.inference_mode():
        reference = model(**probe).logits[:, -1].float().cpu()
    write_json(output / "reload_probe.json", {"input_ids": probe["input_ids"].cpu().tolist(),
               "last_token_logits": reference.tolist(), "precision": config["precision"],
               "quantization": config["quantization"], "model_revision": contract["revision"]})
    model.load_adapter(str(output), adapter_name="reload_check")
    model.set_adapter("reload_check")
    with torch.inference_mode():
        reloaded = model(**probe).logits[:, -1].float().cpu()
    tolerance = 1e-4 if config["precision"] == "float32" else 2e-2
    max_diff = float((reference - reloaded).abs().max())
    if not torch.allclose(reference, reloaded, atol=tolerance, rtol=tolerance):
        raise ValueError("Adapter reload logits differ from saved model")
    record["reload_check"] = {"max_absolute_difference": max_diff, "atol": tolerance, "rtol": tolerance, "same_base": True}
    if config["run"] == "B3":
        validate_same_exposure(b2, record)
        record["b2_exposure_verified"] = True
    write_json(output / "run_manifest.json", record)
    return record
