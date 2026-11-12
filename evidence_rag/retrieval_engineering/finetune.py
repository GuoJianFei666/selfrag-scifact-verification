"""CPU fine-tune of bge-small with in-batch and mined hard negatives."""
import gc
import logging
import time

from ..common import utcnow
from ..data import Dataset
from .chunking import whole_abstract_chunks
from .finetune_data import build_examples, gold_document_ids, pack_batches, split_train_claims
from .models import ensure_snapshot, load_sentence_transformer
from .run import BGE_IGNORE, configure_threads, hardware_info

log = logging.getLogger("evidence_rag.retrieval_engineering")


def run_finetune(data_dir, cfg, cache, model_dir):
    """Train on SciFact train. Returns a record and leaves weights in model_dir."""
    configure_threads()
    spec = cfg["finetune"]
    embedding = cfg["embedding"]
    data = Dataset(data_dir)
    train_claims = list(data.claims["train"].values())
    fit_claims, monitor_claims = split_train_claims(train_claims, float(spec["monitor_fraction"]), int(spec["seed"]))
    log.info("fit claims=%s monitor claims=%s", len(fit_claims), len(monitor_claims))
    started_at = utcnow()
    ensure_snapshot(embedding["model_id"], embedding["revision"], cache, BGE_IGNORE)
    started = time.perf_counter()
    model = load_sentence_transformer(embedding, cache, model_path=None)
    model.max_seq_length = int(spec["max_length"])
    load_seconds = time.perf_counter() - started
    mine_started = time.perf_counter()
    rankings = mine_hard_negatives(
        model, fit_claims + monitor_claims, data.corpus, int(embedding["batch_size"]), embedding["query_prefix"],
    )
    mine_seconds = time.perf_counter() - mine_started
    window, stride = int(cfg["chunking"]["sliding_window"]), int(cfg["chunking"]["sliding_stride"])
    prefix = embedding["query_prefix"]
    fit_examples = build_examples(fit_claims, data.corpus, rankings, window, stride, prefix)
    monitor_examples = build_examples(monitor_claims, data.corpus, rankings, window, stride, prefix)
    log.info("fit examples=%s monitor examples=%s", len(fit_examples), len(monitor_examples))
    loss_before = {
        "fit": mean_ranking_loss(model, fit_examples, int(spec["batch_size"]), float(spec["loss_scale"])),
        "monitor": mean_ranking_loss(model, monitor_examples, int(spec["batch_size"]), float(spec["loss_scale"])),
    }
    train_started = time.perf_counter()
    history = train_ranking(model, fit_examples, spec)
    train_seconds = time.perf_counter() - train_started
    loss_after = {
        "fit": mean_ranking_loss(model, fit_examples, int(spec["batch_size"]), float(spec["loss_scale"])),
        "monitor": mean_ranking_loss(model, monitor_examples, int(spec["batch_size"]), float(spec["loss_scale"])),
    }
    model.eval()
    model.save(str(model_dir))
    train_gold = gold_document_ids(train_claims)
    dev_gold = gold_document_ids(data.claims["dev"].values())
    record = {
        "status": "completed",
        "started_at": started_at,
        "base_model_id": embedding["model_id"],
        "base_revision": embedding["revision"],
        "seed": int(spec["seed"]),
        "epochs": int(spec["epochs"]),
        "batch_size": int(spec["batch_size"]),
        "learning_rate": float(spec["learning_rate"]),
        "warmup_ratio": float(spec["warmup_ratio"]),
        "weight_decay": float(spec["weight_decay"]),
        "max_length": int(spec["max_length"]),
        "loss": spec["loss"],
        "loss_scale": float(spec["loss_scale"]),
        "hard_negatives_per_example": 1,
        "hard_negative_source": "baseline bge-small whole-abstract cosine, gold evidence documents removed, then one lexical sliding window",
        "positive": f"one sliding window (window {window}, stride {stride}) per gold document, maximum rationale overlap",
        "query_prefix_on_anchor": True,
        "monitor_fraction": float(spec["monitor_fraction"]),
        "monitor_selection": "seeded shuffle of eligible train claim ids; int(n * fraction), at least 1 and at most n-1",
        "train_claims": len(train_claims),
        "eligible_train_claims": sum(1 for claim in train_claims if claim.evidence),
        "fit_claims": len(fit_claims),
        "monitor_claims": len(monitor_claims),
        "fit_examples": len(fit_examples),
        "monitor_examples": len(monitor_examples),
        "unique_train_gold_documents": len(train_gold),
        "dev_gold_documents_counted_for_overlap_only": len(dev_gold),
        "train_dev_gold_document_overlap": len(train_gold & dev_gold),
        "model_load_seconds": load_seconds,
        "mine_seconds": mine_seconds,
        "train_seconds": train_seconds,
        "epoch_train_loss_mean": history["epoch_loss_mean"],
        "optimizer_steps": history["steps"],
        "loss_before": loss_before,
        "loss_after": loss_after,
        "loss_mode": "eval-mode MultipleNegativesRankingLoss, dropout off, same pairs before and after the update",
        "hardware": hardware_info(),
        "model_dir": str(model_dir),
        "weights_committed": False,
        "dev_used_for_training": False,
        "dev_used_for_selection": False,
    }
    del model
    _release()
    probe = probe_reranker_step(cfg["reranker"], cache)
    record["reranker"] = probe
    record["finished_at"] = utcnow()
    return record


def mine_hard_negatives(model, claims, corpus, batch_size, query_prefix):
    """Rank the corpus with the frozen baseline encoder and drop each claim's gold documents."""
    import numpy as np
    documents = [corpus[doc_id] for doc_id in sorted(corpus)]
    doc_ids = [document.doc_id for document in documents]
    texts = [whole_abstract_chunks(document)[0]["text"] for document in documents]
    matrix = model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=len(texts) > batch_size,
    )
    queries = [query_prefix + claim.text for claim in claims]
    query_matrix = model.encode(
        queries, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=len(queries) > batch_size,
    )
    rankings = {}
    id_array = np.asarray(doc_ids)
    for row, claim in enumerate(claims):
        order = np.argsort(-query_matrix[row] @ matrix.T, kind="mergesort")
        gold = set(claim.evidence)
        picked = [int(doc_id) for doc_id in id_array[order] if int(doc_id) not in gold]
        if not picked:
            raise ValueError(f"Claim {claim.claim_id} retrieved only gold documents")
        rankings[claim.claim_id] = picked
    return rankings


def train_ranking(model, examples, spec):
    import torch
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    seed = int(spec["seed"])
    epochs = int(spec["epochs"])
    batch_size = int(spec["batch_size"])
    learning_rate = float(spec["learning_rate"])
    torch.manual_seed(seed)
    loss_fn = MultipleNegativesRankingLoss(model, scale=float(spec["loss_scale"]))
    decay, plain = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or "LayerNorm" in name or "layer_norm" in name:
            plain.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(spec["weight_decay"])}, {"params": plain, "weight_decay": 0.0}],
        lr=learning_rate,
    )
    planned = sum(len(pack_batches(examples, batch_size, seed, epoch)) for epoch in range(epochs))
    warmup_steps = int(planned * float(spec["warmup_ratio"]))
    model.train()
    epoch_means = []
    step = 0
    log.info("optimizer steps=%s warmup_steps=%s", planned, warmup_steps)
    for epoch in range(epochs):
        losses = []
        for batch in pack_batches(examples, batch_size, seed, epoch):
            _set_lr(optimizer, _schedule(step, planned, warmup_steps, learning_rate))
            features = [model.tokenize([row[key] for row in batch]) for key in ("anchor", "positive", "negative")]
            optimizer.zero_grad(set_to_none=True)
            value = loss_fn(features, labels=None)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(value.detach()))
            step += 1
        mean = sum(losses) / len(losses)
        epoch_means.append(mean)
        log.info("epoch %s train_loss_mean=%.4f steps=%s", epoch + 1, mean, len(losses))
    if step != planned:
        raise RuntimeError(f"Expected {planned} optimizer steps, ran {step}")
    return {"steps": step, "epoch_loss_mean": epoch_means}


def mean_ranking_loss(model, examples, batch_size, scale):
    import torch
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    loss_fn = MultipleNegativesRankingLoss(model, scale=scale)
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            features = [model.tokenize([row[key] for row in batch]) for key in ("anchor", "positive", "negative")]
            value = loss_fn(features, labels=None)
            total += float(value) * len(batch)
            count += len(batch)
    if was_training:
        model.train()
    if count != len(examples):
        raise RuntimeError("Ranking loss did not cover every example")
    return total / count


def probe_reranker_step(spec, cache):
    """Time one discarded CPU step. The updated weights are not saved or evaluated."""
    import torch
    from sentence_transformers import CrossEncoder
    ensure_snapshot(spec["model_id"], spec["revision"], cache, BGE_IGNORE)
    cross = CrossEncoder(
        spec["model_id"], revision=spec["revision"], max_length=256, device="cpu",
        cache_dir=cache, local_files_only=True,
    )
    cross.model.train()
    left = ["synthetic claim about a protein binding result"] * 8
    right = ["Synthetic title\n" + " ".join(["synthetic sentence"] * 40)] * 8
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float)
    encoded = cross.tokenizer(left, right, padding=True, truncation=True, max_length=256, return_tensors="pt")
    optimizer = torch.optim.AdamW(cross.model.parameters(), lr=2e-5)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logits = cross.model(**encoded).logits.view(-1)
    value = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    value.backward()
    optimizer.step()
    seconds = time.perf_counter() - started
    loss_value = float(value.detach())
    del cross, optimizer, value
    _release()
    return {
        "status": "skipped",
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "reason": "Only the bi-encoder was fine-tuned. One AdamW step on 8 synthetic pairs was timed and discarded.",
        "probe_batch_size": 8,
        "probe_step_seconds": seconds,
        "probe_loss": loss_value,
        "probe_texts": "synthetic",
    }


def _schedule(step, total_steps, warmup_steps, learning_rate):
    if total_steps < 1:
        raise ValueError("Training requires at least one step")
    if warmup_steps and step < warmup_steps:
        return learning_rate * float(step + 1) / float(warmup_steps)
    remain = max(total_steps - warmup_steps, 1)
    progress = float(step - warmup_steps) / float(remain)
    return learning_rate * max(0.0, 1.0 - progress)


def _set_lr(optimizer, learning_rate):
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _release():
    gc.collect()
