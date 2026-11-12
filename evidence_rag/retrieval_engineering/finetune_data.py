"""Train-only pairs for bi-encoder fine-tuning. Dev claims are rejected."""
import random
import re

from .chunking import sliding_chunks


def split_train_claims(claims, monitor_fraction, seed):
    """Hold out a seeded fraction of eligible train claims. The rest are fit claims."""
    _require_train(claims)
    if isinstance(monitor_fraction, bool) or not isinstance(monitor_fraction, (int, float)):
        raise ValueError("monitor_fraction must be a number")
    if not 0 < float(monitor_fraction) < 0.5:
        raise ValueError("monitor_fraction must be between 0 and 0.5")
    eligible = [claim for claim in claims if claim.evidence]
    if len(eligible) < 2:
        raise ValueError("Fine-tuning needs at least two train claims with evidence")
    order = sorted(claim.claim_id for claim in eligible)
    rng = random.Random(seed)
    rng.shuffle(order)
    monitor_count = int(len(order) * float(monitor_fraction))
    monitor_count = min(max(monitor_count, 1), len(order) - 1)
    monitor_ids = set(order[:monitor_count])
    by_id = {claim.claim_id: claim for claim in eligible}
    fit = [by_id[claim_id] for claim_id in order if claim_id not in monitor_ids]
    monitor = [by_id[claim_id] for claim_id in order if claim_id in monitor_ids]
    return fit, monitor


def build_examples(claims, corpus, rankings, window, stride, query_prefix):
    """One row per gold document: prefixed claim, best overlapping window, one hard-negative window."""
    _require_train(claims)
    if not isinstance(query_prefix, str) or not query_prefix.endswith(" "):
        raise ValueError("Query prefix must end with a space")
    examples = []
    for claim in claims:
        if not claim.evidence:
            continue
        gold = set(claim.evidence)
        negative_id, negative_text = _hard_negative(claim, corpus, rankings.get(claim.claim_id, []), gold, window, stride)
        for doc_id in claim.evidence:
            positive = best_positive_window(corpus[doc_id], claim.evidence[doc_id], window, stride)
            examples.append({
                "claim_id": claim.claim_id,
                "positive_doc_id": doc_id,
                "negative_doc_id": negative_id,
                "anchor": query_prefix + claim.text,
                "positive": positive["text"],
                "negative": negative_text,
            })
    if any(row["negative_doc_id"] in claim_gold(claims, row["claim_id"]) for row in examples):
        raise ValueError("A hard negative collided with a gold document")
    return examples


def best_positive_window(document, evidence, window, stride):
    gold = set()
    for rationale in evidence.rationales:
        gold.update(rationale)
    if not gold:
        raise ValueError(f"Evidence for document {document.doc_id} has no rationale sentences")
    ranked = []
    for chunk in sliding_chunks(document, window, stride):
        overlap = sum(1 for index in range(chunk["sentence_start"], chunk["sentence_end"]) if index in gold)
        if overlap:
            length = chunk["sentence_end"] - chunk["sentence_start"]
            ranked.append((-overlap, length, chunk["sentence_start"], chunk))
    if not ranked:
        raise ValueError(f"No sliding window covers a rationale in document {document.doc_id}")
    ranked.sort()
    return ranked[0][3]


def best_lexical_window(document, claim_text, window, stride):
    query = set(_tokens(claim_text))
    ranked = []
    for chunk in sliding_chunks(document, window, stride):
        overlap = len(query & set(_tokens(chunk["text"])))
        length = chunk["sentence_end"] - chunk["sentence_start"]
        ranked.append((-overlap, length, chunk["sentence_start"], chunk))
    if not ranked:
        raise ValueError(f"Document {document.doc_id} produced no windows")
    ranked.sort()
    return ranked[0][3]


def pack_batches(examples, batch_size, seed, epoch):
    """Shuffle within each epoch and keep a claim from appearing twice in one batch."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not examples:
        raise ValueError("Cannot pack an empty example list")
    by_claim = {}
    for row in examples:
        by_claim.setdefault(row["claim_id"], []).append(dict(row))
    rng = random.Random(seed + epoch * 1_000_003)
    claim_ids = list(by_claim)
    rng.shuffle(claim_ids)
    for rows in by_claim.values():
        rng.shuffle(rows)
    batches = []
    while any(by_claim.values()):
        pool = []
        for claim_id in claim_ids:
            rows = by_claim[claim_id]
            if rows:
                pool.append(rows.pop())
        for start in range(0, len(pool), batch_size):
            batches.append(pool[start:start + batch_size])
    return batches


def gold_document_ids(claims):
    found = set()
    for claim in claims:
        found.update(claim.evidence)
    return found


def claim_gold(claims, claim_id):
    for claim in claims:
        if claim.claim_id == claim_id:
            return set(claim.evidence)
    raise KeyError(claim_id)


def _hard_negative(claim, corpus, ranking, gold, window, stride):
    for doc_id in ranking:
        if doc_id in gold:
            continue
        if doc_id not in corpus:
            raise KeyError(f"Hard-negative document {doc_id} is not in the corpus")
        window_row = best_lexical_window(corpus[doc_id], claim.text, window, stride)
        return doc_id, window_row["text"]
    raise ValueError(f"Claim {claim.claim_id} has no non-gold hard negative")


def _require_train(claims):
    bad = sorted({claim.split for claim in claims if claim.split != "train"})
    if bad:
        raise ValueError(f"Fine-tuning examples only accept the train split, found {bad}")


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())
