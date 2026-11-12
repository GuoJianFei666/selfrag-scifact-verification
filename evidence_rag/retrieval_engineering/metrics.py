"""Macro document metrics over SciFact evidence abstracts."""
import math
import time


def macro_metrics(rankings, ks=(1, 3, 5, 10, 20), mrr_k=10, ndcg_k=10):
    """Average Recall@k, MRR and nDCG over rankings of (doc_ids, gold_ids)."""
    if not rankings:
        raise ValueError("No eligible retrieval queries")
    recalls = {str(k): [] for k in ks}
    mrrs = []
    ndcgs = []
    gold_documents = 0
    for ranked, gold in rankings:
        if not gold:
            raise ValueError("Eligible rankings require gold documents")
        if len(ranked) != len(set(ranked)):
            raise ValueError("Ranking contains duplicate documents")
        gold_documents += len(gold)
        for k in ks:
            recalls[str(k)].append(len(set(ranked[:k]) & gold) / len(gold))
        first = next((index + 1 for index, doc_id in enumerate(ranked[:mrr_k]) if doc_id in gold), None)
        mrrs.append(0.0 if first is None else 1.0 / first)
        ndcgs.append(_ndcg(ranked, gold, ndcg_k))
    count = len(rankings)
    return {
        "queries": count,
        "gold_documents": gold_documents,
        "recall": {key: sum(values) / count for key, values in recalls.items()},
        "mrr_at_10": sum(mrrs) / count,
        "ndcg_at_10": sum(ndcgs) / count,
    }


def _ndcg(ranked, gold, k):
    discount = 0.0
    for index, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in gold:
            discount += 1.0 / math.log2(index + 1)
    ideal_hits = min(k, len(gold))
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return discount / ideal if ideal else 0.0


def score_claims(claims, search, ranking_depth, ks=(1, 3, 5, 10, 20), mrr_k=10, ndcg_k=10, warmup=True):
    """Time every claim. Average metrics only over claims that have gold evidence."""
    if type(ranking_depth) is not int or ranking_depth < 1:
        raise ValueError("ranking_depth must be a positive integer")
    ordered = sorted(claims.values(), key=lambda claim: claim.claim_id)
    if not ordered:
        raise ValueError("Claim set is empty")
    if warmup:
        search(ordered[0].text)
    rankings = []
    samples = []
    for claim in ordered:
        started = time.perf_counter()
        ranked = list(search(claim.text))[:ranking_depth]
        samples.append(time.perf_counter() - started)
        doc_ids = [row["doc_id"] for row in ranked]
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("Search returned duplicate documents")
        if claim.evidence:
            rankings.append((doc_ids, set(claim.evidence)))
    metrics = macro_metrics(rankings, ks=ks, mrr_k=mrr_k, ndcg_k=ndcg_k)
    metrics["latency_seconds"] = latency_summary(samples)
    metrics["queries_timed"] = len(ordered)
    metrics["eligible_claims"] = len(rankings)
    return metrics


def latency_summary(samples):
    if not samples or any(not math.isfinite(value) or value < 0 for value in samples):
        raise ValueError("Latency samples must be nonnegative finite seconds")
    ordered = sorted(samples)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        median = ordered[middle]
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2
    return {"mean": sum(ordered) / count, "median": median, "n": count}
