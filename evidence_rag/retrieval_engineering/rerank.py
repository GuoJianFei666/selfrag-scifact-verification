"""Reorder a first-stage candidate list with cross-encoder scores."""


def order_by_scores(doc_ids, scores, depth):
    if len(doc_ids) != len(scores):
        raise ValueError("Reranker scores must cover every candidate")
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("Rerank candidates contain duplicate documents")
    if type(depth) is not int or depth < 1:
        raise ValueError("Rerank depth must be a positive integer")
    ordered = sorted(zip(doc_ids, scores), key=lambda item: (-float(item[1]), item[0]))
    return [
        {"doc_id": doc_id, "rank": rank + 1, "score": float(score)}
        for rank, (doc_id, score) in enumerate(ordered[:depth])
    ]
