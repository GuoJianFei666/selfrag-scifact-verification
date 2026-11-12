"""Collapse chunk hits to one row per source document."""


def aggregate_documents(hits, depth):
    if type(depth) is not int or depth < 1:
        raise ValueError("Document depth must be a positive integer")
    best = {}
    for hit in hits:
        doc_id = hit["doc_id"]
        score = hit["score"]
        if type(doc_id) is not int or isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("Chunk hit needs an integer document id and a numeric score")
        if doc_id not in best or score > best[doc_id]:
            best[doc_id] = float(score)
    ordered = sorted(best.items(), key=lambda item: (-item[1], item[0]))
    return [
        {"doc_id": doc_id, "rank": rank + 1, "score": score}
        for rank, (doc_id, score) in enumerate(ordered[:depth])
    ]
