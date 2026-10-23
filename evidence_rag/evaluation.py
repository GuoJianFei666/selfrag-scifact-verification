"""Official-style SciFact counts plus explicit completeness and selective-risk checks."""
from collections import Counter
import math

from .data import RELATIONS


def f1(correct, retrieved, relevant):
    precision = correct / retrieved if retrieved else 0.0
    recall = correct / relevant if relevant else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "correct": correct, "retrieved": retrieved, "relevant": relevant}


def validate_official(predictions, claims, corpus):
    ids = [p["id"] for p in predictions]
    if any(type(i) is not int for i in ids) or len(ids) != len(set(ids)) or set(ids) != set(claims):
        raise ValueError("Official predictions must contain every target claim exactly once")
    for row in predictions:
        if set(row) != {"id", "evidence"} or not isinstance(row["evidence"], dict):
            raise ValueError("Invalid official prediction schema")
        for key, value in row["evidence"].items():
            if not key.isdigit() or str(int(key)) != key or int(key) not in corpus:
                raise ValueError("Unknown predicted document")
            if set(value) != {"label", "sentences"} or value["label"] not in ("SUPPORT", "CONTRADICT"):
                raise ValueError("Official export allows only SUPPORT and CONTRADICT")
            ids = value["sentences"]
            if not isinstance(ids, list) or not ids or any(type(i) is not int or not 0 <= i < len(corpus[int(key)].abstract) for i in ids) or len(ids) != len(set(ids)):
                raise ValueError("Invalid predicted sentence list")


def official_metrics(predictions, claims, corpus):
    validate_official(predictions, claims, corpus)
    if any(not claim.labeled for claim in claims.values()):
        raise ValueError("Cannot score an unlabeled test split")
    counts = Counter()
    for row in predictions:
        gold = claims[row["id"]]
        counts["abstract_relevant"] += len(gold.evidence)
        for evidence in gold.evidence.values():
            members = [i for rationale in evidence.rationales for i in rationale]
            if len(members) != len(set(members)):
                raise ValueError("Fixed upstream sentence scorer assumes disjoint rationale sets")
            counts["sentence_relevant"] += len(members)
        for key, predicted in row["evidence"].items():
            did, selected = int(key), predicted["sentences"]
            counts["abstract_retrieved"] += 1
            counts["sentence_retrieved"] += len(selected)
            evidence = gold.evidence.get(did)
            if evidence is None:
                continue
            same_label = predicted["label"] == evidence.relation
            if same_label:
                counts["abstract_label"] += 1
                counts["abstract_rationale"] += int(any(set(r) <= set(selected[:3]) for r in evidence.rationales))
            correct = sum(len(r) for r in evidence.rationales if set(r) <= set(selected))
            counts["sentence_selection"] += correct
            counts["sentence_label"] += correct * int(same_label)
    return {"abstract_label_only": f1(counts["abstract_label"], counts["abstract_retrieved"], counts["abstract_relevant"]),
            "abstract_rationalized": f1(counts["abstract_rationale"], counts["abstract_retrieved"], counts["abstract_relevant"]),
            "sentence_selection": f1(counts["sentence_selection"], counts["sentence_retrieved"], counts["sentence_relevant"]),
            "sentence_label": f1(counts["sentence_label"], counts["sentence_retrieved"], counts["sentence_relevant"])}


def export_official(results, target_claims, corpus):
    if {r["claim_id"] for r in results} != set(target_claims):
        raise ValueError("Internal results are missing or adding target claims")
    output = {cid: {"id": cid, "evidence": {}} for cid in target_claims}
    seen = set()
    for result in results:
        cid, did = result["claim_id"], result["doc_id"]
        if cid not in output or (cid, did) in seen:
            raise ValueError("Unexpected or duplicate claim-document result")
        seen.add((cid, did))
        relation = result.get("relation_candidate")
        if result["status"] in ("accepted", "review_required") and relation in ("SUPPORT", "CONTRADICT"):
            output[cid]["evidence"][str(did)] = {"label": relation, "sentences": result["original_sentence_ids"]}
    rows = [output[cid] for cid in sorted(output)]
    validate_official(rows, target_claims, corpus)
    return rows


def valid_result(result, view):
    if any(result.get(k) != getattr(view, k) for k in ("view_id", "claim_id", "doc_id")):
        raise ValueError("Prediction is bound to a different claim, document or view")
    if result.get("status") not in ("accepted", "review_required", "invalid_output", "execution_failed"):
        raise ValueError("Unknown result status")
    if result["status"] not in ("accepted", "review_required"):
        return False
    relation, ids = result.get("relation_candidate"), result.get("original_sentence_ids")
    if relation not in RELATIONS or not isinstance(ids, list) or any(type(i) is not int or i not in view.original_ids for i in ids) or len(ids) != len(set(ids)):
        return False
    if (relation == "NOINFO") != (len(ids) == 0):
        return False
    local = result.get("view_sentence_ids")
    if local is not None and (not isinstance(local, list) or any(type(i) is not int or not 0 <= i < len(view.sentences) for i in local) or [view.original_ids[i] for i in local] != ids):
        return False
    sentences = result.get("evidence_sentences")
    if sentences is not None and sentences != [view.sentences[view.original_ids.index(i)] for i in ids]:
        return False
    return True


def complete_evidence(result, view):
    return (valid_result(result, view) and result.get("relation_candidate") == view.relation
            and any(set(r) <= set(result.get("original_sentence_ids", [])) for r in view.rationales))


def diagnostic_metrics(views, predictions, coverages=(0.0, 0.25, 0.5, 0.75, 1.0)):
    if any(v.verification != "VERIFIED" or v.relation not in RELATIONS for v in views):
        raise ValueError("Diagnostics require a frozen labeled and reviewed set")
    by_id = {p["view_id"]: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {v.view_id for v in views}:
        raise ValueError("Diagnostic predictions must cover the full target set")
    confusion = Counter()
    candidates, accepted = [], []
    invalid = 0
    for view in views:
        result = by_id[view.view_id]
        valid = valid_result(result, view)
        predicted = result.get("relation_candidate") if valid else "INVALID"
        confusion[(view.relation, predicted)] += 1
        invalid += int(not valid)
        if valid and predicted in ("SUPPORT", "CONTRADICT"):
            confidence = result.get("confidence")
            if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Valid conclusion is missing a finite candidate probability")
            item = (view, result, not complete_evidence(result, view))
            candidates.append(item)
            if result["status"] == "accepted":
                accepted.append(item)
    n = len(views)
    per_class = {}
    for label in RELATIONS:
        correct = confusion[(label, label)]
        per_class[label] = f1(correct, sum(v for (g, p), v in confusion.items() if p == label),
                              sum(v for (g, p), v in confusion.items() if g == label))
    candidates.sort(key=lambda item: (-item[1]["confidence"], item[0].claim_id, item[0].doc_id, item[0].view_id))
    curve = []
    for target in coverages:
        if not 0 <= target <= 1:
            raise ValueError("Coverage target must be in [0,1]")
        count = math.floor(target * n)
        reachable = len(candidates) >= count
        chosen = candidates[:count] if reachable else []
        curve.append({"target": target, "reachable": reachable, "accepted": count if reachable else None,
                      "coverage": count / n if n and reachable else None,
                      "risk": sum(item[2] for item in chosen) / count if count and reachable else None})
    return {"n": n, "accuracy": sum(confusion[(r, r)] for r in RELATIONS) / n if n else None,
            "macro_f1": sum(per_class[r]["f1"] for r in RELATIONS) / 3 if n else None,
            "failure_rate": invalid / n if n else None, "accepted": len(accepted),
            "status_counts": dict(Counter(p["status"] for p in predictions)),
            "noinfo_candidates": sum(confusion[(g, "NOINFO")] for g in RELATIONS),
            "coverage": len(accepted) / n if n else None,
            "risk": sum(item[2] for item in accepted) / len(accepted) if accepted else None,
            "per_class": per_class, "fixed_coverage": curve}


def joint_metric(views, predictions, frozen_quadruples):
    from .views import validate_intervention
    by_id = {v.view_id: v for v in views}
    pred = {p["view_id"]: p for p in predictions}
    if len(by_id) != len(views) or len(pred) != len(predictions) or set(pred) != set(by_id):
        raise ValueError("Joint evaluation requires complete predictions")
    correct = 0
    used = set()
    for quadruple in frozen_quadruples:
        if set(quadruple) != {"Full", "Keep", "Break", "Neutral"}:
            raise ValueError("A quadruple must contain all four conditions")
        selected = {kind: by_id[vid] for kind, vid in quadruple.items()}
        key = (selected["Full"].claim_id, selected["Full"].doc_id)
        if key in used or any((v.claim_id, v.doc_id) != key or v.kind != k or v.verification != "VERIFIED" for k, v in selected.items()):
            raise ValueError("Invalid, repeated or unreviewed quadruple")
        used.add(key)
        full = selected["Full"]
        if full.relation not in ("SUPPORT", "CONTRADICT") or not full.rationales:
            raise ValueError("Joint Full view must have positive evidence")
        for kind in ("Keep", "Break", "Neutral"):
            validate_intervention(full, selected[kind])
        good = all(complete_evidence(pred[v.view_id], v) for k, v in selected.items() if k != "Break")
        broken = pred[selected["Break"].view_id]
        good = good and valid_result(broken, selected["Break"]) and broken.get("relation_candidate") == "NOINFO"
        correct += int(good)
    return {"correct": correct, "n": len(used), "joint": correct / len(used) if used else None}


def retrieval_metrics(hits, claims):
    if any(not isinstance(k, str) or not k.isdigit() or str(int(k)) != k for k in hits) or set(map(int, hits)) != set(claims):
        raise ValueError("Retrieval cache must include every target claim")
    for rows in hits.values():
        if len({r["doc_id"] for r in rows}) != len(rows) or [r["rank"] for r in rows] != list(range(1, len(rows) + 1)):
            raise ValueError("Invalid retrieval ranks or duplicate documents")
        if any(type(r["doc_id"]) is not int or not math.isfinite(r["score"]) for r in rows):
            raise ValueError("Invalid retrieval document or score")
    eligible = [claim for claim in claims.values() if claim.evidence]
    if not eligible:
        return {"eligible_claims": 0, "document_recall": None, "at_least_one": None}
    found = total = one = 0
    for claim in eligible:
        ids = {r["doc_id"] for r in hits[str(claim.claim_id)]}
        gold = set(claim.evidence)
        found += len(ids & gold)
        total += len(gold)
        one += int(bool(ids & gold))
    return {"eligible_claims": len(eligible), "gold_documents": total,
            "document_recall": found / total, "at_least_one": one / len(eligible)}
