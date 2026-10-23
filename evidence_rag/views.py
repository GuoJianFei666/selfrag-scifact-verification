"""Evidence views keep reversible sentence IDs and require semantic review."""
from dataclasses import asdict, dataclass, replace
from itertools import combinations

from pathlib import Path

from .common import digest, file_hash, integer, read_jsonl, write_json, write_jsonl
from .data import RELATIONS


@dataclass(frozen=True)
class View:
    view_id: str
    claim_id: int
    doc_id: int
    claim: str
    title: str
    sentences: tuple[str, ...]
    original_ids: tuple[int, ...]
    split: str
    source_group: str
    kind: str
    relation: str | None
    rationales: tuple[tuple[int, ...], ...]
    verification: str
    review_note: str = ""

    def asdict(self):
        return asdict(self)

    @classmethod
    def parse(cls, value):
        v = dict(value)
        for key in ("sentences", "original_ids"):
            v[key] = tuple(v[key])
        v["rationales"] = tuple(tuple(r) for r in v["rationales"])
        view = cls(**v)
        integer(view.claim_id, "claim_id")
        integer(view.doc_id, "doc_id")
        if len(view.sentences) != len(view.original_ids) or len(set(view.original_ids)) != len(view.original_ids):
            raise ValueError("View mapping must be one-to-one")
        if any(type(i) is not int or i < 0 for i in view.original_ids):
            raise ValueError("Invalid original sentence ID")
        if view.relation is not None and view.relation not in RELATIONS:
            raise ValueError("Invalid view relation")
        if view.kind not in ("Full", "Keep", "Break", "Neutral"):
            raise ValueError("Invalid view kind")
        if view.verification not in ("VERIFIED", "UNVERIFIED", "REJECTED"):
            raise ValueError("Invalid verification state")
        if any(not r or any(type(i) is not int for i in r) or len(set(r)) != len(r) or not set(r) <= set(view.original_ids) for r in view.rationales):
            raise ValueError("View rationale must be fully visible")
        if view.relation == "NOINFO" and view.rationales:
            raise ValueError("NOINFO cannot contain positive rationale sets")
        if view.verification == "VERIFIED" and (view.relation is None or (view.relation != "NOINFO" and not view.rationales)):
            raise ValueError("Verified positive views require rationale sets and a relation")
        return view


def full_view(claim, document, split=None, group="", labeled=False):
    evidence = claim.evidence.get(document.doc_id) if labeled else None
    if labeled and (not claim.labeled or document.doc_id not in claim.cited_doc_ids):
        raise ValueError("Cannot label an unreviewed retrieved document")
    relation = evidence.relation if evidence else ("NOINFO" if labeled else None)
    view_id = digest({"claim_id": claim.claim_id, "claim_text": claim.text,
                      "document": document.content_hash, "kind": "Full"})[:24]
    return View(view_id, claim.claim_id, document.doc_id, claim.text, document.title, document.abstract,
                tuple(range(len(document.abstract))), split or claim.split, group, "Full", relation,
                evidence.rationales if evidence else (), "VERIFIED" if labeled else "UNVERIFIED")


def delete_view(full, removed, kind):
    visible = [(i, s) for i, s in zip(full.original_ids, full.sentences) if i not in removed]
    rationales = tuple(r for r in full.rationales if set(r) <= {i for i, _ in visible})
    relation = "NOINFO" if kind == "Break" else full.relation
    vid = digest({"parent": full.view_id, "kind": kind, "removed": sorted(removed)})[:24]
    return replace(full, view_id=vid, sentences=tuple(s for _, s in visible), original_ids=tuple(i for i, _ in visible),
                   kind=kind, relation=relation, rationales=rationales, verification="UNVERIFIED", review_note="")


def validate_intervention(full, derived):
    if full.kind != "Full" or derived.kind not in ("Keep", "Break", "Neutral"):
        raise ValueError("Invalid intervention kinds")
    keys = ("claim_id", "doc_id", "claim", "title", "split", "source_group")
    if any(getattr(full, k) != getattr(derived, k) for k in keys):
        raise ValueError("Interventions must inherit parent, document, source group and split")
    source = dict(zip(full.original_ids, full.sentences))
    if any(source.get(i) != s for i, s in zip(derived.original_ids, derived.sentences)):
        raise ValueError("View contains replaced or foreign document sentences")
    visible = set(derived.original_ids)
    if tuple(i for i in full.original_ids if i in visible) != derived.original_ids:
        raise ValueError("Deletion views must preserve the original sentence order")
    intact = [set(r) <= visible for r in full.rationales]
    removed = set(full.original_ids) - visible
    expected = tuple(r for r in full.rationales if set(r) <= visible)
    if derived.rationales != expected or not removed:
        raise ValueError("Intervention must delete text and preserve its rationale mapping")
    if derived.kind == "Keep" and not (any(intact) and not all(intact)):
        raise ValueError("Keep must break one rationale and retain another")
    if derived.kind == "Break" and (any(intact) or derived.relation != "NOINFO"):
        raise ValueError("Break must hit every annotated rationale")
    if derived.kind == "Neutral" and (not all(intact) or removed & set().union(*map(set, full.rationales))):
        raise ValueError("Neutral may delete only non-evidence sentences")
    if derived.kind != "Break" and derived.relation != full.relation:
        raise ValueError("Keep and Neutral inherit the parent relation after review")


def propose_views(full):
    if full.verification != "VERIFIED" or full.relation == "NOINFO" or not full.rationales:
        return []
    results = []
    universe = sorted(set().union(*map(set, full.rationales)))
    # Branch on one unhit rationale: small exact hitting-set search, no 2^abstract enumeration.
    best = None
    visits = 0
    def search(chosen):
        nonlocal best, visits
        visits += 1
        if visits > 100_000:
            raise ValueError("Hitting-set search budget exceeded")
        if best is not None and len(chosen) >= len(best):
            return
        remaining = [r for r in full.rationales if not chosen & set(r)]
        if not remaining:
            best = chosen
            return
        for sid in sorted(min(remaining, key=lambda r: (len(r), r))):
            search(chosen | {sid})
    search(set())
    if best:
        results.append(delete_view(full, best, "Break"))
    for sid in universe:
        candidate = delete_view(full, {sid}, "Keep")
        if candidate.rationales and len(candidate.rationales) < len(full.rationales):
            results.append(candidate)
            break
    non_evidence = sorted(set(full.original_ids) - set(universe))
    if best and len(non_evidence) >= len(best):
        lengths = dict(zip(full.original_ids, map(len, full.sentences)))
        target_length = sum(lengths[i] for i in best)
        candidates = combinations(non_evidence, len(best))
        # Abstracts are small; cap combinatorics explicitly rather than silently sample.
        choices = []
        for j, choice in enumerate(candidates):
            if j >= 100_000:
                raise ValueError("Neutral candidate search budget exceeded")
            choices.append(choice)
        removed = min(choices, key=lambda ids: (abs(sum(lengths[i] for i in ids) - target_length), ids))
        results.append(delete_view(full, set(removed), "Neutral"))
    for view in results:
        validate_intervention(full, view)
    return results


def apply_reviews(views, reviews):
    by_id = {v.view_id: v for v in views}
    if len(by_id) != len(views):
        raise ValueError("Duplicate view ID")
    rounds = {}
    for review in reviews:
        vid = review["view_id"]
        round_id = review["round"]
        if vid not in by_id or round_id not in (1, 2) or (vid, round_id) in rounds:
            raise ValueError("Unknown or duplicate reviewed view")
        view = by_id[vid]
        if review["content_hash"] != digest(view.asdict()):
            raise ValueError("View changed since review")
        if review["decision"] not in ("VERIFIED", "REJECTED") or not review["reviewer"].strip() or not review["note"].strip():
            raise ValueError("Review needs a decision, reviewer and semantic note")
        if review["decision"] == "VERIFIED" and review["relation"] != view.relation:
            raise ValueError("Semantic review relation disagrees with proposed label; reject the view")
        rounds[(vid, round_id)] = review
    for vid, view in by_id.items():
        if (vid, 1) in rounds or (vid, 2) in rounds:
            entries = [rounds[(vid, r)] for r in (1, 2) if (vid, r) in rounds]
            decisions = {r["decision"] for r in entries}
            state = next(iter(decisions)) if len(entries) == 2 and len(decisions) == 1 else "UNVERIFIED"
            note = "; ".join(f'round {r["round"]}, {r["reviewer"]}: {r["note"]}' for r in entries)
            by_id[vid] = replace(view, verification=state, review_note=note)
    return list(by_id.values())


def training_target(view):
    if view.verification != "VERIFIED" or view.relation is None:
        raise ValueError("Only verified, labeled views may be used for supervision")
    if view.relation == "NOINFO":
        if view.rationales:
            raise ValueError("NOINFO cannot have positive rationale sets")
        return {"relation": "NOINFO", "sentence_ids": []}
    if not view.rationales:
        raise ValueError("Positive relation requires a complete rationale")
    rationale = min(view.rationales, key=lambda r: (len(r), r))
    local = {original: i for i, original in enumerate(view.original_ids)}
    return {"relation": view.relation, "sentence_ids": [local[i] for i in rationale]}


def load_views(path):
    views = [View.parse(row) for row in read_jsonl(path)]
    if len({v.view_id for v in views}) != len(views):
        raise ValueError("Duplicate view IDs")
    return views


def prepare_evaluation(path, output):
    views = load_views(path)
    if not views or len({v.split for v in views}) != 1 or views[0].split not in ("model_select", "calibration", "dev"):
        raise ValueError("Evaluation preparation needs one nonempty held-out partition")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Evaluation output must be empty")
    fulls = {(v.claim_id, v.doc_id): v for v in views if v.kind == "Full"}
    included, exclusions, groups = [], [], {}
    for v in views:
        if v.verification != "VERIFIED":
            exclusions.append({"view_id": v.view_id, "kind": v.kind, "reason": v.verification})
            continue
        if v.kind != "Full":
            if (v.claim_id, v.doc_id) not in fulls:
                raise ValueError("Evaluation intervention lacks Full parent")
            validate_intervention(fulls[(v.claim_id, v.doc_id)], v)
        variants = groups.setdefault((v.claim_id, v.doc_id), {})
        if v.kind in variants:
            raise ValueError("Only one deterministic evaluation view per condition")
        variants[v.kind] = v.view_id
        included.append(v)
    if not included or any(v.verification != "VERIFIED" for v in fulls.values()):
        raise ValueError("All original cited Full pairs must remain in evaluation")
    quadruples = [group for _, group in sorted(groups.items()) if set(group) == {"Full", "Keep", "Break", "Neutral"}]
    write_jsonl(output / "views.jsonl", [v.asdict() for v in included])
    write_json(output / "quadruples.json", quadruples)
    audit = {"input_sha256": file_hash(path), "partition": views[0].split, "original_full_pairs": len(fulls),
             "included": len(included), "quadruples": len(quadruples), "exclusions": exclusions}
    write_json(output / "exclusion_audit.json", audit)
    return audit
