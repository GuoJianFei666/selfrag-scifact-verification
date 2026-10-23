from dataclasses import replace
import pytest

from evidence_rag.common import digest, read_json, write_jsonl
from evidence_rag.data import Claim, Document, Dataset, audit_sources, normalize_claim
from evidence_rag.views import View, apply_reviews, delete_view, full_view, propose_views, training_target, validate_intervention


def test_duplicate_document_and_bad_rationales(dataset):
    raw = {"id": 4, "claim": "claim", "cited_doc_ids": [100], "evidence": {"100": [{"label": "SUPPORT", "sentences": [True]}]}}
    with pytest.raises(ValueError):
        Claim.parse(raw, dataset.corpus, "train")
    for ids in ([12], [], [1, 1]):
        raw["evidence"]["100"][0]["sentences"] = ids
        with pytest.raises(ValueError):
            Claim.parse(raw, dataset.corpus, "train")


def test_missing_gold_does_not_become_noinfo(dataset):
    with pytest.raises(KeyError):
        Claim.parse({"id": 9, "claim": "claim", "cited_doc_ids": [100]}, dataset.corpus, "train")
    assert not dataset.claims["test"][3].labeled
    with pytest.raises(ValueError):
        dataset.gold_relation(dataset.claims["test"][3], 100)


def test_duplicate_cited_is_one_reviewed_pair(dataset):
    claim = Claim.parse({"id": 1245, "claim": "synthetic", "evidence": {}, "cited_doc_ids": [100, 100]}, dataset.corpus, "dev")
    assert claim.cited_doc_ids == (100,)
    assert dataset.gold_relation(claim, 100) == "NOINFO"


def test_unreviewed_retrieved_document_cannot_be_noinfo_gold(dataset):
    claim = dataset.claims["dev"][2]
    with pytest.raises(ValueError):
        full_view(claim, dataset.corpus[200], labeled=True)
    assert full_view(claim, dataset.corpus[200]).relation is None


def test_manifest_tamper_detected(dataset):
    path = dataset.directory / "claims_train.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        Dataset(dataset.directory)


def test_sources_only_normalize_unicode_and_space(dataset, tmp_path):
    assert normalize_claim("not   effective") == "not effective"
    assert normalize_claim("not effective") != normalize_claim("effective")
    rows = [{"s2orc_id": str(i), "citance": "citance " + str(i), "claims": [{"text": c.text, "is_negation": False}]}
            for i, claims in enumerate(dataset.claims.values()) for c in claims.values()]
    source = tmp_path / "sources.jsonl"
    write_jsonl(source, rows)
    audit = audit_sources(dataset, source, tmp_path / "audit.json", tmp_path / "split.json")
    assert len(audit["mapping"]) == 3
    original = read_json(tmp_path / "split.json")
    audit_sources(dataset, source, tmp_path / "audit2.json", tmp_path / "split2.json")
    assert original == read_json(tmp_path / "split2.json")
    rows[0]["claims"][0]["claim_text"] = "conflicting text"
    write_jsonl(source, rows)
    with pytest.raises(ValueError, match="conflicting"):
        audit_sources(dataset, source, tmp_path / "bad.json")


def test_unmatched_source_is_reported_not_randomly_grouped(dataset, tmp_path):
    path = tmp_path / "sources.jsonl"
    write_jsonl(path, [{"s2orc_id": 1, "citance": "x", "claims": [{"text": "unknown", "is_negation": False}]}])
    with pytest.raises(ValueError, match="unresolved"):
        audit_sources(dataset, path, tmp_path / "audit.json")
    assert len(read_json(tmp_path / "audit.json")["unresolved"]) == 3


def test_alternative_evidence_delete_cases(full):
    for removed in ({0}, {0, 1}, {11}):
        keep = delete_view(full, removed, "Keep")
        validate_intervention(full, keep)
        assert keep.relation == "CONTRADICT" and keep.rationales
    broken = delete_view(full, {0, 11}, "Break")
    validate_intervention(full, broken)
    assert broken.verification == "UNVERIFIED"
    with pytest.raises(ValueError):
        training_target(broken)


def test_deterministic_candidates_and_no_single_rationale_keep(full):
    views = propose_views(full)
    assert [v.asdict() for v in views] == [v.asdict() for v in propose_views(full)]
    assert {v.kind for v in views} == {"Keep", "Break", "Neutral"}
    single = replace(full, rationales=((0, 1),))
    assert "Keep" not in {v.kind for v in propose_views(single)}


def test_views_cannot_reorder_or_cross_split(full):
    keep = delete_view(full, {0}, "Keep")
    with pytest.raises(ValueError, match="order"):
        validate_intervention(full, replace(keep, sentences=keep.sentences[::-1], original_ids=keep.original_ids[::-1]))
    with pytest.raises(ValueError, match="inherit"):
        validate_intervention(full, replace(keep, split="dev"))
    raw = full.asdict()
    raw["rationales"] = [[True]]
    with pytest.raises(ValueError):
        View.parse(raw)


def test_two_review_rounds_required_and_content_bound(full):
    view = delete_view(full, {0, 11}, "Break")
    row = {"view_id": view.view_id, "content_hash": digest(view.asdict()), "round": 1,
           "decision": "VERIFIED", "relation": "NOINFO", "reviewer": "test reviewer", "note": "synthetic review fixture"}
    assert apply_reviews([view], [row])[0].verification == "UNVERIFIED"
    reviewed = apply_reviews([view], [row, {**row, "round": 2}])[0]
    assert reviewed.verification == "VERIFIED"
    assert training_target(reviewed) == {"relation": "NOINFO", "sentence_ids": []}
    with pytest.raises(ValueError, match="changed"):
        apply_reviews([view], [{**row, "content_hash": "bad"}])


def test_target_selects_one_full_set_and_maps_local_ids(full):
    assert training_target(full)["sentence_ids"] == [11]
    keep = replace(delete_view(full, {0}, "Keep"), verification="VERIFIED")
    assert training_target(keep)["sentence_ids"] == [10]
