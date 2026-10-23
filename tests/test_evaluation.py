from dataclasses import replace
import pytest

from conftest import prediction
from evidence_rag.data import Claim, Evidence
from evidence_rag.evaluation import diagnostic_metrics, export_official, joint_metric, official_metrics, retrieval_metrics
from evidence_rag.views import delete_view


@pytest.mark.parametrize("selected,label,expected", [([11], "CONTRADICT", [2/3, 2/3, 1/3, 1/3]),
                                                   ([0], "CONTRADICT", [2/3, 0, 0, 0]),
                                                   ([11], "SUPPORT", [0, 0, 1/3, 0])])
def test_official_263_hand_calculation(dataset, selected, label, expected):
    claim = Claim(263, "synthetic", (100, 200), {100: Evidence("CONTRADICT", ((0, 1), (11,))),
                                               200: Evidence("CONTRADICT", ((7, 9),))}, "dev", True)
    rows = [{"id": 263, "evidence": {"100": {"label": label, "sentences": selected}}}]
    metrics = official_metrics(rows, {263: claim}, dataset.corpus)
    assert [m["f1"] for m in metrics.values()] == pytest.approx(expected)


def test_full_evidence_and_empty_predictions(dataset):
    claim = Claim(263, "synthetic", (100, 200), {100: Evidence("CONTRADICT", ((0, 1), (11,))),
                                               200: Evidence("CONTRADICT", ((7, 9),))}, "dev", True)
    result = {"id": 263, "evidence": {"100": {"label": "CONTRADICT", "sentences": [0, 1, 11]},
                                      "200": {"label": "CONTRADICT", "sentences": [7, 9]}}}
    assert all(m["f1"] == 1 for m in official_metrics([result], {263: claim}, dataset.corpus).values())
    assert all(m["f1"] == 0 for m in official_metrics([{"id": 263, "evidence": {}}], {263: claim}, dataset.corpus).values())


def test_first_three_vs_all_sentences(dataset):
    claim = dataset.claims["train"][1]
    rows = [{"id": 1, "evidence": {"100": {"label": "CONTRADICT", "sentences": [3, 4, 5, 11]}}}]
    scores = official_metrics(rows, {1: claim}, dataset.corpus)
    assert scores["abstract_rationalized"]["f1"] == 0
    assert scores["sentence_selection"]["f1"] > 0


def test_missing_duplicate_and_unlabeled_predictions_fail(dataset):
    with pytest.raises(ValueError, match="exactly once"):
        official_metrics([], dataset.claims["train"], dataset.corpus)
    with pytest.raises(ValueError):
        official_metrics([{"id": 1, "evidence": {}}] * 2, dataset.claims["train"], dataset.corpus)
    with pytest.raises(ValueError, match="unlabeled"):
        official_metrics([{"id": 3, "evidence": {}}], dataset.claims["test"], dataset.corpus)


def test_export_keeps_model_sentence_order(dataset, full):
    rows = export_official([prediction(full, ids=[11, 0, 1])], dataset.claims["train"], dataset.corpus)
    assert rows[0]["evidence"]["100"]["sentences"] == [11, 0, 1]
    with pytest.raises(ValueError, match="missing"):
        export_official([], dataset.claims["train"], dataset.corpus)


def test_invalid_references_do_not_get_evidence_credit(full):
    for ids in ([11, 999], [True], [11, 11]):
        result = diagnostic_metrics([full], [prediction(full, ids=ids)])
        assert result["accuracy"] == 0 and result["failure_rate"] == 1


def test_zero_coverage_undefined_and_unreachable_points(full):
    p = prediction(full, status="invalid_output")
    metrics = diagnostic_metrics([full], [p])
    assert metrics["n"] == 1 and metrics["risk"] is None and metrics["coverage"] == 0
    assert metrics["fixed_coverage"][-1]["reachable"] is False
    assert metrics["fixed_coverage"][0]["risk"] is None


def test_noinfo_is_valid_but_not_support_coverage(full):
    view = replace(delete_view(full, {0, 11}, "Break"), verification="VERIFIED")
    result = diagnostic_metrics([view], [prediction(view)])
    assert result["accuracy"] == 1 and result["coverage"] == 0 and result["failure_rate"] == 0


def test_joint_four_conditions_and_shortcut_failure(full):
    views = [full] + [replace(delete_view(full, removed, kind), verification="VERIFIED")
                      for kind, removed in (("Keep", {0}), ("Break", {0, 11}), ("Neutral", {2, 3}))]
    quad = {v.kind: v.view_id for v in views}
    rows = [prediction(v) for v in views]
    assert joint_metric(views, rows, [quad])["joint"] == 1
    broken = next(v for v in views if v.kind == "Break")
    wrong = [prediction(v, relation="CONTRADICT", ids=[1]) if v == broken else prediction(v) for v in views]
    assert joint_metric(views, wrong, [quad])["joint"] == 0
    assert joint_metric([], [], [])["joint"] is None


def test_retrieval_metrics_do_not_reward_no_gold(dataset):
    cache = {"1": [{"doc_id": 100, "rank": 1, "score": 0.5}]}
    assert retrieval_metrics(cache, dataset.claims["train"])["document_recall"] == 1
    with pytest.raises(ValueError):
        retrieval_metrics({**cache, "01": []}, dataset.claims["train"])
