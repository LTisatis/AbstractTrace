import pytest

from abstract_review.scoring import score_predictions


def prediction(sample_id, claim_id, verdict, evidence_ids=(), *, status="completed"):
    return {
        "sample_id": sample_id,
        "claim_id": claim_id,
        "review_status": status,
        "verdict": verdict,
        "evidence_ids": list(evidence_ids),
    }


def label(sample_id, claim_id, verdict, groups=()):
    return {
        "sample_id": sample_id,
        "claim_id": claim_id,
        "verdict": verdict,
        "accepted_evidence_groups": [list(group) for group in groups],
    }


def score(predictions, labels):
    return score_predictions(
        {"schema_version": 1, "predictions": predictions},
        {"schema_version": 1, "labels": labels},
    )


def test_scores_alternative_evidence_groups_missing_and_incomplete_predictions():
    result = score(
        [
            prediction("s1", "c1", "supported", ("e3", "e1", "e4")),
            prediction("s2", "c1", "supported", ("e5",)),
            prediction("s2", "c2", None, ("e6",), status="pending"),
            prediction("s2", "c4", "insufficient"),
        ],
        [
            label("s1", "c1", "supported", (("e1", "e2"), ("e3", "e4"))),
            label("s2", "c1", "refuted", (("e5",),)),
            label("s2", "c2", "refuted", (("e6", "e7"),)),
            label("s2", "c3", "supported", (("e8",),)),
            label("s2", "c4", "insufficient"),
        ],
    )

    assert result["accuracy"] == {"numerator": 2, "denominator": 5, "value": 0.4}
    assert result["completed_coverage"] == {"numerator": 3, "denominator": 5, "value": 0.6}
    assert result["evidence_group_recall"] == {"numerator": 2, "denominator": 4, "value": 0.5}
    assert result["joint_accuracy"] == {"numerator": 1, "denominator": 4, "value": 0.25}
    assert [row["label_correct"] for row in result["cases"]] == [True, False, False, False, True]
    assert [row["evidence_group_hit"] for row in result["cases"]] == [True, True, False, False, None]
    assert result["cases"][2]["review_status"] == "pending"
    assert result["cases"][3]["review_status"] is None


def test_scores_empty_manifests_with_null_ratios():
    result = score([], [])
    assert result["cases"] == []
    for metric in ("accuracy", "completed_coverage", "evidence_group_recall", "joint_accuracy"):
        assert result[metric] == {"numerator": 0, "denominator": 0, "value": None}


@pytest.mark.parametrize(
    ("predictions", "labels", "reason"),
    [
        ([prediction("s", "c", "invalid")], [label("s", "c", "refuted")], "verdict"),
        ([prediction("s", "c", "refuted", status="pending")], [], "must be null"),
        ([prediction("s", "c", None)], [], "completed prediction"),
        ([prediction("s", "c", None, status="unknown")], [], "review_status"),
        ([prediction("s", "c", "refuted")] * 2, [], "repeats"),
        ([], [label("s", "c", "refuted")] * 2, "repeats"),
        ([prediction("stale", "c", "refuted")], [label("s", "c", "refuted")], "no gold label"),
        ([], [label("s", "c", "refuted", ((),))], "empty group"),
        ([prediction(" s", "c", "supported")], [], "sample_id"),
        ([dict(prediction("s", "c", "supported"), gold="refuted")], [], "exactly"),
    ],
)
def test_rejects_invalid_or_stale_rows(predictions, labels, reason):
    with pytest.raises(ValueError, match=reason):
        score(predictions, labels)


def test_rejects_non_integer_manifest_version():
    with pytest.raises(ValueError, match="schema_version"):
        score_predictions(
            {"schema_version": True, "predictions": []},
            {"schema_version": 1, "labels": []},
        )
