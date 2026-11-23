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


def test_ranked_recall_counts_complete_groups_in_order_and_keeps_citations_separate():
    result = score(
        [
            dict(prediction("s1", "c", "supported", ("z",)), candidate_ids=["x", "y", "z"]),
            dict(prediction("s2", "c", None, status="pending"), candidate_ids=["y"], stop_reason="key_source_unreadable"),
        ],
        [
            label("s1", "c", "supported", (("x", "z"), ("y", "z"))),
            label("s2", "c", "supported", (("y",),)),
            label("s3", "c", "refuted", (("w",),)),
            label("s4", "c", "insufficient"),
        ],
    )

    assert result["evidence_group_recall_at_k"]["1"] == {"numerator": 1, "denominator": 3, "value": 1 / 3}
    assert result["evidence_group_recall_at_k"]["3"] == {"numerator": 2, "denominator": 3, "value": 2 / 3}
    assert result["ranked_evidence_coverage"]["numerator"] == 2
    assert result["joint_accuracy"]["numerator"] == 0
    assert [row["phase"] for row in result["errors"]] == [
        "judgment", "parsing", "missing_prediction", "missing_prediction",
    ]


def test_legacy_citations_are_not_assumed_to_be_ranked_retrieval():
    result = score(
        [prediction("s1", "c", "supported", ("x",))],
        [label("s1", "c", "supported", (("x",),))],
    )

    assert result["evidence_group_recall"]["value"] == 1
    assert result["ranked_evidence_coverage"]["value"] == 0
    assert result["evidence_group_recall_at_k"]["1"] is None


def test_extraction_matches_once_on_sentence_interval_and_normalized_proposition():
    def anchor(start, end, proposition="Fixed sample size", sentence_id="s1"):
        return {"sentence_id": sentence_id, "start": start, "end": end, "proposition": proposition}

    predictions = {"schema_version": 1, "predictions": [
        dict(prediction("s", "wide", "supported"), claim_anchor=anchor(0, 18, "fixed  sample size")),
        dict(prediction("s", "first", "supported"), claim_anchor=anchor(1, 6)),
        dict(prediction("s", "different", "supported"), claim_anchor=anchor(0, 6, "Variable sample size")),
        dict(prediction("empty", "extra", "supported"), claim_anchor=anchor(0, 6)),
    ]}
    gold = {"schema_version": 1, "extraction_samples": ["s", "empty"], "labels": [
        dict(label("s", "g1", "supported"), claim_anchor=anchor(1, 7), split="dev"),
        dict(label("s", "g2", "supported"), claim_anchor=anchor(10, 15), split="dev"),
        dict(label("s", "g3", "refuted"), claim_anchor=anchor(10, 15, sentence_id="s2"), split="dev"),
    ]}
    result = score_predictions(predictions, gold, rank_cutoffs=(2,))

    assert [item["predicted_claim_id"] for item in result["cases"]] == ["first", "wide", None]
    assert result["extraction"]["precision"] == {"numerator": 2, "denominator": 4, "value": 0.5}
    assert result["extraction"]["recall"] == {"numerator": 2, "denominator": 3, "value": 2 / 3}
    assert result["extraction"]["missing"] == [{"sample_id": "s", "claim_id": "g3"}]
    assert [row["claim_id"] for row in result["extraction"]["extra"]] == ["extra", "different"]
    assert result["classification_by_split"]["dev"]["classification_by_label"]["refuted"]["gold_count"] == 1
    assert result["errors"][0]["phase"] == "extraction"
