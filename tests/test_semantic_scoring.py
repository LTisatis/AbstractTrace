import copy

import pytest

from abstract_review.scoring import score_predictions, semantic_manifest_sha256


def _anchor(proposition, start=0, end=20, sentence_id="s1"):
    return {"sentence_id": sentence_id, "start": start, "end": end, "proposition": proposition}


def _manifests():
    predictions = {"schema_version": 1, "predictions": [
        {"sample_id": "paper", "claim_id": "p-exact", "verdict": "supported", "review_status": "completed",
         "evidence_ids": ["e1"], "claim_anchor": _anchor("风速上升", 0, 8)},
        {"sample_id": "paper", "claim_id": "p-paraphrase", "verdict": "refuted", "review_status": "completed",
         "evidence_ids": ["e2"], "candidate_ids": ["e2"], "claim_anchor": _anchor("样品重量减少一半", 5, 17)},
        {"sample_id": "paper", "claim_id": "p-extra", "verdict": "supported", "review_status": "completed",
         "evidence_ids": [], "claim_anchor": _anchor("样品数量减少一半", 10, 17)},
    ]}
    gold = {"schema_version": 1, "extraction_samples": ["paper"], "labels": [
        {"sample_id": "paper", "claim_id": "g-exact", "verdict": "supported", "accepted_evidence_groups": [["e1"]],
         "claim_anchor": _anchor("风速上升", 0, 9), "split": "test"},
        {"sample_id": "paper", "claim_id": "g-paraphrase", "verdict": "refuted", "accepted_evidence_groups": [["e2"]],
         "claim_anchor": _anchor("样品重量下降 50%", 7, 20), "split": "test"},
    ]}
    return predictions, gold


def _pair(prediction="p-paraphrase", gold="g-paraphrase", sample="paper"):
    return {"sample_id": sample, "predicted_claim_id": prediction, "gold_claim_id": gold}


def _manifest(predictions, gold, *pairs):
    return {"schema_version": 1, "annotation_id": "held-out-review-01", "annotator_type": "human",
            "predictions_sha256": semantic_manifest_sha256(predictions), "gold_sha256": semantic_manifest_sha256(gold),
            "matches": list(pairs)}


def test_semantic_alignment_reports_additional_matches_without_changing_exact_results():
    predictions, gold = _manifests()
    review = _manifest(predictions, gold, _pair(), _pair("p-exact", "g-exact"))

    exact = score_predictions(predictions, gold)
    scored = score_predictions(predictions, gold, semantic_matches=review)
    extraction = scored.pop("semantic_extraction")
    semantic = scored.pop("semantic_scoring")

    assert scored == exact
    assert exact["extraction"]["recall"]["value"] == 0.5
    assert exact["cases"][1]["predicted_claim_id"] is None
    assert extraction["precision"] == {"numerator": 2, "denominator": 3, "value": 2 / 3}
    assert extraction["recall"]["value"] == 1
    assert extraction["f1"]["value"] == 0.8
    assert extraction["missing"] == []
    assert extraction["extra"] == [{"sample_id": "paper", "claim_id": "p-extra"}]
    assert extraction["matches"] == [
        {**_pair("p-exact", "g-exact"), "source": "exact"},
        {**_pair(), "source": "annotation"},
    ]
    assert extraction["provenance"] == {
        **{key: review[key] for key in ("annotation_id", "annotator_type", "predictions_sha256", "gold_sha256")},
        "annotation_sha256": semantic_manifest_sha256(review),
    }
    assert semantic["extraction"]["recall"] == extraction["recall"]
    assert semantic["joint_accuracy"]["value"] == 1
    assert semantic["completed_coverage"]["value"] == 1
    assert semantic["cases"][1]["predicted_claim_id"] == "p-paraphrase"
    assert semantic["classification_by_split"]["test"]["completed_accuracy"]["value"] == 1


def test_review_binds_to_both_manifests_and_canonical_key_order():
    predictions, gold = _manifests()
    review = _manifest(predictions, gold, _pair())
    reversed_keys = {key: predictions[key] for key in reversed(predictions)}
    assert semantic_manifest_sha256(reversed_keys) == semantic_manifest_sha256(predictions)
    assert score_predictions(reversed_keys, gold, semantic_matches=review)["semantic_scoring"]["accuracy"]["value"] == 1

    different_predictions = copy.deepcopy(predictions)
    different_predictions["predictions"][1]["verdict"] = "supported"
    with pytest.raises(ValueError, match="predictions_sha256"):
        score_predictions(different_predictions, gold, semantic_matches=review)

    different_gold = copy.deepcopy(gold)
    different_gold["labels"][1]["accepted_evidence_groups"] = []
    with pytest.raises(ValueError, match="gold_sha256"):
        score_predictions(predictions, different_gold, semantic_matches=review)


@pytest.mark.parametrize(("pairs", "error"), [
    ([_pair(), _pair()], "one-to-one"),
    ([_pair(), _pair("p-extra")], "one-to-one"),
    ([_pair("p-exact")], "conflicts with an existing exact match"),
    ([_pair("p-paraphrase", "g-exact")], "conflicts with an existing exact match"),
    ([_pair("unknown")], "existing prediction and gold claim"),
    ([_pair(gold="unknown")], "existing prediction and gold claim"),
    ([_pair(sample="another")], "one extraction sample"),
    ([{**_pair(), "verdict": "refuted"}], "must contain exactly"),
])
def test_review_rejects_duplicate_conflicting_and_unknown_pairs(pairs, error):
    predictions, gold = _manifests()
    with pytest.raises(ValueError, match=error):
        score_predictions(predictions, gold, semantic_matches=_manifest(predictions, gold, *pairs))


@pytest.mark.parametrize("anchor", [_anchor("paraphrase", 0, 7), _anchor("paraphrase", sentence_id="s2"), None])
def test_review_requires_same_sentence_and_positive_overlap(anchor):
    predictions, gold = _manifests()
    row = predictions["predictions"][1]
    if anchor is None:
        del row["claim_anchor"]
    else:
        row["claim_anchor"] = anchor
    with pytest.raises(ValueError, match="overlapping source intervals in the same sentence"):
        score_predictions(predictions, gold, semantic_matches=_manifest(predictions, gold, _pair()))


def test_empty_review_preserves_extraction_and_records_annotation_source():
    predictions, gold = _manifests()
    review = _manifest(predictions, gold)
    review["annotator_type"] = "ai"
    scored = score_predictions(predictions, gold, semantic_matches=review)

    assert scored["semantic_extraction"]["provenance"]["annotator_type"] == "ai"
    assert scored["semantic_extraction"]["f1"] == scored["extraction"]["f1"]
    assert [match["source"] for match in scored["semantic_extraction"]["matches"]] == ["exact"]
