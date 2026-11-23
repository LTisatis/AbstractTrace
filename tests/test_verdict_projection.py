from copy import deepcopy

import pytest

from abstract_review.scoring import score_predictions, semantic_manifest_sha256


def anchor(proposition, start, end, sentence_id="s18"):
    return {"sentence_id": sentence_id, "start": start, "end": end, "proposition": proposition}


def manifests():
    predictions = {"schema_version": 1, "predictions": [
        {"sample_id": "paper", "claim_id": "whole", "review_status": "completed", "verdict": "refuted",
         "claim_anchor": anchor("We used the tool and accepted every flag without manual review.", 10, 75),
         "evidence_ids": ["p9-review", "p9-all", "p14-earlier-study"],
         "candidate_ids": ["p14-earlier-study", "p9-review", "p9-all", "p15-procedure"]},
        {"sample_id": "paper", "claim_id": "extra", "review_status": "completed", "verdict": "supported",
         "claim_anchor": anchor("The authors ran the tool.", 10, 28), "evidence_ids": ["p15-procedure"]},
    ]}
    gold = {"schema_version": 1, "extraction_samples": ["paper"], "labels": [
        {"sample_id": "paper", "claim_id": "used-tool", "verdict": "supported",
         "claim_anchor": anchor("The authors used the tool.", 10, 28),
         "accepted_evidence_groups": [["p15-procedure"]]},
        {"sample_id": "paper", "claim_id": "accepted-flags", "verdict": "refuted",
         "claim_anchor": anchor("The study accepted all flags without manual review.", 31, 74),
         "accepted_evidence_groups": [["p9-review", "p9-all"]]},
    ]}
    return predictions, gold


def row(gold):
    return {"sample_id": "paper", "predicted_claim_id": "whole", "gold_claim_id": "accepted-flags",
            "relation": "conjunct", "operand_anchor": deepcopy(gold["labels"][1]["claim_anchor"]),
            "evidence_ids": ["p9-review", "p9-all"],
            "review_note": "The judgment cites the authors' manual check of all flagged results on p9."}


def annotation(predictions, gold, collection, *entries):
    return {"schema_version": 1, "annotation_id": "independent-01", "annotator_type": "human",
            "predictions_sha256": semantic_manifest_sha256(predictions),
            "gold_sha256": semantic_manifest_sha256(gold), collection: list(entries)}


def test_verdict_projection_credits_only_the_contradicted_conjunct():
    predictions, gold = manifests()
    equivalent = annotation(predictions, gold, "matches", {
        "sample_id": "paper", "predicted_claim_id": "extra", "gold_claim_id": "used-tool",
    })
    projection = annotation(predictions, gold, "projections", row(gold))
    exact_semantic = score_predictions(predictions, gold, semantic_matches=equivalent)
    scored = score_predictions(predictions, gold, semantic_matches=equivalent, verdict_projections=projection)
    diagnostic = scored.pop("verdict_projection")
    assert scored == exact_semantic
    assert scored["semantic_scoring"]["joint_accuracy"]["numerator"] == 1
    assert diagnostic == {
        "credited_count": 1,
        "projections": [{**row(gold), "evidence_ids": ["p9-all", "p9-review"]}],
        "provenance": {
            **{key: projection[key] for key in ("annotation_id", "annotator_type", "predictions_sha256", "gold_sha256")},
            "annotation_sha256": semantic_manifest_sha256(projection),
        },
    }


@pytest.mark.parametrize(("mutation", "error"), [
    ({"gold_claim_id": "used-tool"}, "exact gold operand"),
    ({"predicted_claim_id": "extra"}, "strictly contained operand"),
    ({"sample_id": "another"}, "one extraction sample"),
    ({"relation": "disjunct"}, "relation must be conjunct"),
    ({"review_note": "  "}, "review_note must explain"),
    ({"evidence_ids": ["p14-earlier-study"]}, "complete accepted evidence group"),
    ({"evidence_ids": ["p9-review"]}, "complete accepted evidence group"),
    ({"evidence_ids": ["p9-review", "p9-all", "p15-procedure"]}, "both cited and present"),
])
def test_projection_rejects_wrong_operand_relation_or_evidence(mutation, error):
    predictions, gold = manifests()
    projection = annotation(predictions, gold, "projections", {**row(gold), **mutation})
    with pytest.raises(ValueError, match=error):
        score_predictions(predictions, gold, verdict_projections=projection)


@pytest.mark.parametrize(("field", "value", "error"), [
    ("candidate_ids", ["p9-review"], "both cited and present"),
    ("verdict", "supported", "completed refutation"),
    ("claim_anchor", anchor("Whole assertion on another sentence", 10, 75, "s17"), "strictly contained operand"),
    ("claim_anchor", anchor("Different assertion with identical source span", 31, 74), "strictly contained operand"),
])
def test_projection_rechecks_source_status_and_scope(field, value, error):
    predictions, gold = manifests()
    predictions["predictions"][0][field] = value
    projection = annotation(predictions, gold, "projections", row(gold))
    with pytest.raises(ValueError, match=error):
        score_predictions(predictions, gold, verdict_projections=projection)


def test_projection_requires_verifiable_candidate_trace_and_matching_manifest():
    predictions, gold = manifests()
    projection = annotation(predictions, gold, "projections", row(gold))
    del predictions["predictions"][0]["candidate_ids"]
    with pytest.raises(ValueError, match="predictions_sha256"):
        score_predictions(predictions, gold, verdict_projections=projection)
    projection["predictions_sha256"] = semantic_manifest_sha256(predictions)
    with pytest.raises(ValueError, match="both cited and present"):
        score_predictions(predictions, gold, verdict_projections=projection)


def test_projection_cannot_override_semantic_or_exact_alignment():
    predictions, gold = manifests()
    equivalent = annotation(predictions, gold, "matches", {
        "sample_id": "paper", "predicted_claim_id": "whole", "gold_claim_id": "accepted-flags",
    })
    projection = annotation(predictions, gold, "projections", row(gold))
    with pytest.raises(ValueError, match="existing alignment or projection"):
        score_predictions(predictions, gold, semantic_matches=equivalent, verdict_projections=projection)

    predictions["predictions"][1]["claim_anchor"] = deepcopy(gold["labels"][1]["claim_anchor"])
    projection = annotation(predictions, gold, "projections", row(gold))
    with pytest.raises(ValueError, match="existing alignment or projection"):
        score_predictions(predictions, gold, verdict_projections=projection)


def test_projection_does_not_propagate_one_verdict_to_multiple_operands():
    predictions, gold = manifests()
    other = {**gold["labels"][1], "claim_id": "another-operand",
             "claim_anchor": anchor("Another independently refutable assertion.", 11, 23)}
    gold["labels"].append(other)
    projection = annotation(predictions, gold, "projections", row(gold), {
        **row(gold), "gold_claim_id": other["claim_id"], "operand_anchor": other["claim_anchor"],
    })
    with pytest.raises(ValueError, match="existing alignment or projection"):
        score_predictions(predictions, gold, verdict_projections=projection)
