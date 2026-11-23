from copy import deepcopy

import pytest

from abstract_review.review_quality import score_review_quality


def sample(sample_id, *, revision_statuses=(), abstract_status="passed", issues=(), calculations=(), roles=None):
    version_id = f"version-{sample_id}"
    return {"sample_id": sample_id, "pdf_sha256": "a" * 64, "review": {
        "version_id": version_id, "original_text": "Original.", "reviewed_text": "Edited.",
        "revisions": [{"revision_id": rid, "status": status} for rid, status in revision_statuses],
        "abstract_review": {"version_id": version_id, "status": abstract_status, "issues": list(issues)},
        "calculations": list(calculations), "evidence_roles": roles or {},
    }}


def label(sample_id, **fields):
    return {"sample_id": sample_id, "version_id": f"version-{sample_id}", **fields}


def ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def test_revision_correctness_and_whole_review_keep_deferred_checks_separate():
    revisions = (("unsafe", "passed"), ("wrong-return", "returned"), ("deferred", "manual_review"), ("partial", "passed"))
    issue = {"sentence_ids": ["s1"], "reason": "Scope drift."}
    run = {"schema_version": 1, "samples": [
        sample("one", revision_statuses=revisions, abstract_status="returned", issues=[issue]),
        sample("two"),
        sample("three", abstract_status="manual_review", issues=[issue]),
        sample("four", abstract_status="returned", issues=[issue]),
    ]}
    gold = {"schema_version": 1, "revision_checks": [
        label("one", revision_id="unsafe", corrected=True, facts_preserved=False, unsupported_added=False),
        label("one", revision_id="wrong-return", corrected=True, facts_preserved=True, unsupported_added=False),
        label("one", revision_id="deferred", corrected=True, facts_preserved=True, unsupported_added=False),
        label("one", revision_id="partial", corrected=None, facts_preserved=True, unsupported_added=None),
    ], "abstract_checks": [label(case, has_conflict=conflict) for case, conflict in (
        ("one", True), ("two", True), ("three", False), ("four", False),
    )]}
    result = score_review_quality(run, gold)
    assert result["revisions"] == {
        "correction_rate": ratio(3, 3), "fact_preservation_rate": ratio(3, 4),
        "unsupported_addition_rate": ratio(0, 3), "wrong_pass_rate": ratio(1, 1),
        "wrong_return_rate": ratio(1, 2), "deferred_rate": ratio(1, 4),
    }
    assert result["abstracts"] == {"accuracy": ratio(1, 4), "conflict_recall": ratio(1, 2),
                                     "false_conflict_rate": ratio(1, 2), "deferred_rate": ratio(1, 4)}
    assert result["cases"]["revision_checks"][-1]["human_safe"] is None
    assert result["cases"]["abstract_checks"][2]["detected_conflict"] is None
    assert {row["phase"] for row in result["errors"]} == {"revision", "version_verification"}
    assert result["tables"]["joint_accuracy"] == ratio(0, 0)


def test_table_scores_exact_operand_order_and_role_specific_anchors():
    values = ["baseline", "result"]
    headers = ["column", "row"]
    notes = ["note-a"]
    good = {"claim_id": "one", "operation": "percentage_points", "result": "8.0", "unit": "percentage_points",
            "evidence_ids": values, "scope_evidence_ids": [*headers, *notes, "method"], "status": "calculated"}
    wrong = {**good, "claim_id": "two", "evidence_ids": values[::-1],
             "scope_evidence_ids": ["column", "note-b", "method"]}
    wrong_role = {**good, "claim_id": "three", "evidence_ids": ["text", "result"], "unit": "percent"}
    roles = {"baseline": "value", "result": "value", "column": "column_header", "row": "row_header",
             "note-a": "table_note", "note-b": "table_note", "method": "body_text", "text": "body_text"}
    run = {"schema_version": 1, "samples": [sample("one", calculations=[good, wrong, wrong_role], roles=roles)]}
    gold = {"schema_version": 1, "table_checks": [
        label("one", claim_id=claim, operation="percentage_points", operand_ids=operands, header_ids=headers,
              note_ids=notes, result="8", unit="percentage_points")
        for claim, operands in (("one", values), ("two", values), ("three", ["text", "result"]), ("missing", values))
    ]}
    result = score_review_quality(run, gold)
    assert result["tables"] == {"operand_accuracy": ratio(1, 4), "header_accuracy": ratio(2, 4),
                                "note_accuracy": ratio(2, 4), "calculation_accuracy": ratio(2, 4),
                                "joint_accuracy": ratio(1, 4), "calculation_coverage": ratio(3, 4)}
    assert result["cases"]["table_checks"][0]["scope_evidence_ids"] == [*headers, *notes, "method"]
    assert result["cases"]["table_checks"][1]["evidence_ids"] == values[::-1]
    assert [row["claim_id"] for row in result["errors"]] == ["two", "three", "missing"]


def test_quality_rejects_stale_version_duplicate_targets_and_invalid_labels():
    run = {"schema_version": 1, "samples": [sample("one", revision_statuses=[("r", "passed")])]}
    check = label("one", revision_id="r", corrected=True, facts_preserved=True, unsupported_added=False)
    with pytest.raises(ValueError, match="does not match the recorded review"):
        score_review_quality(run, {"schema_version": 1, "revision_checks": [{**check, "version_id": "previous"}]})
    with pytest.raises(ValueError, match="duplicate.*revision_id"):
        score_review_quality(run, {"schema_version": 1, "revision_checks": [check, check]})
    with pytest.raises(ValueError, match="true, false or null"):
        score_review_quality(run, {"schema_version": 1, "revision_checks": [{**check, "corrected": 1}]})
    with pytest.raises(ValueError, match="duplicate sample_id"):
        score_review_quality({**run, "samples": run["samples"] * 2}, {"schema_version": 1})
    stale = deepcopy(run)
    stale["samples"][0]["review"]["abstract_review"]["version_id"] = "previous"
    with pytest.raises(ValueError, match="abstract verification does not match"):
        score_review_quality(stale, {"schema_version": 1, "abstract_checks": [label("one", has_conflict=False)]})
