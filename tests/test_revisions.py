from copy import deepcopy

import pytest

from abstract_review.audit import sentences
from abstract_review.revisions import apply_revisions, build_candidate, new_version, repair_candidate, verify_version


SOURCE = "Accuracy increased by 12 percentage points on Dataset A."
CORRECTION = "Accuracy increased by 8 percentage points on Dataset A."


def cell(eid, text, column, header, *, row=1):
    return {"evidence_id": eid, "text": text, "kind": "table_cell", "doc_id": "paper",
            "row_header": "Dataset A", "column_header": header,
            "locator": {"page_index": 1, "table_index": 0, "row_index": row, "column_index": column}}


def report_for(text=SOURCE):
    original = new_version(text, "pdf_original", "", [])
    evidence = [cell("baseline", "80%", 1, "Baseline"), cell("result", "88%", 2, "Final")]
    return {"original_abstract": original, "findings": [{
        "claim_id": "claim-a", "sentence_id": "s1", "start": 0, "end": len(SOURCE), "text": SOURCE,
        "proposition": SOURCE, "review_status": "completed", "verdict": "refuted",
        "reason": "The original claimed the wrong amount.", "evidence_ids": ["baseline", "result"],
    }], "evidence": evidence, "revisions": [], "verifications": [], "unresolved_items": []}


def test_apply_revisions_preserves_unmodified_text_and_checks_original_ranges():
    original = "甲 乙 丙"
    first = {"start": 0, "end": 1, "original_text": "甲", "replacement": "甲一"}
    last = {"start": 4, "end": 5, "original_text": "丙", "replacement": "丙三"}
    assert apply_revisions(original, []) == original
    assert apply_revisions(original, [last, first]) == "甲一 乙 丙三"
    with pytest.raises(ValueError, match="overlap"):
        apply_revisions(original, [first, first])
    with pytest.raises(ValueError, match="does not match"):
        apply_revisions(original, [{**first, "original_text": "丙"}])
    with pytest.raises(ValueError, match="outside"):
        apply_revisions(original, [{**last, "end": 9}])


def test_version_fingerprint_depends_on_exact_text_origin_and_revision():
    a = new_version("a\r\nb", "candidate", "original", ["r1"])
    assert a == new_version("a\r\nb", "candidate", "original", ["r1"])
    assert a["version_id"] != new_version("a\nb", "candidate", "original", ["r1"])["version_id"]
    assert a["version_id"] != new_version("a\r\nb", "selected", "original", ["r1"])["version_id"]
    assert a["version_id"] != new_version("a\r\nb", "candidate", "another", ["r1"])["version_id"]


def test_offline_revision_is_limited_to_scoped_numeric_correction():
    remainder = "  The primary test split remained fixed."
    report = report_for(SOURCE + remainder)
    original = report["original_abstract"]
    candidate = build_candidate(report)
    assert candidate["text"] == CORRECTION + remainder
    assert candidate["parent_version_id"] == original["version_id"]
    assert original["text"] == SOURCE + remainder
    replacement, = report["revisions"]
    assert replacement["status"] == "passed"
    assert candidate["revision_ids"] == [replacement["revision_id"]]
    change, = replacement["semantic_diff"]
    assert (change["before"], change["after"], change["evidence_ids"]) == ("12", "8", ["baseline", "result"])
    verified = verify_version(report, original, candidate)
    local, = verified["sentence_checks"]
    assert local["status"] == "passed"
    assert local["version_id"] == verified["version_id"] == candidate["version_id"]
    assert verified["status"] == "manual_review"


def test_build_candidate_leaves_empty_and_ambiguous_cases_for_review():
    empty = report_for()
    empty["findings"] = []
    assert build_candidate(empty) is None
    assert empty["revisions"] == []
    ambiguous = report_for()
    ambiguous["evidence"] += [cell("other-base", "70%", 1, "Baseline", row=2),
                              cell("other-result", "79%", 2, "Final", row=2)]
    ambiguous["findings"][0]["candidate_ids"] = ["other-base", "other-result"]
    assert build_candidate(ambiguous) is None
    assert ambiguous["unresolved_items"][0]["code"] == "revision_manual_review"


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def call(self, stage, material, schema):
        expected, payload = next(self.replies)
        assert stage == expected
        self.calls.append((stage, deepcopy(material)))
        return schema.model_validate(payload)


def draft(replacement, number):
    return {"replacement": replacement, "evidence_ids": ["baseline", "result"], "semantic_diff": [{
        "kind": "modified", "field": "percentage_points", "before": "12", "after": number,
        "evidence_ids": ["baseline", "result"],
    }]}


def test_model_requests_are_independent_and_cross_sentence_scope_is_located():
    conclusion = "Every dataset showed that improvement."
    report = report_for(SOURCE + " " + conclusion)
    model = ScriptedModel([
        ("revise", draft(SOURCE.replace("12", "9"), "9")),
        ("verify_sentence", {"status": "returned", "reason": "The cells give 8 points.", "evidence_ids": ["baseline", "result"]}),
        ("revise", draft(CORRECTION, "8")),
        ("verify_sentence", {"status": "passed", "reason": "The scoped amount matches.", "evidence_ids": ["baseline", "result"]}),
        ("verify_abstract", {"status": "returned", "reason": "Scope conflict.", "evidence_ids": ["result"], "issues": [{
            "sentence_ids": ["s1", "s2"], "reason": "Dataset A does not establish every dataset.", "evidence_ids": ["result"],
        }]}),
    ])
    original = report["original_abstract"]
    candidate = build_candidate(report, model)
    verification = verify_version(report, original, candidate, model)
    assert [call[0] for call in model.calls] == ["revise", "verify_sentence", "revise", "verify_sentence", "verify_abstract"]
    for stage, material in model.calls:
        if stage == "verify_sentence":
            assert set(material) == {"original_sentence", "revised_sentence", "evidence"}
            assert material["original_sentence"] == SOURCE
            assert {item["evidence_id"] for item in material["evidence"]} == {"baseline", "result"}
    assert model.calls[-1][1]["version_id"] == candidate["version_id"]
    assert report["verifications"][0]["repair_count"] == 1
    assert verification["status"] == "returned"
    locations = verification["issues"][0]["locations"]
    assert [candidate["text"][item["start"]:item["end"]] for item in locations] == [CORRECTION, conclusion]
    assert repair_candidate(report, candidate, verification, model) is None
    assert len(model.calls) == 5


def test_whole_repair_rebases_selected_location_and_consumes_shared_budget():
    second = "Dataset A recorded an increase of 9 percentage points."
    report = report_for(SOURCE + " " + second)
    report["findings"].append({**report["findings"][0], "claim_id": "claim-b", "sentence_id": "s2",
                               "start": len(SOURCE) + 1, "end": len(SOURCE) + 1 + len(second),
                               "text": second, "proposition": second})
    candidate = build_candidate(report)
    first, original_patch = report["revisions"]
    location = sentences(candidate["text"])[1]
    assert location["start"] != original_patch["start"]
    issue = {"sentence_ids": ["s2"], "reason": "Make the documented split explicit.",
             "locations": [location], "evidence_ids": ["result"]}
    whole = {"status": "returned", "version_id": candidate["version_id"], "issues": [issue]}
    replacement = "Dataset A recorded an increase of 8 percentage points on the fixed split."
    model = ScriptedModel([
        ("revise", {"replacement": replacement, "evidence_ids": ["baseline", "result"], "semantic_diff": [{
            "kind": "modified", "field": "scope", "before": "unspecified split", "after": "fixed split",
            "evidence_ids": ["result"],
        }]}),
        ("verify_sentence", {"status": "passed", "reason": "Scope and value verified.", "evidence_ids": ["result"]}),
    ])

    repaired = repair_candidate(report, candidate, whole, model)
    assert repaired["text"] == CORRECTION + " " + replacement
    latest = report["revisions"][-1]
    assert latest["start"] == original_patch["start"] == len(SOURCE) + 1
    assert repaired["revision_ids"] == [first["revision_id"], latest["revision_id"]]
    assert latest["replaces_revision_id"] == original_patch["revision_id"]
    assert original_patch["status"] == "superseded"
    assert report["verifications"][-1]["repair_count"] == 1
    assert set(model.calls[-1][1]) == {"original_sentence", "revised_sentence", "evidence"}

    repeated_issue = {**issue, "locations": [sentences(repaired["text"])[1]]}
    assert repair_candidate(report, repaired, {**whole, "version_id": repaired["version_id"],
                                                 "issues": [repeated_issue]}, model) is None
    assert len(model.calls) == 2


def test_whole_repair_rejects_ambiguous_duplicate_targets_and_unscoped_evidence():
    second = "Dataset A recorded an increase of 9 percentage points."
    report = report_for(SOURCE + " " + second)
    report["findings"].append({**report["findings"][0], "claim_id": "claim-b", "sentence_id": "s2",
                               "start": len(SOURCE) + 1, "end": len(SOURCE) + 1 + len(second),
                               "text": second, "proposition": second})
    candidate = build_candidate(report)
    issued = {"status": "returned", "version_id": candidate["version_id"], "issues": [{
        "reason": "The reviewer has not selected one position to repair.", "evidence_ids": ["result"],
        "locations": sentences(candidate["text"]),
    }]}
    model = ScriptedModel([])
    assert repair_candidate(report, candidate, issued, model) is None
    issued["issues"][0]["locations"] = sentences(candidate["text"])[:1]
    issued["issues"][0]["evidence_ids"] = ["out-of-scope"]
    assert repair_candidate(report, candidate, issued, model) is None
    assert not model.calls


def test_unknown_citation_fails_independent_revision_check():
    report = report_for()
    model = ScriptedModel([
        ("revise", draft(CORRECTION, "8")),
        ("verify_sentence", {"status": "passed", "reason": "Out of scope", "evidence_ids": ["unseen"]}),
    ])
    assert build_candidate(report, model) is None
    assert report["revisions"][0]["status"] == "manual_review"
    assert report["verifications"][0]["evidence_ids"] == []


def test_whole_check_cannot_pass_with_a_failed_manual_edit():
    report = report_for()
    original = report["original_abstract"]
    manual = new_version(CORRECTION.replace("Dataset A", "all datasets"), "selected", original["version_id"], [])
    model = ScriptedModel([
        ("verify_sentence", {"status": "returned", "reason": "The new scope is unsupported.", "evidence_ids": ["result"]}),
        ("verify_abstract", {"status": "passed", "reason": "No cross-sentence issue.", "evidence_ids": ["result"], "issues": []}),
    ])
    checked = verify_version(report, original, manual, model)
    assert checked["status"] == "manual_review"
    assert checked["version_id"] == manual["version_id"]
    assert checked["sentence_checks"][0]["status"] == "returned"
