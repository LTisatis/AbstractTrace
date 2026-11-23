from copy import deepcopy

import pytest

from abstract_review.audit import sentences
from abstract_review.calculations import check_revision_numbers
from abstract_review.followups import FollowUps
from abstract_review.models import VerificationQuery
from abstract_review.revisions import apply_revisions, build_candidate, new_version, repair_candidate, verify_version
from abstract_review.workflow import _refresh_findings


SOURCE = "Accuracy increased by 12 percentage points on Dataset A."
CORRECTION = "Accuracy increased by 8 percentage points on Dataset A."


def cell(eid, text, column, header, *, row=1, subject="Dataset A"):
    return {"evidence_id": eid, "text": text, "kind": "table_cell", "doc_id": "paper",
            "row_header": subject, "column_header": header,
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
    assert local["calculations"][0]["verification"] == "matched"
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
    report["evidence"].append({"evidence_id": "split", "doc_id": "paper", "kind": "body_text",
                               "text": "Dataset A uses the fixed split.",
                               "locator": {"page_index": 0, "section": "Methods"}})
    candidate = build_candidate(report)
    first, original_patch = report["revisions"]
    location = sentences(candidate["text"])[1]
    assert location["start"] != original_patch["start"]
    issue = {"sentence_ids": ["s2"], "reason": "Make the documented split explicit.",
             "locations": [location], "evidence_ids": ["result", "split"]}
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
    assert report["verifications"][-1]["calculations"][0]["scope_evidence_ids"] == ["split"]
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


def test_recalculation_overrules_online_approval_and_gives_the_repair_its_evidence():
    report = report_for()
    approvals = {"status": "passed", "reason": "The model accepts it.", "evidence_ids": ["result"]}
    model = ScriptedModel([
        ("revise", draft(SOURCE.replace("12", "9"), "9")),
        ("verify_sentence", approvals),
        ("revise", draft(CORRECTION, "8")),
        ("verify_sentence", approvals),
    ])
    candidate = build_candidate(report, model)
    assert candidate["text"] == CORRECTION
    feedback = model.calls[2][1]["verification_feedback"]
    assert feedback["status"] == "returned"
    assert {"baseline", "result"} == set(feedback["evidence_ids"])
    assert "9" in feedback["reason"] and "8" in feedback["reason"]
    calculation, = report["verifications"][0]["calculations"]
    assert calculation["verification"] == "matched"
    assert calculation["operands"] == ["80%", "88%"]


@pytest.mark.parametrize("replacement,verification,sentence_status", [
    ("On Dataset A, accuracy increased by 8%.", "mismatch", "returned"),
    ("On Dataset A, accuracy decreased by 8 percentage points.", "mismatch", "returned"),
    ("Accuracy increased from 82% to 88% on Dataset A.", "mismatch", "returned"),
    ("Against baseline B, accuracy increased by 8 percentage points on Dataset A.", "unresolved", "manual_review"),
    ("Accuracy increased by 8 percentage points on Dataset A's validation split.", "unresolved", "manual_review"),
])
def test_selected_version_does_not_accept_a_model_pass_for_numbers_with_issues(replacement, verification, sentence_status):
    report = report_for()
    original = report["original_abstract"]
    report["evidence"] += [cell("other-base", "80%", 1, "Baseline", row=2, subject="Dataset B"),
                          cell("other-result", "88%", 2, "Final", row=2, subject="Dataset B")]
    selected = new_version(replacement, "selected", original["version_id"], [])
    approval = {"status": "passed", "reason": "The model accepts it.", "evidence_ids": ["result"]}
    model = ScriptedModel([("verify_sentence", approval), ("verify_abstract", {**approval, "issues": []})])
    whole = verify_version(report, original, selected, model)
    check, = whole["sentence_checks"]
    assert check["status"] == sentence_status
    assert whole["status"] == "manual_review"
    assert selected["text"] == replacement
    failed = next(item for item in check["calculations"] if item["verification"] == verification)
    if verification == "mismatch":
        assert failed["evidence_ids"] == ["baseline", "result"]


def test_saved_passed_sentence_is_recalculated_without_replaying_a_model_request():
    report = report_for()
    candidate = build_candidate(report)
    original = report["original_abstract"]
    wrong = SOURCE.replace("12", "9")
    report["revisions"][0]["replacement"] = wrong
    report["verifications"][0]["revised_sentence"] = wrong
    report["verifications"][0].pop("calculations")
    legacy = new_version(wrong, "candidate", original["version_id"], candidate["revision_ids"])
    model = ScriptedModel([("verify_abstract", {
        "status": "passed", "reason": "Stored sentence should be checked.", "evidence_ids": ["result"], "issues": [],
    })])
    checked = verify_version(report, original, legacy, model)
    assert checked["status"] == "manual_review"
    assert checked["sentence_checks"][0]["calculations"][0]["verification"] == "mismatch"
    assert build_candidate(report, ScriptedModel([])) is None
    assert report["verifications"][0]["status"] == "passed"


def test_recalculation_keeps_undefined_relative_change_and_conflicting_units_unresolved():
    evidence = [cell("baseline", "0%", 1, "Baseline"), cell("result", "10%", 2, "Final")]
    [relative] = check_revision_numbers(SOURCE, "On Dataset A, accuracy increased by 10%.", evidence)
    assert (relative["operation"], relative["status"], relative["verification"]) == (
        "relative_change", "not_calculable", "unresolved",
    )
    [absolute] = check_revision_numbers(SOURCE, "On Dataset A, accuracy increased by 10.", evidence)
    assert absolute["verification"] == "unresolved"
    assert "units" in absolute["reason"]


def test_recalculation_accepts_scalar_differences_and_preserves_chinese_percentages():
    evidence = [cell("baseline", "1.6", 1, "Baseline RMSE"), cell("result", "1.4", 2, "Final RMSE")]
    [scalar] = check_revision_numbers("On Dataset A, RMSE fell by 0.3.", "On Dataset A, RMSE fell by 0.2.", evidence)
    assert (scalar["operation"], scalar["result"], scalar["verification"]) == ("absolute_difference", "-0.2", "matched")
    [points] = check_revision_numbers("数据集A的准确率提高12个百分点。", "数据集A的准确率提高8个百分点。", report_for()["evidence"])
    assert points["verification"] == "matched"


def test_sentence_supplement_is_shared_with_whole_review_and_restart():
    report = report_for()
    report["evidence"].append({"evidence_id": "limitation", "doc_id": "paper", "kind": "body_text",
                               "text": "Dataset A used a fixed validation split.",
                               "locator": {"page_index": 2, "section": "Limitations"}})
    query = {"query": "fixed validation split", "requirement": "scope", "section": "Limitations"}
    model = ScriptedModel([
        ("revise", draft(CORRECTION, "8")),
        ("verify_sentence", {"status": "returned", "reason": "Check the split.",
                             "evidence_ids": ["result"], "follow_up_queries": [query]}),
        ("verify_sentence", {"status": "passed", "reason": "Split checked.",
                             "evidence_ids": ["result", "limitation"]}),
    ])
    candidate = build_candidate(report, model)
    assert candidate["text"] == CORRECTION
    assert "limitation" not in {item["evidence_id"] for item in model.calls[1][1]["evidence"]}
    assert "limitation" in {item["evidence_id"] for item in model.calls[2][1]["evidence"]}
    assert report["follow_up_rounds"]["claim-a"]["evidence_ids"] == ["limitation"]

    restored = deepcopy(report)
    whole = ScriptedModel([("verify_abstract", {
        "status": "returned", "reason": "Check another result.", "evidence_ids": ["limitation"],
        "issues": [{"reason": "Need another source.", "sentence_ids": ["s1"], "evidence_ids": [],
                    "follow_up_queries": [{**query, "section": "Results"}]}],
    })])
    assert verify_version(restored, restored["original_abstract"], candidate, whole)["status"] == "manual_review"
    assert len(whole.calls) == 1
    assert restored["follow_up_rounds"] == report["follow_up_rounds"]


def test_grounded_omission_uses_same_independent_revision_path():
    source = "The primary endpoint was evaluated."
    report = report_for(source)
    report["findings"] = []
    report["evidence"].append({"evidence_id": "limit", "doc_id": "paper", "kind": "body_text",
                               "text": "The primary endpoint used a fixed split.",
                               "locator": {"page_index": 2, "section": "Methods"}})
    report["omissions"] = [{"status": "possible_omission", "anchor_sentence_id": "s1",
                            "reason": "The split should be stated.", "evidence_ids": ["limit"]}]
    replacement = source.removesuffix(".") + " using a fixed split."
    model = ScriptedModel([
        ("revise", {"replacement": replacement, "evidence_ids": ["limit"], "semantic_diff": [{
            "kind": "added", "field": "scope", "before": None, "after": "fixed split", "evidence_ids": ["limit"],
        }]}),
        ("verify_sentence", {"status": "passed", "reason": "Scope confirmed.", "evidence_ids": ["limit"]}),
    ])
    candidate = build_candidate(report, model)
    assert build_candidate(report, ScriptedModel([])) == candidate
    assert len(report["revisions"]) == 1
    assert model.calls[0][1]["findings"][0]["extension"] == "omission"
    assert candidate["text"] == replacement


def test_reserved_supplement_is_replayed_when_interrupted_before_lookup():
    report = report_for()
    report["evidence"].append({"evidence_id": "new", "doc_id": "paper", "kind": "body_text",
                               "text": "Dataset A used a fixed validation split.",
                               "locator": {"page_index": 2, "section": "Limitations"}})
    snapshots = []

    def interrupt():
        snapshots.append(deepcopy(report))
        raise KeyboardInterrupt()

    query = VerificationQuery(query="fixed split", requirement="scope", section="Limitations")
    with pytest.raises(KeyboardInterrupt):
        FollowUps(report, interrupt).collect(["claim-a"], [query], [], "verify_sentence")
    restored = FollowUps(snapshots[0])
    assert restored.rounds["claim-a"]["queries"][0]["section"] == "Limitations"
    assert [item["evidence_id"] for item in restored.material(["claim-a"], [])] == ["new"]
    assert restored.rounds["claim-a"]["evidence_ids"] == ["new"]


def test_changed_original_judgement_invalidates_only_affected_drafts(monkeypatch, tmp_path):
    report = report_for()
    report["evidence"].append({"evidence_id": "corrected-result", "doc_id": "paper", "kind": "body_text",
                               "text": "A corrected result supersedes the earlier table.", "locator": {"page_index": 3}})
    report["claims"] = [deepcopy(report["findings"][0])]
    report["documents"] = []
    report["calculations"] = []
    candidate = build_candidate(report)
    report["candidate_proposal"] = candidate
    report["revisions"].append({**report["revisions"][0], "sentence_id": "s2", "revision_id": "unrelated"})
    followups = FollowUps(report)
    followups.rounds["claim-a"] = {"stage": "verify_sentence", "evidence_ids": ["corrected-result"]}

    def recheck(claim, index, model, **kwargs):
        assert [row["evidence_id"] for row in kwargs["additional_evidence"]] == ["corrected-result"]
        return {**claim, "verdict": "supported", "evidence_ids": ["corrected-result"]}, None

    monkeypatch.setattr("abstract_review.workflow.review_claim", recheck)
    assert _refresh_findings(report, object(), followups, None, tmp_path)
    assert report["findings"][0]["verdict"] == "supported"
    assert report["finding_history"][0]["verdict"] == "refuted"
    assert "candidate_proposal" not in report
    assert [row["status"] for row in report["revisions"]] == ["superseded", "passed"]
    assert not _refresh_findings(report, object(), followups, None, tmp_path)
