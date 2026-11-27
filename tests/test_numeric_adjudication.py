from contextlib import closing

import pytest

from abstract_review.audit import review_claim, supplied_claims
from abstract_review.calculations import check_revision_numbers
from abstract_review.evidence import EvidenceIndex
from abstract_review.revisions import build_candidate, new_version


def _cells(baseline="80%", result="88%", *, page=1, prefix=""):
    return [
        {"evidence_id": f"{prefix}{eid}", "doc_id": "paper", "kind": "table_cell", "text": value,
         "row_header": "Dataset A", "column_header": heading,
         "locator": {"page_index": page, "table_index": 0, "row_index": 1, "column_index": column}}
        for eid, value, heading, column in [("baseline", baseline, "Baseline accuracy", 1),
                                              ("final", result, "Final accuracy", 2)]
    ]


class _Judge:
    def __init__(self, verdict):
        self.verdict = verdict
        self.material = None

    def call(self, stage, material, schema):
        assert stage == "judge"
        self.material = material
        ids = material["calculation"]["evidence_ids"]
        role = "support" if self.verdict == "supported" else "refute"
        response = {"verdict": self.verdict, "reason": "The cited table gives the comparison.",
                    "evidence_groups": [{"role": role, "evidence_ids": ids}], "missing_fields": [],
                    "follow_up_queries": [],
                    "coverage": [{"requirement": requirement, "status": "found", "evidence_ids": ids}
                                 for requirement in material["requirements"]]}
        if role == "support":
            response["empirical_support"] = {"result_source": "direct_result", "result_evidence_ids": ids,
                                             "scope_evidence_ids": ids, "same_study_and_scope": "same"}
        else:
            response["direct_contradiction"] = {"evidence_ids": ids, "subject_and_conditions": "same",
                                                "relation": "opposite_result", "scope_reason": "Same table row.",
                                                "supplementary_fields": []}
        return schema.model_validate(response)


@pytest.mark.parametrize(("stated", "verdict", "expected"), [
    ("12", "supported", None), ("8", "refuted", None),
    ("8", "supported", "supported"), ("12", "refuted", "refuted"),
])
def test_online_verdict_must_agree_with_the_independently_computed_table_comparison(stated, verdict, expected):
    claim, = supplied_claims((("claim", f"Accuracy increased by {stated} percentage points on Dataset A."),))
    model = _Judge(verdict)
    with closing(EvidenceIndex(_cells())) as index:
        finding, calculation = review_claim(claim, index, model)
    assert calculation["result"] == "8" and calculation["unit"] == "percentage_points"
    assert finding["verdict"] == expected
    assert finding["review_status"] == ("pending" if expected is None else "completed")
    assert finding["evidence_ids"] == ["baseline", "final"]
    assert finding["decision_trace"]["attempts"][0]["proposed_verdict"] == verdict
    assert ("calculation_disagrees" in finding["decision_trace"]["blocker_codes"]) == (expected is None)


def test_original_and_revision_use_the_same_reported_precision_for_relative_change():
    text = "Accuracy increased by 16.7% on Dataset A."
    claim, = supplied_claims((("claim", text),))
    rows = _cells("60%", "70%")
    revised, = check_revision_numbers(text, text, rows)
    with closing(EvidenceIndex(rows)) as index:
        finding, calculation = review_claim(claim, index)
    assert calculation == revised
    assert (calculation["rounded_result"], calculation["unit"], calculation["verification"]) == ("16.7", "percent", "matched")
    assert finding["verdict"] == "supported"


def test_numeric_revision_preserves_the_claimed_decimal_precision():
    text = "Accuracy increased by 17.0% on Dataset A."
    claim, = supplied_claims((("claim", text),))
    rows = _cells("60%", "70%")
    with closing(EvidenceIndex(rows)) as index:
        finding, calculation = review_claim(claim, index)
    assert calculation["verification"] == "mismatch"
    report = {"original_abstract": new_version(text, "pdf_original", "", []),
              "findings": [finding], "evidence": rows}
    candidate = build_candidate(report)
    assert candidate["text"] == text.replace("17.0%", "16.7%")
    verification, = report["verifications"]
    assert verification["calculations"][0]["verification"] == "matched"


def test_named_experiment_requires_a_link_to_the_operand_table():
    text = "In Experiment E1, accuracy increased by 8 percentage points on Dataset A."
    cells = [{**row, "caption_evidence_id": "caption"} for row in _cells()]
    caption = {"evidence_id": "caption", "doc_id": "paper", "kind": "table_caption",
               "text": "Table 1. Results for Experiment E2.", "locator": {"page_index": 1, "table_index": 0}}
    unresolved, = check_revision_numbers(text, text, [*cells, caption])
    assert unresolved["verification"] == "unresolved"
    caption["text"] = "Table 1. Results for Experiment E1."
    matched, = check_revision_numbers(text, text, [*cells, caption])
    assert matched["verification"] == "matched"
    assert matched["scope_evidence_ids"] == ["caption"]


@pytest.mark.parametrize(("text", "rows"), [
    ("Accuracy increased by 8 percentage points on Dataset A.", _cells() + _cells(page=2, prefix="other-")),
    ("Accuracy increased by 10% on Dataset A.", _cells("0%", "10%")),
    ("Accuracy did not increase by 12 percentage points on Dataset A.", _cells()),
    ("Accuracy increased by at least 7 percentage points on Dataset A.", _cells()),
    ("Accuracy increased by approximately 8 percentage points on Dataset A.", _cells()),
])
def test_ambiguous_undefined_or_nonexact_comparisons_do_not_produce_a_deterministic_verdict(text, rows):
    claim, = supplied_claims((("claim", text),))
    with closing(EvidenceIndex(rows)) as index:
        finding, calculation = review_claim(claim, index)
    assert calculation is None
    assert finding["review_status"] == "pending"


def test_calculation_can_use_operands_from_the_final_packet(monkeypatch):
    text = "Accuracy increased by 8 percentage points on Dataset A."
    claim, = supplied_claims((("claim", text),))
    source = {"evidence_id": "source", "doc_id": "paper", "kind": "body_text",
              "text": "Accuracy on Dataset A is described in Table 1.", "locator": {"page_index": 1}}
    model = _Judge("supported")
    with closing(EvidenceIndex([*_cells(), source])) as index:
        search = index.search

        def primary_query(*args, **kwargs):
            result = search(*args, **kwargs)
            result["evidence"] = [row for row in result["evidence"] if row["evidence_id"] != "final"]
            return result

        monkeypatch.setattr(index, "search", primary_query)
        monkeypatch.setattr(index, "referenced_table_region", lambda *_: {
            "reference_evidence_id": "source", "evidence": index.get(["final"]),
        })
        finding, calculation = review_claim(claim, index, model, documents=[{"doc_id": "paper", "role": "main"}])
    assert "final" not in {row["evidence_id"] for row in model.material["evidence"]}
    assert "final" in finding["sent_candidate_ids"]
    assert calculation["evidence_ids"] == ["baseline", "final"]
    assert finding["verdict"] == "supported"
