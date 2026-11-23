from contextlib import closing
from types import SimpleNamespace

import pytest

from abstract_review.audit import audit_version, extract_claims, recover_compound_claim, review_claim, sentences
from abstract_review.compound import aggregate_compound_findings
from abstract_review.evidence import EvidenceIndex


PROCEDURE = "We applied an automated detector for screening in all included records and accepted every flag without manual review."
THEOREM = ("Across both soft and hard attention, the model cannot recognize mirrored strings, nor balanced markers, "
           "unless the number of layers or heads in-creases with input length.")


def _evidence(eid, text):
    return {"evidence_id": eid, "text": text, "doc_id": "paper", "kind": "body_text",
            "locator": {"section": "Results", "page_index": 0}}


@pytest.mark.parametrize("omitted", [True, False])
def test_recovers_grounded_procedure_when_the_model_omits_all_or_part(omitted):
    def call(stage, material, schema):
        assert stage == "extract"
        left = PROCEDURE.split(" and accepted")[0]
        claim = {"sentence_id": "s1", "quote": left, "proposition": left, "kind": "result",
                 "requirements": ["screening setting"], "support_query": left, "counter_query": left}
        return schema.model_validate({"claims": [] if omitted else [claim]})

    model = SimpleNamespace(call=call)
    recovered = extract_claims(PROCEDURE, model)
    assert [{key: value for key, value in claim.items() if key != "extraction_guard"} for claim in recovered] == (
        extract_claims(PROCEDURE)
    )
    if not omitted:
        assert all(claim["extraction_guard"]["status"] == "source_fallback" for claim in recovered)


@pytest.mark.parametrize("condition", ["either memory grows with sequence length or inputs are restricted",
                                       "memory grows with sequence length or batch size"])
def test_shared_exception_with_nested_condition_stays_guarded(condition):
    source = f"The finite-state controller cannot recognize mirrored tokens, nor balanced markers, unless {condition}."
    claims = extract_claims(source)
    assert len(claims) == 2
    assert all(claim["logic_review_required"] for claim in claims)


@pytest.mark.parametrize("check_exception", [False, True])
def test_scoped_counterexample_can_refute_only_a_grounded_exception_operand(check_exception):
    claims = extract_claims(THEOREM)
    first, second = claims
    assert first["compound"]["operator"] == "and"
    assert "unless the number of layers or heads in-creases" in first["proposition"]
    rows = [_evidence("example", "Hard attention recognized mirrored strings for every input length."),
            _evidence("scope", "The mirrored-string construction used one layer and one head at all lengths.")]

    def call(stage, material, schema):
        assert stage == "judge" and material["claim"] == first["proposition"]
        assert material["sentence_context"] == THEOREM
        assert not material.get("logic_review_required")
        required, exception = material["requirements"]
        assert exception.startswith("scope in which the exception does not hold")
        return schema.model_validate({
            "verdict": "refuted", "reason": "A construction in the stated domain contradicts the limitation.",
            "evidence_groups": [{"role": "refute", "evidence_ids": ["example", "scope"]}], "missing_fields": [],
            "follow_up_queries": [], "direct_contradiction": {"evidence_ids": ["example"],
                "scope_evidence_ids": ["scope"], "subject_and_conditions": "same",
                "relation": "in_scope_counterexample", "scope_reason": "Fixed hard-attention architecture.",
                "supplementary_fields": []},
            "coverage": [{"requirement": required, "status": "found", "evidence_ids": ["example"]},
                         {"requirement": exception, "status": "found" if check_exception else "unchecked",
                          "evidence_ids": ["scope"] if check_exception else []}],
        })

    model = SimpleNamespace(call=call)
    with closing(EvidenceIndex(rows)) as index:
        finding, _ = review_claim(first, index, model)
    rollup, = aggregate_compound_findings(claims, [finding])
    assert rollup["verdict"] == finding["verdict"] == ("refuted" if check_exception else None)
    assert rollup["members"][1]["claim_id"] == second["claim_id"]
    assert rollup["members"][1]["review_status"] == "pending"
    if not check_exception:
        assert finding["refutation_blocker"] == "unchecked_decisive_requirement"


def _legacy_state(kind):
    source = f"We published our protocol. {PROCEDURE}"
    _, part = sentences(source)
    first, *_ = extract_claims(source)
    whole = {"claim_id": "legacy", "kind": "result", "sentence_id": part["sentence_id"], "text": part["text"],
             "start": part["start"], "end": part["end"], "sentence_start": part["start"], "sentence_end": part["end"],
             "quote": part["text"], "proposition": part["text"], "logic_review_required": True,
             "requirements": ["observed result and scope"], "support_query": "flags", "counter_query": "manual review"}
    version = {"text": source, "kind": kind, "version_id": "original"}
    settled = {"claim_id": first["claim_id"], "review_status": "completed", "verdict": "supported", "evidence_ids": []}
    report = {"original_abstract": version, "reviewed_abstract": version, "claims": [first, whole], "findings": [settled],
              "calculations": [], "evidence": [_evidence("manual", "We checked every flagged record manually.")]}
    return report, whole, settled


def test_cached_whole_claim_recovery_checks_offsets_and_resumes_idempotently():
    report, legacy, settled = _legacy_state("pasted_original")
    source = report["original_abstract"]["text"]
    expected = recover_compound_claim(legacy, source)
    assert len(expected) == 2
    assert recover_compound_claim({**legacy, "start": legacy["start"] + 1}, source) == []
    snapshots = []
    audit_version(report, None, lambda: snapshots.append(tuple(claim["claim_id"] for claim in report["claims"])))
    assert report["claims"][1:] == expected
    assert report["findings"][0] is settled
    assert snapshots[0] == tuple(row["claim_id"] for row in report["claims"])
    assert [row["claim_id"] for row in report["findings"]] == list(snapshots[0])
    snapshots.clear()
    audit_version(report, None, lambda: snapshots.append(True))
    assert snapshots == []


def test_cached_exception_operand_upgrade_preserves_its_identity_and_validates_group():
    current, _ = extract_claims(THEOREM)
    cached = {**current, "logic_review_required": True, "requirements": current["requirements"][:1]}
    assert recover_compound_claim(cached, THEOREM) == [current]
    invalid_group = {**cached, "compound": {**cached["compound"], "operator": "or"}}
    assert recover_compound_claim(invalid_group, THEOREM) == []


@pytest.mark.parametrize("scope", ["provided_original", "settled", "followup_reserved"])
def test_cached_migration_preserves_supplied_and_processed_whole_claim_ids(scope):
    report, whole, _ = _legacy_state("provided_original" if scope == "provided_original" else "pasted_original")
    if scope == "settled":
        report["findings"].append({"claim_id": whole["claim_id"], "review_status": "completed", "verdict": "refuted",
                                   "reason": "Historical finding", "evidence_ids": []})
    if scope == "followup_reserved":
        report["follow_up_rounds"] = {whole["claim_id"]: {"stage": "judge"}}
    audit_version(report, None, lambda: None)
    assert report["claims"][1] is whole
    assert report["findings"][1]["claim_id"] == whole["claim_id"]
