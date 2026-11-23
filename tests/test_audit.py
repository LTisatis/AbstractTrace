from contextlib import closing

import pytest

from abstract_review.audit import audit_version, extract_claims, review_claim, targeted_evidence
from abstract_review.evidence import EvidenceIndex
from abstract_review.models import ModelPaused


CLAIM = {
    "claim_id": "claim-1", "kind": "result", "text": "The prototype tracks biodiversity.",
    "quote": "The prototype tracks biodiversity.", "proposition": "The prototype tracks biodiversity.",
    "requirements": ["measurement definition"], "support_query": "biodiversity",
    "counter_query": "biodiversity exception",
}


def _source(eid, text, section="Results", page=0):
    return {"evidence_id": eid, "text": text, "doc_id": "paper", "kind": "body_text",
            "locator": {"section": section, "page_index": page}}


def _judgement(*, targets=(), queries=(), verdict="insufficient", observations=(), groups=(), missing=None):
    return {"verdict": verdict, "reason": "The report offers no measurement definition here.",
            "evidence_groups": list(groups), "missing_fields": CLAIM["requirements"] if missing is None else missing,
            "follow_up_queries": list(queries), "follow_up_targets": list(targets), "coverage": list(observations)}


class Model:
    def __init__(self, *results):
        self.results = iter(results)
        self.calls = []

    def call(self, stage, data, schema):
        assert stage == "judge"
        self.calls.append(data)
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return schema.model_validate(result)


def _methods_target():
    return {"query": "measurement definition", "requirement": "measurement definition", "section": "Methods"}


def test_ranked_search_cannot_mark_a_requirement_searched_missing():
    model = Model(_judgement(observations=[{"requirement": "measurement definition", "status": "searched_missing",
                                          "evidence_ids": ["e1"]}]))
    finding, _ = review_claim(CLAIM, EvidenceIndex([_source("e1", "Biodiversity changed.")]), model)
    assert finding["review_status"] == "pending" and finding["verdict"] is None
    assert finding["coverage"][0]["status"] == "unchecked"
    assert all(scope["mode"] == "ranked" for scope in finding["searched_scope"])


@pytest.mark.parametrize(("subject", "extras", "missing_scope", "expected"), [
    ("same", ["instrument calibration records"], False, "refuted"),
    ("different", ["instrument calibration records"], False, None),
    ("uncertain", ["instrument calibration records"], False, None),
    ("same", [], False, None),
    ("same", ["instrument calibration records"], True, None),
    ("same", ["measurement definition", "instrument calibration records"], True, None),
])
def test_direct_refutation_requires_same_scope_and_only_supplementary_missing_fields(
    subject, extras, missing_scope, expected,
):
    claim = {**CLAIM, "requirements": ["measurement definition", "instrument calibration records"]}
    groups = [{"role": "refute", "evidence_ids": ["opposite"]}]
    proof = {"evidence_ids": ["opposite"], "subject_and_conditions": subject,
             "relation": "opposite_result", "scope_reason": "The same prototype and study are identified.",
             "supplementary_fields": extras}
    checks = [] if missing_scope else [{"requirement": "measurement definition", "status": "found",
                                       "evidence_ids": ["opposite"]}]
    pending = ["instrument calibration records"]
    if missing_scope:
        pending.append("measurement definition")
    model = Model({**_judgement(verdict="refuted", groups=groups, observations=checks, missing=pending),
                   "direct_contradiction": proof})
    row = _source("opposite", "The prototype did not track biodiversity in the specified study.")
    with closing(EvidenceIndex([row])) as index:
        result, _ = review_claim(claim, index, model)
    assert result["verdict"] == expected
    assert ("direct_contradiction" in result) == (expected == "refuted")
    assert len(model.calls) == 1


@pytest.mark.parametrize(("proof_mode", "blocker"), [
    ("complete", None),
    ("missing", "missing_direct_contradiction"),
    ("different_experiment", "unconfirmed_contradiction_scope"),
    ("scope_in_another_group", "ungrounded_contradiction"),
    ("scope_not_sent", "ungrounded_contradiction"),
    ("scope_coverage_missing", "missing_scope_coverage"),
    ("condition_only_in_another_experiment", "missing_scope_coverage"),
    ("unknown_decisive_condition", "unchecked_decisive_requirement"),
    ("conflicting_support", "conflicting_support"),
    ("full_coverage_wrong_scope", "missing_scope_coverage"),
    ("full_coverage_conflicting_support", "conflicting_support"),
    ("complex_logic", "unresolved_sentence_logic"),
])
def test_direct_refutation_oracle_separates_scope_from_the_opposite_result(proof_mode, blocker):
    claim = {**CLAIM, "proposition": "Removing a filter from the coastal sensor suppressed noise.",
             "requirements": ["coastal sensor", "filter removal", "noise calibration"],
             "support_query": "coastal sensor filter", "counter_query": "filter removal increased noise"}
    sources = [
        _source("scope", "We ablated the filter in the coastal sensor arm."),
        _source("result", "It produced more, rather than less, measurement noise."),
        _source("other", "The dry-lab configuration had lower noise."),
    ]
    group_ids = ["scope", "result"]
    proof = {"evidence_ids": ["result"], "scope_evidence_ids": ["scope"],
             "subject_and_conditions": "same", "relation": "opposite_result",
             "scope_reason": "The ablation and the resulting increase concern the same coastal sensor arm.",
             "supplementary_fields": ["noise calibration"]}
    coverage = [
        {"requirement": field, "status": "found", "evidence_ids": ["scope"]}
        for field in ("coastal sensor", "filter removal")
    ]
    if proof_mode == "different_experiment":
        proof["subject_and_conditions"] = "different"
    elif proof_mode == "scope_in_another_group":
        group_ids = ["result"]
    elif proof_mode == "scope_not_sent":
        proof["scope_evidence_ids"] = ["unknown"]
    elif proof_mode in {"scope_coverage_missing", "full_coverage_wrong_scope"}:
        proof["scope_evidence_ids"] = []
    elif proof_mode == "unknown_decisive_condition":
        coverage.pop()
    elif proof_mode == "condition_only_in_another_experiment":
        coverage[1]["evidence_ids"] = ["other"]
    elif proof_mode == "complex_logic":
        claim["logic_review_required"] = True
    groups = [{"role": "refute", "evidence_ids": group_ids}]
    if proof_mode == "scope_in_another_group":
        groups.append({"role": "refute", "evidence_ids": ["scope"]})
    if proof_mode in {"conflicting_support", "full_coverage_conflicting_support"}:
        groups.append({"role": "support", "evidence_ids": ["other"]})
    missing = ["noise calibration"]
    if proof_mode.startswith("full_coverage"):
        coverage.append({"requirement": "noise calibration", "status": "found", "evidence_ids": ["other"]})
        proof["supplementary_fields"] = []
        missing = []
    model = Model({
        **_judgement(verdict="refuted", groups=groups, observations=coverage, missing=missing),
        "direct_contradiction": None if proof_mode == "missing" else proof,
        "reason": "Removing the filter raised noise in the coastal ablation.",
    })
    with closing(EvidenceIndex(sources)) as index:
        finding, _ = review_claim(claim, index, model)
    assert {"scope", "result"} <= set(finding["sent_candidate_ids"])
    assert len(model.calls) == 1
    assert finding.get("refutation_blocker") == blocker
    assert finding["verdict"] == ("refuted" if blocker is None else None)
    if blocker is None:
        assert finding["direct_contradiction"] == proof
        assert finding["coverage"][-1]["status"] == "unchecked"


@pytest.mark.parametrize("malformed_group", [False, True])
def test_extraction_recovers_a_procedural_conjunction_without_an_extra_model_call(malformed_group):
    text = "We applied an automated filter for screening and accepted every hit without manual review."
    group = {"sentence_id": "s1", "operator": "and", "claim_indexes": [0, 1]}

    class UnsplitExtractor:
        def call(self, stage, material, schema):
            assert stage == "extract"
            return schema.model_validate({"claims": [
                {"sentence_id": "s1", "quote": text, "proposition": text, "kind": "result",
                 "requirements": ["screening workflow"], "support_query": text, "counter_query": text},
            ], "compounds": [group] if malformed_group else []})

    left, right = extract_claims(text, UnsplitExtractor())
    assert right["quote"] == "accepted every hit without manual review"
    assert right["proposition"] == "We accepted every hit without manual review."
    assert right["support_query"] == right["proposition"]
    assert left["compound"] == right["compound"]
    assert "logic_review_required" not in right
    model = Model(_judgement(verdict="pending", missing=[]))
    with closing(EvidenceIndex([_source("e1", "The filter had hits.")])) as index:
        review_claim(right, index, model)
    assert model.calls[0]["claim"] == right["proposition"]
    assert model.calls[0]["sentence_context"] == text


def test_methods_hint_prefers_alias_in_current_document_without_claiming_complete_inspection():
    here = _source("methods", "The sample flags were checked by hand.", "Materials and methods", 1)
    other = _source("elsewhere", "The retrospective flags were never checked.", "Results", 2)
    foreign = {**here, "doc_id": "foreign", "evidence_id": "foreign"}
    with closing(EvidenceIndex([other, foreign, here])) as index:
        result = targeted_evidence(index, [{"query": "sample flags checked", "source_kind": "methods"}],
                                   documents=[{"doc_id": "paper", "role": "main"},
                                              {"doc_id": "foreign", "role": "supplement"}])
    assert result["evidence"][0]["evidence_id"] == "methods"
    assert not {row["evidence_id"] for row in result["evidence"]} & {"foreign"}
    direct, = [scope for scope in result["scopes"] if scope["mode"] == "direct"]
    assert direct["matching_count"] == 1 and direct["complete"] is False


def test_completing_named_section_can_mark_requirement_searched_missing():
    methods = _source("methods", "We visited known sites.", "2. Methods", 1)
    index = EvidenceIndex([_source("results", "Biodiversity changed."), methods])
    model = Model(_judgement(targets=[_methods_target()]), _judgement())
    reserved = []
    finding, _ = review_claim(CLAIM, index, model, reserve_followup=lambda: not reserved.append(True))

    assert reserved == [True]
    assert finding["review_status"] == "completed" and finding["verdict"] == "insufficient"
    assert finding["coverage"] == [{"requirement": "measurement definition", "status": "searched_missing",
                                    "evidence_ids": ["methods"], "checked_sources": [[
                                        {"doc_id": "paper", "page_index": 1, "section": "2. Methods"},
                                    ]], "unreadable_locations": []}]
    assert finding["stop_reason"] == "source_range_exhausted"
    assert model.calls[-1]["follow_up_already_performed"] is True
    assert [item["evidence_id"] for item in model.calls[-1]["evidence"]] == finding["sent_candidate_ids"]


def test_truncated_named_source_is_not_exhaustive_even_with_ranked_hits():
    rows = [_source(f"m{i}", f"Paragraph {i}", "Methods", 1) for i in range(10)]
    index = EvidenceIndex([*rows, _source("r", "Biodiversity shifted.")])
    model = Model(_judgement(targets=[_methods_target()]), _judgement())
    finding, _ = review_claim(CLAIM, index, model)
    assert finding["review_status"] == "pending"
    assert finding["coverage"][0]["status"] == "unchecked"
    assert model.calls[-1]["direct_inspections"][0]["matching_count"] == 10
    assert len(finding["sent_candidate_ids"]) == 8


def test_followup_ranks_named_methods_and_keeps_nearby_refutation_lines():
    rows = [_source(f"a{i:02}", f"Routine instrumentation note {i}", "Materials and methods", 1)
            for i in range(12)]
    lines = [
        ("z-condition", "For units submerged before each calibration,"),
        ("z-result", "the optical sensor saturated"),
        ("z-effect", "and did not produce usable readings."),
    ]
    for i, (eid, text) in enumerate(lines):
        row = _source(eid, text, "Materials and methods", 1)
        row["locator"]["bbox"] = [30, 230 + 12 * i, 170, 240 + 12 * i]
        rows.append(row)
    rows.append(_source("audit", "All overflow flags were retained in the case log.", "Appendix", 3))
    rows.append(_source("initial", "Biodiversity changed after prototype deployment."))
    targets = [
        {"query": "optical sensor saturated", "requirement": "measurement definition", "section": "Methods"},
        {"query": "overflow flags", "requirement": "measurement definition", "section": "Appendix"},
    ]
    refutation = [eid for eid, _ in lines]
    model = Model(_judgement(verdict="pending", targets=targets), _judgement(
        verdict="refuted", missing=[], groups=[{"role": "refute", "evidence_ids": refutation}],
        observations=[{"requirement": "measurement definition", "status": "found", "evidence_ids": refutation}],
    ))
    with closing(EvidenceIndex(rows)) as index:
        finding, _ = review_claim(CLAIM, index, model)

    assert finding["verdict"] == "refuted"
    assert set(refutation) | {"audit"} <= set(finding["sent_candidate_ids"])
    assert {"evidence_ids": refutation} in model.calls[-1]["body_passages"]
    assert set(refutation) <= set(finding["follow_up_evidence_ids"])
    first, second = model.calls[-1]["direct_inspections"]
    assert (first["matching_count"], first["complete"], second["complete"]) == (15, False, True)
    assert len(finding["sent_candidate_ids"]) == 8


def test_missing_named_section_falls_back_to_same_document_without_claiming_completion():
    evidence = _source("relevant", "The negative control saturated.", "Body", 2)
    outside = {**evidence, "doc_id": "unrelated", "evidence_id": "outside"}
    with closing(EvidenceIndex([evidence, outside])) as index:
        received = targeted_evidence(index, [
            {"query": "negative control saturated", "section": "Results", "doc_id": "paper"},
        ])
    assert [row["evidence_id"] for row in received["evidence"]] == ["relevant"]
    direct, = [scope for scope in received["scopes"] if scope["mode"] == "direct"]
    assert direct["matching_count"] == 0 and direct["complete"] is False


def test_partial_followup_keeps_initial_result_in_the_second_judge_packet():
    rows = [_source(f"m{i}", f"Measurement definition note {i}", "Methods", 1) for i in range(12)]
    result = _source("result", "Biodiversity changed.")
    model = Model(_judgement(targets=[_methods_target()]), _judgement())
    with closing(EvidenceIndex([result, *rows])) as index:
        finding, _ = review_claim(CLAIM, index, model)
    assert finding["review_status"] == "pending"
    assert "result" in finding["sent_candidate_ids"] and len(finding["sent_candidate_ids"]) == 8


def test_unqualified_page_scope_accounts_for_unreadable_attached_documents():
    docs = [
        {"doc_id": "paper", "pages": [{"page_index": 0, "status": "parsed"}]},
        {"doc_id": "appendix", "pages": [{"page_index": 0, "status": "unreadable"}]},
    ]
    with closing(EvidenceIndex([_source("e1", "Biodiversity changed.")])) as index:
        result = targeted_evidence(index, [{"query": "biodiversity", "page_index": 0}], documents=docs)
    direct, = [scope for scope in result["scopes"] if scope["mode"] == "direct"]
    assert direct["complete"] is False
    assert direct["unreadable_locations"] == [{"doc_id": "appendix", "page_index": 0, "status": "unreadable"}]


def test_unreadable_targeted_page_is_not_proof_of_missing_evidence():
    index = EvidenceIndex([_source("result", "Biodiversity shifted.")])
    target = {"query": "page 2", "page_index": 1, "doc_id": "paper", "requirement": "measurement definition"}
    model = Model(_judgement(targets=[target]))
    document = {"doc_id": "paper", "pages": [{"page_index": 0, "status": "parsed"},
                                             {"page_index": 1, "status": "unreadable"}]}
    finding, _ = review_claim(CLAIM, index, model, documents=[document])

    assert finding["coverage"][0]["status"] == "unreadable"
    assert finding["coverage"][0]["unreadable_locations"] == [
        {"doc_id": "paper", "page_index": 1, "status": "unreadable"},
    ]
    assert finding["verdict"] is None and finding["stop_reason"] == "unreadable"
    assert len(model.calls) == 1


def test_scope_is_scoped_by_document_and_limits_supplement_queries():
    index = EvidenceIndex([_source("anchor", "A source.", "Methods"), _source("r", "Results about biodiversity.")])
    files = [{"doc_id": "paper", "pages": [{"page_index": 0, "status": "parsed"}]}]
    actual = targeted_evidence(index, [{"query": "outside", "doc_id": "foreign", "page_index": 0},
                                       "Methods", "biodiversity", "Results"], documents=files)
    assert [scope["query"] for scope in actual["scopes"]] == ["Methods", "biodiversity"]
    assert [scope["mode"] for scope in actual["scopes"]] == ["direct", "ranked"]
    assert [row["evidence_id"] for row in actual["evidence"]] == ["anchor", "r"]


def test_only_real_scope_sources_can_fill_a_required_field():
    index = EvidenceIndex([_source("original", "Biodiversity changed."), _source("extra", "Original source text.")])
    model = Model(_judgement(verdict="supported", missing=[], groups=[{"role": "support", "evidence_ids": ["extra"]}],
                             observations=[{"requirement": "measurement definition", "status": "found",
                                            "evidence_ids": ["extra"]}]))
    finding, _ = review_claim(CLAIM, index, model, additional_evidence=[{"evidence_id": "extra", "text": "fake"}])
    assert finding["coverage"][0]["status"] == "found"
    assert model.calls[0]["evidence"][0]["text"] == "Original source text."
    assert finding["candidate_ids"][0] == "extra"


def test_positive_verdict_with_unchecked_required_field_stays_pending():
    index = EvidenceIndex([_source("result", "The prototype tracks biodiversity.")])
    model = Model(_judgement(verdict="supported", groups=[{"role": "support", "evidence_ids": ["result"]}],
                             observations=[{"requirement": "measurement definition", "status": "unchecked",
                                            "evidence_ids": []}]))
    finding, _ = review_claim(CLAIM, index, model)
    assert finding["review_status"] == "pending"
    assert finding["verdict"] is None


def test_retry_reuses_saved_target_without_reserving_a_second_round():
    report = {"reviewed_abstract": {"text": CLAIM["text"]}, "claims": [CLAIM], "findings": [],
              "evidence": [_source("r", "Biodiversity changed."), _source("m", "Visited sites.", "Methods")],
              "documents": [], "calculations": []}
    reserved = []
    persisted = []

    def mark_reservation(claim_id):
        reserved.append(claim_id)
        return len(reserved) == 1

    def checkpoint():
        persisted.append(list(report["claim_followups"]))

    first = Model(_judgement(targets=[_methods_target()]), ModelPaused("interrupted"))
    audit_version(report, first, checkpoint, reserve_followup=mark_reservation)
    assert report["findings"][0]["retryable"] is True
    assert persisted[0] == [CLAIM["claim_id"]]
    assert report["claim_followups"][CLAIM["claim_id"]] == [_methods_target()]

    second = Model(_judgement())
    audit_version(report, second, checkpoint, reserve_followup=mark_reservation)
    assert reserved == [CLAIM["claim_id"]]
    assert len(report["findings"]) == 1 and report["findings"][0]["verdict"] == "insufficient"
    assert second.calls[0]["follow_up_already_performed"] is True


def test_disallowed_supplement_leaves_claim_pending_without_a_second_judgement():
    index = EvidenceIndex([_source("m", "Visited sites.", "Methods")])
    model = Model(_judgement(targets=[_methods_target()]))
    finding, _ = review_claim(CLAIM, index, model, reserve_followup=lambda: False)
    assert finding["stop_reason"] == "follow_up_budget"
    assert finding["verdict"] is None and len(model.calls) == 1
