from contextlib import closing

import pytest

from abstract_review.audit import audit_version, extract_claims, review_claim, sentences, targeted_evidence
from abstract_review.evidence import EvidenceIndex
from abstract_review.models import InvalidJudgeReference, ModelPaused


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


def test_offline_verbatim_quote_is_a_candidate_even_when_context_disputes_it():
    rows = [_source("quote", CLAIM["quote"], "Related Work"),
            _source("result", "Our primary trial found no biodiversity tracking effect.")]
    with closing(EvidenceIndex(rows)) as index:
        finding, _ = review_claim(CLAIM, index)
    assert finding["review_status"] == "pending" and finding["verdict"] is None
    assert finding["verbatim_candidate_ids"] == ["quote"] and finding["evidence_ids"] == []
    assert {"quote", "result"} <= set(finding["sent_candidate_ids"])


@pytest.mark.parametrize(("verdict", "roles", "blocker"), [
    ("supported", ["support", "refute"], "opposing_evidence"),
    ("refuted", ["refute"], "missing_direct_contradiction"),
])
def test_fulltext_does_not_finalize_opposing_or_unproved_judgements(verdict, roles, blocker):
    rows = [_source("support", "Biodiversity increased on the tested plot."),
            _source("refute", "Biodiversity decreased on the tested plot.")]
    output = _judgement(verdict=verdict, missing=[], groups=[{"role": role, "evidence_ids": [role]} for role in roles],
                        observations=[{"requirement": CLAIM["requirements"][0], "status": "found", "evidence_ids": roles}])
    with closing(EvidenceIndex(rows)) as index:
        finding, _ = review_claim(CLAIM, index, Model(output), documents=[{"doc_id": "paper", "role": "main"}])
    assert finding["review_status"] == "pending" and finding["verdict"] is None
    assert finding["evidence_ids"] == roles
    assert blocker in finding["decision_trace"]["blocker_codes"]
    assert finding["decision_trace"]["attempts"][0]["proposed_verdict"] == verdict


def test_trace_distinguishes_wire_references_from_citation_checks_and_budget():
    rows = [_source("known", "Biodiversity was measured twice.")]
    output = _judgement(verdict="pending", groups=[{"role": "limit", "evidence_ids": ["invented"]}])
    with closing(EvidenceIndex(rows)) as index:
        finding, _ = review_claim(CLAIM, index, Model(InvalidJudgeReference("invalid alias"), output))
    assert finding["stop_reason"] == "invalid_judge_reference"
    attempts = finding["decision_trace"]["attempts"]
    assert [(a["status"], a["proposed_verdict"], a["repair"]) for a in attempts] == [
        ("model_reference_error", None, False), ("citation_error", "pending", True),
    ]
    assert attempts[0]["material_sha256"] != attempts[1]["material_sha256"]
    with closing(EvidenceIndex(rows)) as index:
        finding, _ = review_claim(CLAIM, index, Model(InvalidJudgeReference("invalid alias"), ModelPaused("model budget exhausted")))
    assert finding["stop_reason"] == "budget_exhausted" and finding["retryable"]
    assert finding["decision_trace"]["attempts"][-1]["status"] == "model_paused"


def test_academic_abbreviations_preserve_citation_offsets_and_later_sentence_ids():
    abstract = "  We replicated the analysis by (Lee et al. 2011).\n\n We obtained 3.14 units (see Fig. 2). Next result."
    parts = sentences(abstract)
    assert [part["sentence_id"] for part in parts] == ["s1", "s3", "s5"]
    assert [part["text"] for part in parts] == [
        "We replicated the analysis by (Lee et al. 2011).",
        "We obtained 3.14 units (see Fig. 2).",
        "Next result.",
    ]
    assert all(abstract[part["start"]:part["end"]] == part["text"] for part in parts)
    assert extract_claims(abstract)[0]["proposition"] == parts[0]["text"]


def test_nonterminal_abbreviations_do_not_suppress_actual_sentence_ends():
    abstract = "A result from Lee et al. We replicated Lee et al. (2021), e.g. their first trial. Next result."
    assert [part["text"] for part in sentences(abstract)] == [
        "A result from Lee et al.",
        "We replicated Lee et al. (2021), e.g. their first trial.",
        "Next result.",
    ]


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


@pytest.mark.parametrize(("field", "proposition", "condition_text"), [
    ("human patient population", "The intervention lowered recurrence in human patients.",
     "The study used mice and enrolled no human patients."),
    ("held-out test split", "The prototype improved detection on the held-out test split.",
     "Detection was measured only on the training split."),
    ("instrument calibration records", "With the specified instrument calibration, the prototype reduced drift.",
     "The experiment did not record instrument calibration."),
])
def test_refutation_cannot_waive_decisive_conditions(field, proposition, condition_text):
    claim = {**CLAIM, "proposition": proposition, "requirements": [field, "outcome"],
             "support_query": condition_text, "counter_query": "opposite tested outcome"}
    output = _judgement(verdict="refuted", missing=[field], groups=[{"role": "refute", "evidence_ids": ["scope", "opposite"]}],
                        observations=[{"requirement": "outcome", "status": "found", "evidence_ids": ["opposite"]}])
    output["direct_contradiction"] = {
        "evidence_ids": ["opposite"], "scope_evidence_ids": ["scope"], "subject_and_conditions": "same",
        "relation": "opposite_result", "scope_reason": "An opposite result was reported.", "supplementary_fields": [field],
    }
    model = Model(output)
    with closing(EvidenceIndex([_source("scope", condition_text), _source("opposite", "The opposite tested outcome occurred.")])) as index:
        result, _ = review_claim(claim, index, model)
    assert model.calls[0]["optional_refutation_requirements"] == []
    assert result["review_status"] == "pending" and result["verdict"] is None
    assert result["refutation_blocker"] == "unapproved_supplementary_requirement"
    assert result["evidence_ids"] == ["scope", "opposite"]


@pytest.mark.parametrize(("verdict", "direction", "blocker"), [
    ("supported", "support", "unlinked_support_requirements"),
    ("refuted", "refute", "missing_scope_coverage"),
])
def test_decisive_scope_must_be_linked_to_the_directional_evidence(verdict, direction, blocker):
    claim = {**CLAIM, "requirements": ["test split", "biodiversity outcome"],
             "support_query": "biodiversity result test split", "counter_query": "test split"}
    observed = _judgement(verdict=verdict, missing=[], groups=[{"role": direction, "evidence_ids": ["result"]}], observations=[
        {"requirement": "biodiversity outcome", "status": "found", "evidence_ids": ["result"]},
        {"requirement": "test split", "status": "found", "evidence_ids": ["other"]},
    ])
    if verdict == "refuted":
        observed["direct_contradiction"] = {
            "evidence_ids": ["result"], "subject_and_conditions": "same", "relation": "opposite_result",
            "scope_reason": "The prototype was the same.", "supplementary_fields": [],
        }
    rows = [_source("result", "The biodiversity result was measured in the development run."),
            _source("other", "The test split was used in a different experiment.")]
    with closing(EvidenceIndex(rows)) as index:
        result, _ = review_claim(claim, index, Model(observed))
    assert {"result", "other"} <= set(result["sent_candidate_ids"])
    assert result["review_status"] == "pending" and result["verdict"] is None
    assert blocker in result["decision_trace"]["blocker_codes"]


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


def test_judge_can_cite_adjacent_body_context_within_original_source():
    claim = {**CLAIM, "support_query": "coastal signal calibration", "counter_query": "coastal signal calibration",
             "requirements": ["measurement definition"]}
    rows = []
    for number in range(14):
        text = "Routine operation was recorded." if number else "Coastal signal calibration."
        if number == 12:
            text = "The target was a validated pulse."
        row = _source(f"line-{number}", text, "Methods")
        row["locator"]["bbox"] = [20, 50 + number * 12, 190, 60 + number * 12]
        rows.append(row)
    rows.append({**rows[-1], "doc_id": "other", "evidence_id": "out-of-scope"})
    model = Model(_judgement(
        verdict="supported", missing=[],
        observations=[{"requirement": "measurement definition", "status": "found", "evidence_ids": ["line-12"]}],
        groups=[{"role": "support", "evidence_ids": ["line-12"]}],
    ))
    with closing(EvidenceIndex(rows)) as index:
        finding, _ = review_claim(claim, index, model)
    assert len(model.calls) == 1
    packet, = model.calls[0]["body_context"]
    assert packet["doc_id"] == "paper" and packet["page_index"] == 0
    assert [item[0] for item in packet["lines"]] == [f"line-{number}" for number in range(14)]
    assert ["line-12", "The target was a validated pulse."] in packet["lines"]
    assert "line-12" not in finding["candidate_ids"]
    assert "line-12" in finding["sent_candidate_ids"]
    assert "out-of-scope" not in finding["sent_candidate_ids"]
    assert finding["verdict"] == "supported"


def test_ranked_body_context_reserves_both_query_cores_and_obeys_page_scope():
    rows = []
    for page, topic in enumerate(("coastal calibration", "thermal saturation")):
        for number in range(26):
            row = _source(f"p{page}-{number}", topic if number == 8 else f"Routine observation {number}.",
                          page=page)
            row["locator"]["bbox"] = [20, 50 + number * 12, 190, 60 + number * 12]
            rows.append(row)
    with closing(EvidenceIndex(rows)) as index:
        ranges = index.ranked_body_context(["coastal calibration", "thermal saturation"])
        supported, counter = index.pack_body_context(ranges, [])
        restricted = index.ranked_body_context(["thermal saturation", "coastal calibration"], page_index=1)
        restricted_groups = index.pack_body_context(restricted, [])
    assert all(eid.startswith("p0-") for eid in supported["evidence_ids"])
    assert all(eid.startswith("p1-") for eid in counter["evidence_ids"])
    assert "p0-8" in supported["new_evidence_ids"]
    assert "p1-8" in counter["new_evidence_ids"]
    assert len(supported["new_evidence_ids"] + counter["new_evidence_ids"]) <= 24
    assert restricted_groups and all(eid.startswith("p1-") for group in restricted_groups for eid in group["evidence_ids"])


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


def test_primary_body_anchor_adds_its_own_source_range_to_the_review_packet():
    rows = []
    for page in range(2):
        for number in range(5):
            text = f"{'Wetland observations' if page else 'Calibration details'} were recorded at site {number}."
            row = _source(f"p{page}-{number}", text, "Methods", page)
            row["locator"]["bbox"] = [20, 50 + number * 12, 190, 60 + number * 12]
            rows.append(row)
    with closing(EvidenceIndex(rows)) as index:
        ranked = index.ranked_body_context(["wetland observations"], page_index=1)
        anchor = index.anchored_body_context(["p0-2"], ranked)
        groups = index.pack_body_context([anchor, *ranked], ["p0-2"])
    assert groups[0]["evidence_ids"] == [f"p0-{number}" for number in range(5)]
    assert "p0-2" not in groups[0]["new_evidence_ids"]
    assert groups[1]["evidence_ids"] == [f"p1-{number}" for number in range(5)]


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
    reviewed = _judgement(
        verdict="refuted", missing=[], groups=[{"role": "refute", "evidence_ids": refutation}],
        observations=[{"requirement": "measurement definition", "status": "found", "evidence_ids": refutation}],
    )
    reviewed["direct_contradiction"] = {"evidence_ids": refutation, "subject_and_conditions": "same",
                                         "relation": "opposite_result", "scope_reason": "Same prototype and condition.",
                                         "supplementary_fields": []}
    model = Model(_judgement(verdict="pending", targets=targets), reviewed)
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


@pytest.mark.parametrize("legacy_query", [True, False])
def test_repeated_unscoped_followup_keeps_the_initial_primary_and_body_packet(legacy_query):
    rows = []
    for number in range(14):
        text = "Biodiversity was measured in the main trial." if number == 0 else "The main trial used a fixed pulse."
        row = _source(f"main-{number}", text, "Methods")
        row["locator"]["bbox"] = [20, 50 + number * 12, 190, 60 + number * 12]
        rows.append(row)
    rows.extend(_source(f"near-match-{number}", "Biodiversity note.", "Discussion", 1) for number in range(12))
    query = "  BIODIVERSITY  "
    request = {"queries": [query]} if legacy_query else {"targets": [{"query": query, "doc_id": "paper"}]}
    model = Model(_judgement(verdict="pending", **request))
    reserved, saved = [], []
    with closing(EvidenceIndex(rows)) as index:
        finding, _ = review_claim(CLAIM, index, model, documents=[{"doc_id": "paper"}],
                                  reserve_followup=lambda: reserved.append(True) or True, save_targets=saved.append)
    packet, = model.calls
    primary = [row["evidence_id"] for row in packet["evidence"]]
    body = [line[0] for window in packet["body_context"] for line in window["lines"] if len(line) == 2]
    assert "main-12" in body
    assert finding["candidate_ids"] == primary
    assert finding["sent_candidate_ids"] == [*primary, *body]
    assert finding["stop_reason"] == "no_progress"
    assert not finding["follow_up_performed"] and not reserved and not saved


def test_duplicate_followup_queries_leave_room_for_a_same_query_named_proof():
    rows = [_source("prior", "Biodiversity changed."), _source("proof", "The construction is stated here.", "Methods", 1)]
    target = {"query": CLAIM["support_query"], "doc_id": "paper", "section": "Methods", "source_kind": "proof",
              "requirement": "measurement definition"}
    model = Model(_judgement(targets=[{"query": CLAIM["support_query"]},
                                       {"query": CLAIM["counter_query"], "doc_id": "paper"}, target]),
                  _judgement(verdict="pending"))
    saved = []
    with closing(EvidenceIndex(rows)) as index:
        finding, _ = review_claim(CLAIM, index, model, documents=[{"doc_id": "paper"}], save_targets=saved.append)
    assert len(model.calls) == 2 and len(saved) == 1
    assert saved[0][0]["targets"] == [target]
    assert model.calls[1]["direct_inspections"][0]["section"] == "Methods"
    assert "proof" in finding["sent_candidate_ids"]
    assert finding["follow_up_performed"]


@pytest.mark.parametrize("narrow_document", [True, False])
def test_meaningful_document_narrowing_or_query_refinement_remains_actionable(narrow_document):
    rows = [_source(f"p{number}", "Biodiversity changed as measured.") for number in range(10)]
    if narrow_document:
        rows.append({**_source("supplement", "Biodiversity changed as measured."), "doc_id": "appendix"})
        target = {"query": CLAIM["support_query"], "doc_id": "appendix"}
    else:
        rows.append(_source("comparator", "The comparator used a calibrated pulse."))
        target = {"query": f"{CLAIM['support_query']} comparator"}
    model = Model(_judgement(targets=[target]), _judgement(verdict="pending"))
    with closing(EvidenceIndex(rows)) as index:
        finding, _ = review_claim(CLAIM, index, model, documents=[{"doc_id": "paper"}, {"doc_id": "appendix"}])
    assert len(model.calls) == 2
    assert ("supplement" if narrow_document else "comparator") in finding["sent_candidate_ids"]


def test_printed_heading_hint_reaches_the_physical_page_when_the_parser_section_is_stale():
    heading = "4.7 Biodiversity pulse comparator"
    evidence_id = "actual-threshold"
    evidence_page = 9
    pages = [{"page_index": page, "status": "parsed", "source_headings": [f"Protocol note {page}"]}
             for page in range(evidence_page)]
    pages[1]["source_headings"] = "invalid legacy metadata"
    pages.append({"page_index": evidence_page, "status": "parsed", "source_headings": [
        " ", "x" * 101, None, "Overview", "  4.7  Biodiversity pulse comparator  ",
        heading, "4.8 Prototype threshold", "Appendix note",
    ]})
    report = {
        "claims": [CLAIM], "findings": [], "calculations": [], "reviewed_abstract": {"text": CLAIM["text"]},
        "documents": [{"doc_id": "paper", "role": "main", "pages": pages}, {"doc_id": "legacy"}],
        "evidence": [
            _source("first-result", "Biodiversity changed.", "Results"),
            _source("misleading-methods", "The simulation used an uncalibrated sensor.", "Methods", 2),
            _source(evidence_id, "All field sites used a calibrated pulse threshold.", "Data processing", evidence_page),
        ],
    }

    class Navigator:
        def __init__(self):
            self.calls = []

        def call(self, stage, material, schema):
            assert stage == "judge"
            self.calls.append(material)
            if len(self.calls) == 1:
                assert evidence_id not in {item["evidence_id"] for item in material["evidence"]}
                hints = material["source_hints"]
                assert len(hints) == 6 and all(len(item["headings"]) <= 2 for item in hints)
                assert [item["page_index"] for item in hints] == sorted(item["page_index"] for item in hints)
                assert all(len(text) <= 100 for item in hints for text in item["headings"])
                hint, = [item for item in hints if heading in item["headings"]]
                assert hint["headings"] == [heading, "4.8 Prototype threshold"]
                target = {"query": "calibrated pulse threshold", "requirement": "measurement definition",
                          "doc_id": hint["doc_id"], "page_index": hint["page_index"]}
                return schema.model_validate(_judgement(verdict="pending", targets=[target]))

            assert all(hint["page_index"] != evidence_page for hint in material.get("source_hints", []))
            scope, = material["direct_inspections"]
            assert scope["doc_id"] == "paper" and scope["page_index"] == evidence_page and scope["section"] is None
            assert scope["complete"] and scope["examined_evidence_ids"] == [evidence_id]
            assert any(item["evidence_id"] == evidence_id for item in material["evidence"])
            return schema.model_validate(_judgement(
                verdict="supported", missing=[], groups=[{"role": "support", "evidence_ids": [evidence_id]}],
                observations=[{"requirement": "measurement definition", "status": "found", "evidence_ids": [evidence_id]}],
            ))

    model = Navigator()
    audit_version(report, model, lambda: None)
    assert len(model.calls) == 2
    finding, = report["findings"]
    assert finding["verdict"] == "supported" and finding["evidence_ids"] == [evidence_id]
    assert evidence_id in finding["sent_candidate_ids"]


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
    plan, = report["claim_followups"][CLAIM["claim_id"]]
    assert plan["targets"] == [_methods_target()] and plan["sent_ids"] == ["r"]

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


def test_new_scoped_followup_keeps_prior_anchors_and_replays_both_rounds():
    rows = [_source("prior", "The prototype tracks biodiversity."),
            _source("protocol", "Sampling sites were visited twice.", "Methods", 1),
            _source("definition", "Species counts excluded unverified sightings.", "Appendix", 2)]
    documents = [{"doc_id": "paper", "role": "main",
                  "pages": [{"page_index": page, "status": "parsed"} for page in range(3)]}]
    required = "measurement definition"
    first = {"query": "sampling protocol", "page_index": 1, "requirement": required, "doc_id": "paper"}
    second = {"query": "species counting rules", "page_index": 2, "requirement": required, "doc_id": "paper"}
    groups = [{"role": "support", "evidence_ids": ["prior"]}]
    final = _judgement(verdict="supported", missing=[],
                       groups=[{"role": "support", "evidence_ids": ["prior", "definition"]}],
                       observations=[{"requirement": required, "status": "found", "evidence_ids": ["definition"]}])
    initial = Model(_judgement(verdict="pending", targets=[first], groups=groups),
                    _judgement(verdict="pending", targets=[second]), ModelPaused("interrupted"))
    saved, reservations = [], []
    with closing(EvidenceIndex(rows)) as index:
        paused, _ = review_claim(CLAIM, index, initial, documents=documents, save_targets=saved.append,
                                 reserve_followup=lambda: not reservations.append(True))

    assert paused["retryable"] and paused["follow_up_round_count"] == 2
    assert len(initial.calls) == 3 and reservations == [True]
    assert [plan["targets"] for plan in saved[-1]] == [[first], [second]]
    assert {row["evidence_id"] for row in initial.calls[-1]["evidence"]} >= {"prior", "definition"}
    resumed_model = Model(final)
    with closing(EvidenceIndex(rows)) as index:
        result, _ = review_claim(CLAIM, index, resumed_model, documents=documents, saved_targets=saved[-1],
                                 reserve_followup=lambda: pytest.fail("a saved inspection was reserved again"))
    assert resumed_model.calls == [initial.calls[-1]]
    assert result["verdict"] == "supported" and result["follow_up_round_count"] == 2
    assert result["evidence_ids"] == ["prior", "definition"]


@pytest.mark.parametrize("page", [1, 2])
def test_second_followup_needs_a_new_source_and_new_sent_evidence(page):
    records = [_source("prior", "The prototype tracks biodiversity."),
               _source("method", "Sites used equal visits.", "Methods", 1)]
    required = "measurement definition"
    model = Model(_judgement(verdict="pending", targets=[{
        "query": "visits", "requirement": required, "page_index": 1, "doc_id": "paper",
    }]), _judgement(verdict="pending", targets=[{
        "query": "measurement counts", "requirement": required, "page_index": page, "doc_id": "paper",
    }]))
    saved = []
    with closing(EvidenceIndex(records)) as index:
        result, _ = review_claim(CLAIM, index, model, documents=[{"doc_id": "paper"}], save_targets=saved.append)
    assert len(model.calls) == 2 and len(saved) == 1
    assert result["follow_up_round_count"] == 1
    assert result["review_status"] == "pending"
