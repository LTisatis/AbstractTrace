from contextlib import closing

import pytest

from abstract_review.audit import ScopedJudgement, _empirical_context_query, _pending_targets, review_claim, supplied_claims
from abstract_review.evidence import EvidenceIndex


def empirical_claim(**fields):
    claim, = supplied_claims((("claim", "The evaluation results show that load limits detector performance."),))
    return {**claim, "support_query": "detector", "counter_query": "detector evidence", **fields}


def evidence(eid, text, *, page=0, doc_id="paper", version=None):
    record = {"evidence_id": eid, "text": text, "doc_id": doc_id, "kind": "body_text",
              "locator": {"page_index": page, "section": "Related Work"}}
    if version is not None:
        record["doc_version_id"] = version
    return record


def supported(claim, proof, cited):
    return {"verdict": "supported", "reason": "The paper reports an effect for this detector.",
            "evidence_groups": [{"role": "support", "evidence_ids": cited}],
            "missing_fields": [], "follow_up_queries": [],
            "coverage": [{"requirement": claim["requirements"][0], "status": "found", "evidence_ids": cited}],
            "empirical_support": proof}


class Model:
    def __init__(self, *views):
        self.views = iter(views)
        self.materials = []

    def call(self, stage, material, schema):
        assert stage == "judge"
        self.materials.append(material)
        return schema.model_validate(next(self.views))


def test_author_summary_triggers_one_grounded_check_for_primary_results():
    claim = empirical_claim(proposition="The evaluation results demonstrate the limitations of the detector under load.",
                            support_query="detector summary", counter_query="detector summary")
    records = [
        evidence("summary", "Detector summary: long inputs exposed important limitations."),
        evidence("outcome", "Evaluation failure cases: load caused 12 missed pulses in the detector.", page=1),
        evidence("setup", "Error categories were measured for the tested detector at high load.", page=1),
    ]
    documents = [{"doc_id": "paper", "role": "main", "pages": [
        {"page_index": 0, "source_headings": ["1 Introduction"]},
        {"page_index": 1, "source_headings": ["4.3 Error Breakdown"]},
    ]}]
    restart = supported(claim, {"result_source": "summary", "result_evidence_ids": ["summary"],
                                "scope_evidence_ids": ["summary"], "same_study_and_scope": "same"}, ["summary"])
    complete = supported(claim, {"result_source": "direct_result", "result_evidence_ids": ["outcome"],
                                 "scope_evidence_ids": ["setup"], "same_study_and_scope": "same"},
                         ["outcome", "setup"])
    model = Model(restart, complete)
    targets = []
    with closing(EvidenceIndex(records)) as index:
        finding, _ = review_claim(claim, index, model, documents=documents, save_targets=targets.extend)
    assert len(model.materials) == 2
    target, = targets
    assert target["doc_id"] == "paper" and target["page_index"] == 1
    assert "Error Breakdown" in target["query"] and target["source_kind"] == "results"
    assert model.materials[0]["empirical_support_required"] is True
    assert finding["verdict"] == "supported"
    assert finding["empirical_support"] == complete["empirical_support"]
    assert "empirical_support_blocker" not in finding


def test_results_page_keeps_typed_query_and_uses_printed_failure_heading_for_context():
    claim = empirical_claim(proposition="The evaluation results demonstrate detector limitations under load.")
    records = [
        evidence("summary", "Detector evaluation summary."),
        evidence("failures", "Evaluation error categories showed 12 missing pulses.", page=1),
        evidence("scope", "All evaluation trials used the same detector at high load.", page=2),
    ]
    documents = [{"doc_id": "paper", "role": "main", "pages": [
        {"page_index": 1, "source_headings": ["4.3 Error Breakdown", "4.4 Evaluation Results"]},
        {"page_index": 2, "source_headings": ["3 Methods"]},
    ]}]
    follow_up = {"verdict": "pending", "reason": "Read the results and methods.", "evidence_groups": [],
                 "missing_fields": [], "follow_up_queries": [],
                 "follow_up_targets": [
                     {"query": "Compare Evaluation Results", "source_kind": "results", "page_index": 1},
                     {"query": "Detector evaluation trials", "source_kind": "methods", "page_index": 2},
                 ]}
    model = Model(follow_up, supported(claim, {"result_source": "direct_result",
                                               "result_evidence_ids": ["failures"], "scope_evidence_ids": ["scope"],
                                               "same_study_and_scope": "same"}, ["failures", "scope"]))
    with closing(EvidenceIndex(records)) as index:
        finding, _ = review_claim(claim, index, model, documents=documents)
    result, method = model.materials[1]["direct_inspections"]
    assert result["query"] == "Compare Evaluation Results"
    assert result["body_context_query"] == "4.3 Error Breakdown evaluation failure cases and error categories"
    assert method["query"] == "Detector evaluation trials" and "body_context_query" not in method
    assert finding["verdict"] == "supported"


@pytest.mark.parametrize(("query", "section", "expected"), [
    ("Find the main detector results and comparison", None, "Experimental Results"),
    ("Find the alternate-room experiment", None, None),
    ("Find the main detector results", "Alternate room", None),
])
def test_main_result_context_requires_a_matching_printed_heading(query, section, expected):
    scope = {"doc_id": "paper", "page_index": 2, "source_kind": "results", "section": section, "query": query}
    documents = [{"doc_id": "paper", "pages": [
        {"page_index": 2, "source_headings": ["5 Alternate Room", "4 Experimental Results"]},
    ]}]
    assert _empirical_context_query(empirical_claim(), scope, documents) == expected


@pytest.mark.parametrize(("variant", "blocker"), [
    ("summary", "summary_only_result"),
    ("missing", "missing_empirical_support"),
    ("no_result", "missing_observed_result"),
    ("different_conditions", "unconfirmed_tested_scope"),
    ("another_document", "unconfirmed_tested_scope"),
    ("another_version", "unconfirmed_tested_scope"),
    ("separate_groups", "unlinked_result_and_scope"),
    ("unsent_anchor", "ungrounded_empirical_support"),
])
def test_completed_support_requires_observation_and_matching_tested_scope(variant, blocker):
    claim = empirical_claim()
    result = evidence("outcome", "The detector missed 12 pulses under load.", version="v1")
    scope = evidence("setup", "The detector was tested at high load.", version="v1")
    proof = {"result_source": "direct_result", "result_evidence_ids": ["outcome"],
             "scope_evidence_ids": ["setup"], "same_study_and_scope": "same"}
    if variant == "summary":
        proof["result_source"] = "summary"
    elif variant == "missing":
        proof = None
    elif variant == "no_result":
        proof["result_evidence_ids"] = []
    elif variant == "different_conditions":
        proof["same_study_and_scope"] = "different"
    elif variant == "another_document":
        scope["doc_id"] = "other"
    elif variant == "another_version":
        scope["doc_version_id"] = "v2"
    elif variant == "unsent_anchor":
        proof["scope_evidence_ids"] = ["absent"]
    outcome = supported(claim, proof, ["outcome", "setup"])
    if variant == "separate_groups":
        outcome["evidence_groups"] = [{"role": "support", "evidence_ids": [eid]} for eid in ("outcome", "setup")]
    model = Model(outcome)
    with closing(EvidenceIndex([result, scope])) as index:
        finding, _ = review_claim(claim, index, model)
    assert finding["review_status"] == "pending" and finding["verdict"] is None
    assert finding["empirical_support_blocker"] == blocker
    assert "empirical_support" not in finding


def test_explicitly_nonempirical_method_can_complete_with_ordinary_citations():
    claim = empirical_claim(empirical_result=False)
    model = Model(supported(claim, None, ["method"]))
    with closing(EvidenceIndex([evidence("method", "The detector uses an event-driven design.")])) as index:
        finding, _ = review_claim(claim, index, model)
    assert finding["verdict"] == "supported"
    assert "empirical_support_required" not in model.materials[0]


def test_tested_scope_can_come_from_the_attached_supplement():
    claim = empirical_claim()
    records = [evidence("outcome", "The detector missed 12 pulses."),
               evidence("setup", "The detector was tested at high load.", doc_id="appendix")]
    proof = {"result_source": "direct_result", "result_evidence_ids": ["outcome"],
             "scope_evidence_ids": ["setup"], "same_study_and_scope": "same"}
    model = Model(supported(claim, proof, ["outcome", "setup"]))
    documents = [{"doc_id": "paper", "role": "main"}, {"doc_id": "appendix", "role": "supplement"}]
    with closing(EvidenceIndex(records)) as index:
        finding, _ = review_claim(claim, index, model, documents=documents)
    assert finding["verdict"] == "supported" and finding["empirical_support"] == proof


def test_typed_target_replaces_legacy_duplicate_without_losing_second_check():
    claim = empirical_claim()
    output = ScopedJudgement.model_validate({**supported(claim, None, []),
        "follow_up_targets": [{"query": "detector diagnostics", "page_index": 2}],
        "follow_up_queries": [" DETECTOR  DIAGNOSTICS ", "deployment setup"],
    })
    targets = _pending_targets(output, claim["requirements"], ())
    assert [target["query"] for target in targets] == ["detector diagnostics", "deployment setup"]
    assert targets[0]["page_index"] == 2


@pytest.mark.parametrize("comparable", [None, "uncertain", "same"])
def test_comparative_emphasis_requires_verified_outcomes_under_two_conditions(comparable):
    claim = empirical_claim(
        proposition="Evaluation results show the detector's failures are especially frequent under high load.",
        support_query="detector pulse failures", counter_query="detector pulse baseline", empirical_result=False,
    )
    proof = {"result_source": "direct_result", "result_evidence_ids": ["a", "b"],
             "scope_evidence_ids": ["a"], "same_study_and_scope": "same"}
    if comparable is not None:
        proof["comparison"] = {
            "focal_condition": "High load", "comparator_condition": "Normal load", "shared_outcome": "Missed pulses",
            "focal_evidence_ids": ["a"], "comparator_evidence_ids": ["b"], "comparable": comparable,
        }
    model = Model(supported(claim, proof, ["a", "b"]))
    records = [evidence("a", "The detector missed 12 pulses under high load."),
               evidence("b", "The detector missed one pulse under normal load.")]
    with closing(EvidenceIndex(records)) as index:
        finding, _ = review_claim(claim, index, model, reserve_followup=lambda: False)
    assert "empirical_support.comparison" in model.materials[0]["coverage_instructions"]
    if comparable == "same":
        assert finding["verdict"] == "supported"
    else:
        assert finding["review_status"] == "pending"
        assert finding["stop_reason"] == "follow_up_budget"
        assert finding["empirical_support_blocker"] == "unconfirmed_matched_comparison"
