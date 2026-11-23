from abstract_review.audit import audit_version, review_claim, targeted_evidence
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
