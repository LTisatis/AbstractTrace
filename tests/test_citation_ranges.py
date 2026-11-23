from contextlib import closing

import pytest

from abstract_review.audit import ScopedJudgement, _checked_judgement, review_claim
from abstract_review.evidence import EvidenceIndex
from abstract_review.models import InvalidJudgeReference


CLAIM = {
    "claim_id": "carrier", "kind": "theoretical", "proposition": "The coastal threshold identifies the carrier.",
    "requirements": ["carrier scope"], "support_query": "coastal threshold", "counter_query": "coastal threshold",
}


def line(eid, text, y, *, doc="paper", version="v1", page=0, section="Results"):
    return {"evidence_id": eid, "text": text, "kind": "body_text", "doc_id": doc, "doc_version_id": version,
            "locator": {"page_index": page, "section": section, "bbox": [30, y, 195, y + 10]}}


def body_rows():
    return [line(f"b{i}", "Coastal threshold source." if i == 0 else f"Source condition {i}.", 100 + 12 * i)
            for i in range(12)]


def citation(start, end, role="support"):
    return {"role": role, "start_evidence_id": start, "end_evidence_id": end}


def answer(*ranges, field_id="b4", verdict="supported", groups=(), proof=None):
    return {
        "verdict": verdict, "reason": "The original source gives the relevant condition and result.",
        "evidence_groups": list(groups), "evidence_ranges": list(ranges), "missing_fields": [], "follow_up_queries": [],
        "coverage": [{"requirement": "carrier scope", "status": "found", "evidence_ids": [field_id]}],
        "direct_contradiction": proof,
    }


class Judge:
    def __init__(self, *replies):
        self.replies = iter(replies)
        self.packets = []

    def call(self, stage, material, schema):
        assert stage == "judge"
        self.packets.append(material)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return schema.model_validate(reply)


def test_explicit_body_context_range_cites_every_selected_original_line_in_order():
    requested = citation("b4", "b8")
    fallback = {"role": "limit", "evidence_ids": ["other"]}
    judge = Judge(answer(requested, groups=[fallback]))
    other = line("other", "Coastal threshold comparator.", 100, page=1)
    with closing(EvidenceIndex([*body_rows(), other])) as index:
        finding, _ = review_claim(CLAIM, index, judge)
    expected = [f"b{i}" for i in range(4, 9)]
    assert len(judge.packets) == 1 and finding["verdict"] == "supported"
    assert not set(expected) & set(finding["candidate_ids"])
    assert set(expected) <= set(finding["sent_candidate_ids"])
    assert finding["evidence_ids"] == ["other", *expected]
    assert finding["evidence_groups"] == [fallback, {"role": "support", "evidence_ids": expected}]
    assert finding["evidence_ranges"] == [{**requested, "evidence_ids": expected}]


def test_range_can_reference_a_wrapped_primary_passage():
    rows = [line("first", "The coastal threshold stays fixed under", 100),
            line("last", "the coastal threshold control.", 112)]
    judge = Judge(answer(citation("first", "last"), field_id="last"))
    with closing(EvidenceIndex(rows)) as index:
        finding, _ = review_claim(CLAIM, index, judge)
    packet, = judge.packets
    assert "body_context" not in packet
    assert packet["body_passages"] == [{"evidence_ids": ["first", "last"]}]
    assert finding["evidence_ids"] == ["first", "last"]


@pytest.mark.parametrize("windows, requested, changes", [
    (("abc",), ("c", "a"), {}),
    (("abc",), ("a", "unknown"), {}),
    (("abc",), ("a", "c"), {"b": {"page": 1}}),
    (("abc",), ("a", "c"), {"b": {"doc": "appendix"}}),
    (("abc",), ("a", "c"), {"b": {"version": "v2"}}),
    (("abc",), ("a", "c"), {"b": {"section": "Methods"}}),
    (("ab", "bc"), ("a", "c"), {}),
    (("abc", "adc"), ("a", "c"), {}),
])
def test_invalid_ranges_cannot_cross_scope_or_join_overlapping_windows(windows, requested, changes):
    material = {"body_context": [{"lines": [[eid] for eid in window]} for window in windows]}
    records = {eid: line(eid, eid, 100 + i * 12, **changes.get(eid, {})) for i, eid in enumerate("abcd")}
    output = ScopedJudgement.model_validate(answer(citation(*requested)))
    with pytest.raises(InvalidJudgeReference, match="invalid or ambiguous citation range"):
        _checked_judgement(output, material, list(records), records)


def test_unsent_middle_anchor_invalidates_a_range_even_when_endpoints_are_sent():
    material = {"body_passages": [{"evidence_ids": ["a", "b", "c"]}]}
    records = {eid: line(eid, eid, 100 + i * 12) for i, eid in enumerate("abc")}
    output = ScopedJudgement.model_validate(answer(citation("a", "c")))
    with pytest.raises(InvalidJudgeReference, match="invalid or ambiguous citation range"):
        _checked_judgement(output, material, ["a", "c"], records)


def test_matching_subrange_in_two_windows_resolves_to_one_group():
    material = {"body_passages": [{"evidence_ids": list("abc")}],
                "body_context": [{"lines": [[eid] for eid in "abcd"]}]}
    records = {eid: line(eid, eid, 100 + i * 12) for i, eid in enumerate("abcd")}
    requested = citation("b", "c")
    output = ScopedJudgement.model_validate(answer(requested, requested))
    resolved, ids, trace = _checked_judgement(output, material, list(records), records)
    assert ids == ["b", "c"]
    assert len(resolved.evidence_groups) == len(trace) == 1


@pytest.mark.parametrize("split", [False, True])
def test_direct_contradiction_needs_one_group_with_all_its_anchors(split):
    proof = {"evidence_ids": ["b6"], "scope_evidence_ids": ["b2"], "subject_and_conditions": "same",
             "relation": "opposite_result", "scope_reason": "The tested carrier matches the proposed setting.",
             "supplementary_fields": []}
    ranges = [citation("b2", "b4", "refute"), citation("b4", "b6", "refute")] if split else [citation("b2", "b6", "refute")]
    judge = Judge(answer(*ranges, field_id="b2", verdict="refuted", proof=proof))
    with closing(EvidenceIndex(body_rows())) as index:
        finding, _ = review_claim(CLAIM, index, judge)
    assert len(judge.packets) == 1
    assert finding["verdict"] == (None if split else "refuted")
    assert finding.get("refutation_blocker") == ("ungrounded_contradiction" if split else None)


@pytest.mark.parametrize("repair_ok", [True, False])
def test_transport_reference_error_and_bad_range_share_one_repair_and_keep_the_sent_trace(repair_ok):
    requested = citation("b4", "b8") if repair_ok else citation("b8", "b4")
    judge = Judge(InvalidJudgeReference("unavailable transport alias"), answer(requested))
    with closing(EvidenceIndex(body_rows())) as index:
        finding, _ = review_claim(CLAIM, index, judge)
    first, retry = judge.packets
    assert {key: value for key, value in retry.items() if key != "retry"} == first
    assert set(finding["sent_candidate_ids"]) == {row["evidence_id"] for row in body_rows()}
    assert finding.get("body_context")
    assert finding["verdict"] == ("supported" if repair_ok else None)
    assert ("evidence_ranges" in finding) is repair_ok
    if not repair_ok:
        assert "invalid or ambiguous citation range" in finding["reason"]
