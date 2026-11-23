from contextlib import closing

import pytest

from abstract_review.audit import review_claim
from abstract_review.evidence import EvidenceIndex


def _row(eid, text, page, *, section="Results", box=None, version="v1"):
    locator = {"page_index": page, "section": section}
    if box is not None:
        locator["bbox"] = box
    return {"evidence_id": eid, "doc_id": "paper", "doc_version_id": version,
            "kind": "body_text", "text": text, "locator": locator}


def _document(next_status="parsed"):
    pages = [{"page_index": page, "width": 600, "height": 800, "status": status}
             for page, status in enumerate(("parsed", next_status))]
    return {"doc_id": "paper", "role": "main", "pages": pages}


def _output(*, targets=(), groups=(), coverage=(), verdict="pending", proof=None):
    return {"verdict": verdict, "reason": "source-grounded check", "evidence_groups": list(groups),
            "coverage": list(coverage), "missing_fields": [], "follow_up_queries": [],
            "follow_up_targets": list(targets), "direct_contradiction": proof}


class _Replies:
    def __init__(self, *replies):
        self.calls = []
        self.replies = iter(replies)

    def call(self, stage, material, schema):
        assert stage == "judge"
        self.calls.append(material)
        return schema.model_validate(next(self.replies))


CLAIM = {
    "claim_id": "claim", "kind": "theoretical", "proposition": "Fixed attention cannot recognize finite-state language.",
    "support_query": "fixed attention finite-state language", "counter_query": "finite-state attention witness",
    "requirements": ["fixed architecture"],
}


def _continuing_source(*, section="Results", version="v1", tail_end="alphabet"):
    return [
        _row("unrelated", "Prior work considered larger networks.", 0, box=[60, 660, 260, 670]),
        _row("tail", f"Consider the regular language over the {tail_end}", 0, box=[322, 711, 542, 722]),
        _row("printed-page-number", "10", 0, box=[292, 752, 308, 762]),
        _row("next-1", "alphabet where every accepted symbol is one.", 1, section=section,
             box=[60, 52, 280, 63], version=version),
    ]


def test_followup_completes_a_cross_page_source_without_claiming_exhaustive_scope():
    rows = _continuing_source(tail_end="binary")
    target = {"query": "regular language", "page_index": 0, "doc_id": "paper", "requirement": "fixed architecture"}
    found = {"requirement": "fixed architecture", "status": "found", "evidence_ids": ["next-1"]}
    proof = {"evidence_ids": ["tail", "next-1"], "subject_and_conditions": "same",
             "relation": "in_scope_counterexample", "scope_reason": "The fixed architecture applies to both pages.",
             "supplementary_fields": []}
    refutation = {"role": "refute", "evidence_ids": ["tail", "next-1"]}
    model = _Replies(_output(targets=[target]), _output(verdict="refuted", groups=[refutation], coverage=[found], proof=proof))
    with closing(EvidenceIndex(rows)) as index:
        finding, _ = review_claim(CLAIM, index, model, documents=[_document()])

    assert finding["verdict"] == "refuted"
    assert len(model.calls) == 2
    followup = model.calls[-1]
    assert followup["body_continuations"] == [{"from_evidence_id": "tail", "to_evidence_ids": ["next-1"]}]
    assert {"doc_id": "paper", "page_index": 1, "section": "Results",
            "lines": [["next-1", rows[-1]["text"]]]} in followup["body_context"]
    direct, = followup["direct_inspections"]
    assert direct["page_index"] == 0 and not direct["complete"]
    assert finding["retrieval_trace"]["page_continuations"] == followup["body_continuations"]
    assert "next-1" in finding["follow_up_evidence_ids"]
    assert set(finding["evidence_ids"]) == {"tail", "next-1"}


def test_continuation_keeps_the_other_requested_scope_in_the_body_budget():
    rows = _continuing_source(tail_end="binary")
    rows.extend(_row(f"setup-{i}", f"Earlier binary setup detail {i}.", 0, box=[60, 150 + 12 * i, 280, 160 + 12 * i])
                for i in range(3))
    rows.extend(_row(f"controlled-{i}", f"Controlled saturation comparison detail {i}.", 2,
                     box=[60, 100 + 12 * i, 280, 110 + 12 * i]) for i in range(12))
    document = _document()
    document["pages"].append({"page_index": 2, "width": 600, "height": 800, "status": "parsed"})
    requirement = {"doc_id": "paper", "requirement": "fixed architecture"}
    targets = [{**requirement, "query": "binary regular language", "page_index": 0},
               {**requirement, "query": "controlled saturation comparison", "page_index": 2}]
    model = _Replies(_output(targets=targets), _output())
    with closing(EvidenceIndex(rows)) as index:
        finding, _ = review_claim(CLAIM, index, model, documents=[document])
    windows = model.calls[-1]["body_context"]
    assert [window["page_index"] for window in windows] == [1, 2]
    assert "next-1" in finding["follow_up_evidence_ids"]
    extra = [line[1] for window in windows for line in window["lines"] if len(line) == 2]
    assert len(extra) <= 24 and sum(map(len, extra)) <= 1800


@pytest.mark.parametrize(("source", "document"), [
    (_continuing_source(section="Methods"), _document()),
    (_continuing_source(version="v2"), _document()),
    (_continuing_source(tail_end="alphabet."), _document()),
    (_continuing_source(), _document("unreadable")),
])
def test_page_continuation_requires_unfinished_prose_in_same_readable_source(source, document):
    with closing(EvidenceIndex(source)) as index:
        assert index.page_continuation(["tail"], [document]) is None


def test_printed_heading_at_next_page_head_prevents_joining_different_sections():
    heading = "4 Follow-up analysis"
    document = _document()
    document["pages"][0]["source_headings"] = "Legacy heading"
    document["pages"][1]["source_headings"] = [None, heading]
    rows = _continuing_source()
    rows.append(_row("new-section", heading, 1, box=[60, 30, 210, 43]))
    with closing(EvidenceIndex(rows)) as index:
        assert index.page_continuation(["tail"], [document]) is None


def test_followup_preserves_both_prior_key_anchors_and_reuses_them_on_resume():
    rows = [
        _row("scope", "The fixed architecture uses one attention head.", 0),
        _row("result", "The model recognizes the finite-state language.", 0),
        *(_row(f"experiment-{number}", f"Controlled comparison measured variant {number}.", 1)
          for number in range(12)),
    ]
    target = {"query": "controlled comparison", "page_index": 1, "doc_id": "paper", "requirement": "fixed architecture"}
    cited = ["scope", "result"]
    pending = _output(targets=[target], groups=[{"role": "refute", "evidence_ids": cited}])
    model = _Replies(pending, _output())
    saved = []
    with closing(EvidenceIndex(rows)) as index:
        finding, _ = review_claim(CLAIM, index, model, documents=[_document()], save_targets=saved.append)
    assert len(finding["candidate_ids"]) == 8
    assert set(cited).issubset(finding["candidate_ids"])
    assert saved == [[{**target, "retained_evidence_ids": cited}]]

    with closing(EvidenceIndex(rows)) as index:
        resumed, _ = review_claim(CLAIM, index, _Replies(_output()), documents=[_document()], saved_targets=saved[0])
    assert resumed["candidate_ids"] == finding["candidate_ids"]
