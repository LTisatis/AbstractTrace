import json
from contextlib import closing
from copy import deepcopy
from types import SimpleNamespace

import pytest

from abstract_review.audit import review_claim
from abstract_review.evidence import EvidenceIndex
from abstract_review.models import InvalidJudgeReference, Judgement, ResponsesModel, Schema


MAIN = "main-" + "a" * 64
SUPPLEMENT = "supplement-" + "b" * 64
PRIMARY = "ev-" + "c" * 64
TABLE = "ev-" + "d" * 64
BODY = "ev-" + "e" * 64
UNSENT = "ev-" + "f" * 64


class _Field(Schema):
    evidence_ids: list[str]


class _Proof(_Field):
    scope_evidence_ids: list[str]


class _Range(Schema):
    start_evidence_id: str
    end_evidence_id: str
    role: str


class _Support(Schema):
    result_evidence_ids: list[str]
    scope_evidence_ids: list[str]


class _Target(Schema):
    query: str
    doc_id: str | None


class _JudgeResult(Judgement):
    coverage: list[_Field]
    direct_contradiction: _Proof
    evidence_ranges: list[_Range]
    empirical_support: _Support
    follow_up_targets: list[_Target]


def _material():
    text = f"供核对的原文引用 {PRIMARY}，其中 D1 是原文写法。"
    return {
        "claim": text,
        "evidence": [{"evidence_id": PRIMARY, "doc_id": MAIN, "text": text,
                      "locator": {"page_index": 1, "bbox": [0, 1, 2, 3]}}],
        "table_context": [{"cell_evidence_id": PRIMARY,
                           "evidence": [{"evidence_id": TABLE, "doc_id": MAIN, "text": "Units"}]}],
        "body_passages": [{"evidence_ids": [PRIMARY, BODY]}],
        "body_context": [{"doc_id": SUPPLEMENT, "lines": [[PRIMARY], [BODY, text]]}],
        "direct_inspections": [{"doc_id": MAIN, "document_ids": [MAIN, SUPPLEMENT],
                                "examined_evidence_ids": [PRIMARY, UNSENT]}],
        "source_hints": [{"doc_id": SUPPLEMENT, "page_index": 4, "headings": ["Findings"]}],
        "calculation": {"evidence_ids": [PRIMARY, TABLE], "scope_evidence_ids": [BODY]},
        "figure_observations": [{"evidence_id": PRIMARY, "status": "available"}],
    }


def _result():
    return {
        "verdict": "refuted", "reason": "The reported result differs.",
        "evidence_groups": [{"role": "refute", "evidence_ids": ["E1", "E3"]}],
        "missing_fields": [], "follow_up_queries": [],
        "coverage": [{"evidence_ids": ["E2"]}],
        "direct_contradiction": {"evidence_ids": ["E3"], "scope_evidence_ids": ["E2"]},
        "evidence_ranges": [{"start_evidence_id": "E1", "end_evidence_id": "E3", "role": "refute"}],
        "empirical_support": {"result_evidence_ids": ["E3"], "scope_evidence_ids": ["E2"]},
        "follow_up_targets": [{"query": "look for results", "doc_id": "D2"}, {"query": "alternate", "doc_id": None}],
    }


def _model(parse, max_requests=2):
    return ResponsesModel(client=SimpleNamespace(responses=SimpleNamespace(parse=parse)), max_requests=max_requests)


def _parsed(schema, result):
    return SimpleNamespace(id="r1", status="completed", model="pinned", usage=None,
                           output_parsed=schema.model_validate(result))


def test_judge_compacts_typed_refs_and_restores_refs_without_touching_source_text():
    material = _material()
    original = deepcopy(material)
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        assert kwargs["text_format"] is _JudgeResult and kwargs["store"] is False
        payload = kwargs["input"][1]["content"]
        wire = json.loads(payload)
        assert len(payload) < len(json.dumps(material, ensure_ascii=False, separators=(",", ":")))
        assert wire["claim"] == original["claim"]
        assert wire["evidence"] == [{**original["evidence"][0], "evidence_id": "E1", "doc_id": "D1"}]
        assert wire["table_context"] == [{"cell_evidence_id": "E1", "evidence": [
            {"evidence_id": "E2", "doc_id": "D1", "text": "Units"},
        ]}]
        assert wire["body_passages"] == [{"evidence_ids": ["E1", "E3"]}]
        assert wire["body_context"] == [{"doc_id": "D2", "lines": [["E1"], ["E3", original["claim"]]]}]
        assert wire["direct_inspections"] == [{"doc_id": "D1", "document_ids": ["D1", "D2"],
                                               "examined_evidence_ids": ["E1", "E4"]}]
        assert wire["source_hints"][0]["doc_id"] == "D2"
        assert wire["calculation"] == {"evidence_ids": ["E1", "E2"], "scope_evidence_ids": ["E3"]}
        assert wire["figure_observations"][0]["evidence_id"] == "E1"
        return _parsed(kwargs["text_format"], _result())

    model = _model(parse)
    restored = model.call("judge", material, _JudgeResult)

    assert material == original and len(calls) == 1
    assert restored.evidence_groups[0].evidence_ids == [PRIMARY, BODY]
    assert restored.coverage[0].evidence_ids == [TABLE]
    assert restored.direct_contradiction.model_dump() == {"evidence_ids": [BODY], "scope_evidence_ids": [TABLE]}
    assert restored.evidence_ranges[0].model_dump() == {"start_evidence_id": PRIMARY, "end_evidence_id": BODY,
                                                        "role": "refute"}
    assert restored.empirical_support.model_dump() == {"result_evidence_ids": [BODY], "scope_evidence_ids": [TABLE]}
    assert [target.doc_id for target in restored.follow_up_targets] == [SUPPLEMENT, None]


def test_judge_can_only_cite_table_region_cells_actually_supplied():
    material = _material()
    material["table_region"] = {"reference_evidence_id": BODY, "evidence": [
        {"evidence_id": "new-cell", "doc_id": MAIN, "text": "27 ms"},
    ]}

    def parse(**kwargs):
        wire = json.loads(kwargs["input"][1]["content"])
        region = wire["table_region"]
        assert region["reference_evidence_id"] == "E3"
        assert region["evidence"] == [{"evidence_id": "E4", "doc_id": "D1", "text": "27 ms"}]
        result = {**_result(), "evidence_groups": [{"role": "refute", "evidence_ids": ["E4"]}]}
        return _parsed(kwargs["text_format"], result)

    restored = _model(parse).call("judge", material, _JudgeResult)
    assert restored.evidence_groups[0].evidence_ids == ["new-cell"]


def test_priority_evidence_order_can_refer_to_supplied_context():
    material = _material()
    material["priority_evidence_ids"] = [BODY, TABLE, PRIMARY]

    def parse(**kwargs):
        wire = json.loads(kwargs["input"][1]["content"])
        assert wire["priority_evidence_ids"] == ["E3", "E2", "E1"]
        return _parsed(kwargs["text_format"], _result())

    assert _model(parse).call("judge", material, _JudgeResult).evidence_groups[0].evidence_ids == [PRIMARY, BODY]


@pytest.mark.parametrize(("path", "bad"), [
    (("evidence_groups", 0, "evidence_ids", 0), PRIMARY),
    (("evidence_groups", 0, "evidence_ids", 0), "E4"),
    (("coverage", 0, "evidence_ids", 0), "E987"),
    (("direct_contradiction", "scope_evidence_ids", 0), "D1"),
    (("evidence_ranges", 0, "end_evidence_id"), "E4"),
    (("empirical_support", "result_evidence_ids", 0), "E4"),
    (("follow_up_targets", 0, "doc_id"), "E1"),
])
def test_invalid_aliases_raise_for_audit_to_repair_without_transport_retry(path, bad):
    result = _result()
    parent = result
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = bad
    model = _model(lambda **kwargs: _parsed(kwargs["text_format"], result))

    with pytest.raises(InvalidJudgeReference, match="outside the supplied scope"):
        model.call("judge", _material(), _JudgeResult)

    assert len(model.events) == 1
    assert model.events[0]["status"] == "invalid_judge_reference"


def test_judge_figure_label_compacts_and_aliases_are_local_to_each_request():
    image = "data:image/png;base64,AAAA"
    figure = {"evidence_id": PRIMARY, "image_url": image}

    def parse(**kwargs):
        wire, marker, crop = kwargs["input"][1]["content"]
        assert wire["type"] == "input_text"
        assert json.loads(wire["text"])["evidence"][0]["evidence_id"] == "E1"
        assert marker["text"] == "Source image for evidence E1"
        assert crop["image_url"] == image
        return _parsed(kwargs["text_format"], {
            "verdict": "supported", "reason": "The source states it.",
            "evidence_groups": [{"role": "support", "evidence_ids": ["E1"]}],
            "missing_fields": [], "follow_up_queries": [],
        })

    model = _model(parse)
    for eid in (PRIMARY, BODY):
        material = {"evidence": [{"evidence_id": eid, "doc_id": MAIN, "text": "Caption"}],
                    "figure_images": [{**figure, "evidence_id": eid}]}
        result = model.call("judge", material, Judgement)
        assert result.evidence_groups[0].evidence_ids == [eid]


def test_audit_repairs_an_invalid_wire_alias_once_and_restores_original_citation():
    claim = {"claim_id": "c", "kind": "result", "proposition": "The protocol had two checks.",
             "text": "The protocol had two checks.", "requirements": ["reported checks"],
             "support_query": "protocol checks", "counter_query": "protocol validation"}
    row = {"evidence_id": PRIMARY, "doc_id": MAIN, "kind": "body_text",
           "text": "The protocol required two checks.", "locator": {"page_index": 0, "section": "Methods"}}
    payloads = []

    def parse(**kwargs):
        wire = json.loads(kwargs["input"][1]["content"])
        payloads.append(wire)
        assert wire["evidence"][0]["evidence_id"] == "E1"
        return _parsed(kwargs["text_format"], {
            "verdict": "pending", "reason": "The other condition is still unclear.",
            "evidence_groups": [{"role": "limit", "evidence_ids": ["E1" if "retry" in wire else "E90"]}],
            "missing_fields": [], "follow_up_queries": [],
        })

    model = _model(parse)
    with closing(EvidenceIndex([row])) as index:
        finding, _ = review_claim(claim, index, model)

    assert len(payloads) == 2
    first, second = payloads
    assert {key: value for key, value in second.items() if key != "retry"} == first
    assert finding["evidence_ids"] == [PRIMARY]
    assert finding["sent_candidate_ids"] == [PRIMARY]
    assert [event["status"] for event in model.events] == ["invalid_judge_reference", "completed"]


def test_nonjudge_stage_uses_canonical_ids():
    material = _material()

    def parse(**kwargs):
        assert kwargs["input"][1]["content"] == json.dumps(material, ensure_ascii=False)
        return _parsed(kwargs["text_format"], {
            "verdict": "supported", "reason": "The source states it.",
            "evidence_groups": [{"role": "support", "evidence_ids": [PRIMARY]}],
            "missing_fields": [], "follow_up_queries": [],
        })

    restored = _model(parse).call("verify_sentence", material, Judgement)
    assert restored.evidence_groups[0].evidence_ids == [PRIMARY]
