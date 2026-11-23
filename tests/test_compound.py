import hashlib

import pytest

from abstract_review.audit import extract_claims, refresh_compound_findings, supplied_claims
from abstract_review.compound import aggregate_compound_findings, shared_procedure_parts


CONJUNCTION = "Against the same baseline, accuracy increased by 10 percentage points on datasets A, B, and C."
DISJUNCTION = "Either the model reduced errors or recall increased."


def _verdicts(claims, values):
    return [{"claim_id": claim["claim_id"], "review_status": status, "verdict": verdict,
             "reason": f"result for {i}", "evidence_ids": [f"e-{i}"]}
            for i, (claim, (status, verdict)) in enumerate(zip(claims, values))]


def test_english_dataset_scope_preserves_exact_operand_anchors_and_shared_clause():
    claims = extract_claims(CONJUNCTION)
    assert [claim["text"] for claim in claims] == ["A", "B", "C"]
    assert [claim["proposition"] for claim in claims] == [
        f"Against the same baseline, accuracy increased by 10 percentage points on dataset {name}."
        for name in "ABC"
    ]
    assert claims[0]["start"] == CONJUNCTION.index(" A,") + 1
    assert all(CONJUNCTION[atom["start"]:atom["end"]] == atom["quote"] for atom in claims)
    assert all(atom["compound"]["member_claim_ids"] == [claim["claim_id"] for claim in claims] for atom in claims)
    assert claims == extract_claims(CONJUNCTION)


@pytest.mark.parametrize("source", [
    "与同一基线相比，数据集 A、B 和 C 的准确率均提高 10 个百分点。",
    "在 A、B 和 C 三个数据集上，相较同一基线，准确率均提高10个百分点。",
])
def test_chinese_distributive_dataset_scope_preserves_the_shared_condition(source):
    claims = extract_claims(source)
    assert [claim["text"] for claim in claims] == ["A", "B", "C"]
    assert all("同一基线" in claim["proposition"] and "百分点" in claim["proposition"] for claim in claims)
    assert all(claim["compound"]["operator"] == "and" for claim in claims)


@pytest.mark.parametrize("source", [
    "The comparison uses accuracy and recall for datasets A and B.",
    "The model improved accuracy and recall.",
    "The pooled accuracy increased by 10 percentage points on datasets A and B.",
    "There was no increase of 10 percentage points on datasets A or B.",
    "Either the model reduced errors or recall increased, but not both.",
    "数据集 A 或 B 的准确率未提高10个百分点。",
    "数据集 A 和 B 并非都提高10个百分点。",
])
def test_noun_phrases_and_unsupported_scope_do_not_become_independent_operands(source):
    claim, = extract_claims(source)
    assert claim["text"] == source and "compound" not in claim


def test_simple_independent_conjunction_and_real_alternative_have_different_operators():
    and_claims = extract_claims("The model reduced errors and recall increased.")
    or_claims = extract_claims(DISJUNCTION)
    assert [claim["compound"]["operator"] for claim in and_claims] == ["and", "and"]
    assert [claim["compound"]["operator"] for claim in or_claims] == ["or", "or"]
    assert [claim["text"] for claim in or_claims] == ["the model reduced errors", "recall increased."]


@pytest.mark.parametrize(("first", "connector", "second"), [
    ("We applied statcheck for the detection of consistency errors in all included papers",
     " and ", "manually corrected false positives"),
    ("We applied statcheck for the detection of consistency errors in all included papers",
     " and ", "accepted every flag without manual review"),
    ("We collected identifiers", ", and ", "selected controls"),
    ("We collected identifiers", " and ", "recorded the sample sizes"),
])
def test_shared_first_person_subject_leaves_each_procedure_anchored(first, connector, second):
    sentence = f"{first}{connector}{second}."
    atoms, operator = shared_procedure_parts(sentence)
    assert operator == "and"
    assert atoms == [
        (0, len(first), first),
        (len(first + connector), len(sentence) - 1, f"We {second}."),
    ]

    source = f"Background information. {sentence}"
    _, *claims = extract_claims(source)
    assert [claim["text"] for claim in claims] == [first, second]
    assert [claim["proposition"] for claim in claims] == [first, f"We {second}."]
    assert all(source[claim["start"]:claim["end"]] == claim["quote"] for claim in claims)
    assert all(claim["compound"]["operator"] == "and" for claim in claims)


@pytest.mark.parametrize("sentence", [
    "We did not collect measurements and record observations.",
    "We could collect measurements and record observations.",
    "We collected no measurements and recorded observations.",
    "We screened participants and must record observations.",
    "We automatically screened participants and recorded observations.",
    "We screened participants and recorded observations automatically.",
    "Before the trial, we screened participants and recorded observations.",
    "We screened participants and recorded observations before the trial.",
    "We screened participants and recorded observations if approved.",
    "We reported clinicians screened participants and recorded observations.",
    "We accepted clinicians screened participants and recorded the observations.",
    "We recorded she screened participants and selected the controls.",
    "We screened participants who selected controls and recorded observations.",
    "We screened and recorded clinical observations.",
    "We screened participants or recorded observations.",
    "We screened participants and recorded observations or excluded outliers.",
    "We screened participants, recorded observations, and excluded outliers.",
    "We recorded observations and response times.",
    "We screened patients and selected controls.",
    "We reviewed false negatives and manually corrected false positives.",
    "We screened patients and recorded outcomes in the pilot.",
    "We screened patients and recorded outcomes without replacement.",
])
def test_shared_procedure_refuses_ambiguous_or_dependent_conjuncts(sentence):
    assert shared_procedure_parts(sentence) is None
    claim, = extract_claims(sentence)
    assert claim["quote"] == sentence and "compound" not in claim


def test_conjunction_uses_only_refuting_operands_for_its_decisive_evidence():
    claims = extract_claims(CONJUNCTION)
    findings = _verdicts(claims, [
        ("completed", "supported"), ("completed", "refuted"), ("completed", "insufficient"),
    ])
    compound, = aggregate_compound_findings(claims, findings)
    assert compound["review_status"] == "completed" and compound["verdict"] == "refuted"
    assert compound["evidence_ids"] == ["e-1"]
    assert [child["verdict"] for child in compound["members"]] == ["supported", "refuted", "insufficient"]
    assert [child["evidence_ids"] for child in compound["members"]] == [["e-0"], ["e-1"], ["e-2"]]
    report = {"claims": claims, "findings": findings, "original_abstract": {"version_id": "original"}}
    refresh_compound_findings(report)
    assert report["compound_findings"] == [{**compound, "original_version_id": "original"}]
    assert report["findings"] == findings


@pytest.mark.parametrize(("source", "values", "expected"), [
    ("A improved accuracy and B reduced latency.", [("completed", "supported"), ("completed", "supported")], ("completed", "supported")),
    ("A improved accuracy and B reduced latency.", [("completed", "supported"), ("completed", "insufficient")], ("completed", "insufficient")),
    ("A improved accuracy and B reduced latency.", [("completed", "refuted")], ("completed", "refuted")),
    (DISJUNCTION, [("completed", "supported")], ("completed", "supported")),
    (DISJUNCTION, [("completed", "refuted"), ("completed", "refuted")], ("completed", "refuted")),
    (DISJUNCTION, [("completed", "refuted"), ("completed", "insufficient")], ("completed", "insufficient")),
    (DISJUNCTION, [("completed", "refuted")], ("pending", None)),
    (DISJUNCTION, [("completed", "refuted"), ("manual_conflict", None)], ("manual_conflict", None)),
])
def test_grounded_operator_truth_table_preserves_unknown_states(source, values, expected):
    claims = extract_claims(source)
    assert len(claims) == 2
    compound, = aggregate_compound_findings(claims, _verdicts(claims, values))
    assert (compound["review_status"], compound["verdict"]) == expected


class ExtractionModel:
    def __init__(self, quotes, groups):
        self.quotes = quotes
        self.groups = groups

    def call(self, role, material, schema):
        assert role == "extract" and material["sentences"]
        return schema.model_validate({"claims": [
            {"sentence_id": sid, "quote": quote, "proposition": f"Dataset {quote} improved.", "kind": "result",
             "requirements": ["observed result"], "support_query": quote, "counter_query": quote}
            for sid, quote in self.quotes
        ], "compounds": self.groups})


def test_structured_compound_maps_model_indices_to_source_order_and_anchored_ids():
    groups = [{"sentence_id": "s1", "operator": "and", "claim_indexes": [2, 0, 1]}]
    claims = extract_claims(CONJUNCTION, ExtractionModel([("s1", item) for item in "ABC"], groups))
    assert claims[0]["start"] == CONJUNCTION.index(" A,") + 1
    assert claims[0]["compound"]["member_claim_ids"] == [claim["claim_id"] for claim in claims]


@pytest.mark.parametrize(("source", "operator", "indexes"), [
    (CONJUNCTION, "or", [0, 1]),
    (CONJUNCTION, "and", [0, 9]),
    (CONJUNCTION, "and", [0]),
    ("Average accuracy for datasets A and B was 88%.", "and", [0, 1]),
    ("There was no increase of 10 percentage points on datasets A or B.", "or", [0, 1]),
    ("Against the baseline, dataset B improved accuracy.", "and", [0, 1]),
])
def test_invalid_structured_group_keeps_the_full_source_assertion(source, operator, indexes):
    group = {"sentence_id": "s1", "operator": operator, "claim_indexes": indexes}
    whole, = extract_claims(source, ExtractionModel([("s1", "A"), ("s1", "B")], [group]))
    assert whole["text"] == source and whole["logic_review_required"] is True
    assert "compound" not in whole


def test_claim_ids_and_given_benchmark_contract_are_stable_without_compound():
    source = "The entity is an entity."
    raw_id = hashlib.sha256(f"4:entity:Dataset entity improved.".encode()).hexdigest()[:20]
    result, = extract_claims(source, ExtractionModel([("s1", "entity")], []))
    assert result["claim_id"] == raw_id
    supplied, = supplied_claims((("explicit-t06-id", CONJUNCTION),))
    assert supplied["claim_id"] == "explicit-t06-id" and supplied["text"] == CONJUNCTION
    assert "compound" not in supplied
