import hashlib

import pytest

from abstract_review.audit import extract_claims, refresh_compound_findings, sentences, supplied_claims
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


def test_reported_role_and_ablation_use_independent_source_ranges():
    sentence = "We ﬁnd that calibration serves as a safeguard and removing it causes unstable predictions."
    source = f"We studied the model. {sentence}"
    _, *claims = extract_claims(source)
    assert [claim["quote"] for claim in claims] == [
        "calibration serves as a safeguard", "removing it causes unstable predictions.",
    ]
    assert [claim["proposition"] for claim in claims] == [
        "We ﬁnd that calibration serves as a safeguard", "We ﬁnd that removing calibration causes unstable predictions.",
    ]
    assert all(source[claim["start"]:claim["end"]] == claim["quote"] for claim in claims)
    assert all(claim["compound"]["text"] == sentence for claim in claims)


def test_reported_absolute_metrics_and_comparative_gains_keep_their_units():
    source = (
        "We evaluate classifiers on medical transcripts, and the compact and full models achieve an average "
        "of 82.5% and 91.4% precision respectively, a 7.2% and 2.5% improvement over the prior system."
    )
    absolute, relative = extract_claims(source)
    assert absolute["quote"] == "the compact and full models achieve an average of 82.5% and 91.4% precision respectively"
    assert relative["quote"] == "a 7.2% and 2.5% improvement over the prior system"
    assert all(claim["kind"] == "quantitative" and "medical transcripts" in claim["proposition"]
               for claim in [absolute, relative])
    assert "7.2% and 2.5% improvement in precision over the prior system" in relative["proposition"]
    assert all(source[claim["start"]:claim["end"]] == claim["quote"] for claim in [absolute, relative])


@pytest.mark.parametrize("marker", ["1 1", "12\t12", "¹²"])
def test_terminal_layout_footnote_preserves_source_offsets_and_other_claim_ids(marker):
    first = "The system reduced loss by 4.2%."
    sentence = "We show that tuning improves precision, and pruning reduces latency."
    last = "The baseline also completed."
    source = f"{first} {sentence}{marker}\n{last}"
    assert sentences(source)[1]["text"] == sentence + marker
    before, *atoms, after = extract_claims(source)
    assert [claim["quote"] for claim in atoms] == ["tuning improves precision", "pruning reduces latency."]
    assert all(claim["compound"]["text"] == sentence for claim in atoms)
    assert before["claim_id"] == extract_claims(first)[0]["claim_id"]
    assert after["sentence_id"] == "s3"
    after_start = source.index(last)
    assert after["claim_id"] == hashlib.sha256(f"{after_start}:{last}:{last}".encode()).hexdigest()[:20]
    assert "4.2%" in before["proposition"]
    assert all(source[claim["start"]:claim["end"]] == claim["quote"] for claim in [before, *atoms, after])


@pytest.mark.parametrize("source", [
    "The measured rate was 2.1 1", "We used variant v.1 1", "We observed a result.1 2",
    "The model is available.1", "The upper bound was 4.2%.", "See Fig.12 12",
])
def test_ambiguous_numeric_suffixes_are_kept(source):
    claim, = extract_claims(source)
    assert claim["quote"] == source


def test_shared_negated_capability_retains_its_exception_for_both_operands():
    source = "In both settings, the finite-state controller cannot recognize mirrored tokens, nor balanced markers, unless memory grows with sequence length."
    claims = extract_claims(source)
    assert [atom["quote"] for atom in claims] == [
        "In both settings, the finite-state controller cannot recognize mirrored tokens", "nor balanced markers",
    ]
    assert [atom["proposition"] for atom in claims] == [
        f"In both settings, the finite-state controller cannot recognize {subject}, unless memory grows with sequence length"
        for subject in ["mirrored tokens", "balanced markers"]
    ]
    assert all(atom["logic_review_required"] and atom["compound"]["operator"] == "and" for atom in claims)


@pytest.mark.parametrize("source", [
    "We find that calibration acts as an alternative to averaging and removing it causes instability.",
    "We show that the fast model improves accuracy, and the large model improves recall, when deployed on edge devices.",
    "The team investigates whether the system cannot recognize mirrored tokens, nor balanced markers, unless memory grows.",
    "It is not true that the system cannot recognize mirrored tokens, nor balanced markers, unless memory grows.",
])
def test_reported_compounds_keep_ambiguous_scope_and_antecedents_together(source):
    atom, = extract_claims(source)
    assert atom["quote"] == source and "compound" not in atom


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


def test_model_whole_sentence_falls_back_to_grounded_reporting_split():
    source = "We also show—both through analysis and testing—that training improves calibration, and tuning further reduces error."
    assert [claim["quote"] for claim in extract_claims(source, ExtractionModel([("s1", source)], []))] == [
        "training improves calibration", "tuning further reduces error.",
    ]


def test_model_extracts_from_clean_sentence_without_rewriting_the_source():
    sentence = "The hardware is available."
    source = f"  {sentence}4 4"

    class FootnoteModel(ExtractionModel):
        def call(self, role, material, schema):
            assert material["sentences"] == [
                {"sentence_id": "s1", "text": sentence, "start": 2, "end": 2 + len(sentence)}
            ]
            return super().call(role, material, schema)

    claim, = extract_claims(source, FootnoteModel([("s1", "hardware")], []))
    assert source[claim["start"]:claim["end"]] == "hardware"


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
