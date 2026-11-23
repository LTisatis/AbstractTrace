import json
from contextlib import closing

import pytest

from abstract_review.audit import review_claim, supplied_claims
from abstract_review.evidence import EvidenceIndex
from abstract_review.scifact import prepare_scifact, run_scifact, score_scifact


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class AbstractJudge:
    def __init__(self, verdict, *, status=None, proof_scope="same", follow_up=(), conflicting=False):
        self.verdict = verdict
        self.status = status
        self.proof_scope = proof_scope
        self.follow_up = list(follow_up)
        self.conflicting = conflicting
        self.requests = []

    def call(self, stage, material, schema):
        assert stage == "judge"
        self.requests.append(material)
        required, = material["requirements"]
        rationale = "scifact:11:2"
        groups = []
        if self.verdict in {"supported", "refuted"}:
            role = "support" if self.verdict == "supported" else "refute"
            groups = [{"role": role, "evidence_ids": [rationale]}]
        elif self.conflicting:
            groups = [{"role": "support", "evidence_ids": [rationale]}]
        status = self.status or ("searched_missing" if self.verdict == "insufficient" else "found")
        result = {"verdict": self.verdict, "reason": "The provided abstract was checked for the outcome.",
                  "evidence_groups": groups, "missing_fields": [], "follow_up_queries": self.follow_up,
                  "coverage": [{"requirement": required, "status": status,
                                "evidence_ids": [rationale] if status == "found" else []}]}
        if self.verdict == "refuted" and self.proof_scope is not None:
            result["direct_contradiction"] = {
                "evidence_ids": [rationale], "scope_evidence_ids": [rationale], "relation": "opposite_result",
                "subject_and_conditions": self.proof_scope,
                "scope_reason": "The source gives the outcome in the stated cohort.", "supplementary_fields": [],
            }
        return schema.model_validate(result)


def _one_scifact_abstract(tmp_path, sentences):
    corpus = write_jsonl(tmp_path / "scope-corpus.jsonl", [{"doc_id": 11, "abstract": sentences}])
    claims = write_jsonl(tmp_path / "scope-claims.jsonl", [
        {"id": 52, "claim": "Treatment lowered the target observation rate.",
         "evidence": {}, "cited_doc_ids": [11]},
    ])
    return prepare_scifact(corpus, claims, split="dev", dataset_version="scope-fixture")[0]


def test_scifact_can_classify_missing_evidence_after_sending_the_entire_abstract(tmp_path):
    rows = ["Treatment was randomized.", " ", "The protocol lacked an outcome definition.",
            *(f"The abstract reported an enrollment count for site {site}." for site in range(7))]
    model = AbstractJudge("insufficient")
    prediction, = run_scifact(_one_scifact_abstract(tmp_path, rows), "openai", "unused",
                              make_model=lambda *_a, **_kw: model)

    request, = model.requests
    assert [(row["evidence_id"], row["text"]) for row in request["evidence"]] == [
        (f"scifact:11:{i}", text) for i, text in enumerate(rows) if text.strip()
    ]
    assert request["evidence_boundary"] == {
        "kind": "cited_abstract", "complete": True, "sentence_count": 9, "supplied_sentence_count": 9,
    }
    assert prediction["review_status"] == "completed"
    assert prediction["verdict"] == "insufficient"
    assert prediction["evidence_ids"] == []
    assert prediction["stop_reason"] == "source_range_exhausted"


@pytest.mark.parametrize("options", [
    {"status": "unchecked"}, {"follow_up": ["Inspect the full Methods section"]}, {"conflicting": True},
])
def test_scifact_does_not_complete_insufficient_with_unresolved_or_conflicting_output(tmp_path, options):
    model = AbstractJudge("insufficient", **options)
    inputs = _one_scifact_abstract(tmp_path, ["Treatment was randomized.", " ", "The records tracked enrollment only."])
    prediction, = run_scifact(inputs, "openai", "unused", make_model=lambda *_a, **_kw: model)

    assert prediction["review_status"] == "pending" and prediction["verdict"] is None
    assert len(model.requests) == 1


@pytest.mark.parametrize(("sentence_count", "repeat"), [(17, 1), (3, 40)])
def test_scifact_cannot_complete_insufficient_on_an_abstract_larger_than_the_review_budget(
    tmp_path, sentence_count, repeat,
):
    sentences = [f"Treatment enrollment and observation summary for site {site}. " * repeat
                 for site in range(sentence_count)]
    model = AbstractJudge("insufficient")
    inputs = _one_scifact_abstract(tmp_path, sentences)
    prediction, = run_scifact(inputs, "openai", "unused", make_model=lambda *_a, **_kw: model)

    request, = model.requests
    assert len(request["evidence"]) <= 8
    assert sum(len(row["text"]) for row in request["evidence"]) <= 5500
    assert request["evidence_boundary"] == {
        "kind": "cited_abstract", "complete": False, "sentence_count": sentence_count,
        "supplied_sentence_count": len(request["evidence"]),
    }
    assert prediction["review_status"] == "pending" and prediction["verdict"] is None


def test_scifact_empty_abstract_is_not_a_completed_insufficient_verdict(tmp_path):
    model = AbstractJudge("insufficient")
    prediction, = run_scifact(_one_scifact_abstract(tmp_path, ["", " "]), "openai", "unused",
                              make_model=lambda *_a, **_kw: model)

    assert prediction["review_status"] == "pending" and prediction["verdict"] is None
    assert model.requests == []


@pytest.mark.parametrize(("verdict", "proof_scope", "expected"), [
    ("supported", None, "supported"),
    ("refuted", "same", "refuted"),
    ("refuted", "different", None),
    ("refuted", None, None),
])
def test_scifact_preserves_grounded_citations_and_rejects_scope_mismatched_refutation(
    tmp_path, verdict, proof_scope, expected,
):
    evidence = "Treatment lowered the target observation rate."
    if verdict == "refuted":
        evidence = "Treatment increased the target observation rate."
    if proof_scope == "different":
        evidence = "Treatment increased a different observation rate in another population."
    inputs = _one_scifact_abstract(tmp_path, ["Treatment was randomized.", " ", evidence])
    model = AbstractJudge(verdict, proof_scope=proof_scope)
    prediction, = run_scifact(inputs, "openai", "unused", make_model=lambda *_a, **_kw: model)

    assert prediction["verdict"] == expected
    if expected is not None:
        assert prediction["review_status"] == "completed" and prediction["evidence_ids"] == ["scifact:11:2"]
    else:
        assert prediction["review_status"] == "pending"


def test_ordinary_pdf_review_does_not_infer_insufficiency_from_ranked_abstract_hits():
    model = AbstractJudge("insufficient")
    claim, = supplied_claims((("52", "Treatment lowered the target observation rate."),))
    row = {"evidence_id": "scifact:11:2", "doc_id": "11", "kind": "body_text",
           "text": "Treatment was randomized.", "locator": {"page_index": 0, "section": "abstract"}}
    with closing(EvidenceIndex([row])) as index:
        finding, _ = review_claim(claim, index, model, documents=[{"doc_id": "11", "role": "main"}])

    assert finding["review_status"] == "pending" and finding["verdict"] is None
    assert finding["coverage"][0]["status"] == "unchecked"
    assert "evidence_boundary" not in model.requests[0]


def test_partial_abstract_ranking_never_sends_other_indexed_sources():
    model = AbstractJudge("insufficient")
    claim, = supplied_claims((("52", "Treatment lowered the target observation rate."),))
    allowed = [
        {"evidence_id": f"scifact:11:{i}", "doc_id": "11", "kind": "body_text",
         "text": f"Treatment participants at site {i} completed the observation period.",
         "locator": {"page_index": 0, "section": "abstract", "sentence_index": i}}
        for i in range(17)
    ]
    unrelated = [
        {"evidence_id": f"a_{name}_{i}", "doc_id": doc, "kind": "body_text", "text": claim["proposition"],
         "locator": {"page_index": 0, "section": section}}
        for name, doc, section in (("other_doc", "19", "abstract"), ("other_section", "11", "Results"))
        for i in range(10)
    ]
    allowed_ids = [row["evidence_id"] for row in allowed]
    with closing(EvidenceIndex([*allowed, *unrelated])) as index:
        finding, _ = review_claim(claim, index, model, cited_abstract_ids=allowed_ids)

    request, = model.requests
    transmitted = [row["evidence_id"] for row in request["evidence"]]
    assert transmitted == finding["sent_candidate_ids"]
    assert 0 < len(transmitted) <= 8 and set(transmitted) <= set(allowed_ids)
    assert request["evidence_boundary"]["complete"] is False


@pytest.mark.parametrize(("claim_id", "split"), [(0, "train"), (52, "dev")])
def test_scifact_provider_adapter_preserves_original_sentence_indices_and_separates_gold(tmp_path, claim_id, split):
    corpus = write_jsonl(tmp_path / "corpus.jsonl", [
        {"doc_id": 11, "title": "A fixed sample", "abstract": ["Unrelated sentence.", "Sample size was fixed."],
         "structured": False},
        {"doc_id": 13, "title": "Control", "abstract": ["This abstract studies a different problem."],
         "structured": False},
    ])
    claims = write_jsonl(tmp_path / f"claims_{split}.jsonl", [
        {"id": claim_id, "claim": "Sample size was fixed.",
         "evidence": {"11": [{"sentences": [1], "label": "SUPPORT"}]}, "cited_doc_ids": [11, 13, 13]},
    ])

    inputs, gold = prepare_scifact(corpus, claims, split=split, dataset_version="fixture-v1")
    assert "SUPPORT" not in json.dumps(inputs)
    assert inputs["source_sha256"]["claims"]
    assert gold["labels"][1]["verdict"] == "insufficient"
    predictions = run_scifact(inputs, "offline", "unused")

    assert [row["sample_id"] for row in predictions] == [f"scifact:{claim_id}:11", f"scifact:{claim_id}:13"]
    assert [row["claim_id"] for row in predictions] == [str(claim_id)] * 2
    assert predictions[0]["evidence_ids"] == ["scifact:11:1"]
    assert predictions[1]["review_status"] == "pending"
    scored = score_scifact({"schema_version": 1, "predictions": predictions}, gold)
    assert scored["scifact_cited_abstracts"]["sentence_label"]["f1"]["value"] == 1
    assert scored["completed_coverage"]["value"] == 0.5
    inputs["samples"][-1]["evidence"] = {"label": "FORBIDDEN"}
    with pytest.raises(ValueError, match="only sample_id"):
        run_scifact(inputs, "openai", "unused", make_model=lambda *_a, **_kw: pytest.fail("started inference"))


@pytest.mark.parametrize("invalid_claim_id", [-1, False, 0.0, "0"])
def test_scifact_requires_integer_claim_ids_at_ingest_and_run(tmp_path, invalid_claim_id):
    corpus = write_jsonl(tmp_path / "corpus.jsonl", [{"doc_id": 11, "abstract": ["Sample size was fixed."]}])
    claim = {"id": 0, "claim": "Sample size was fixed.", "evidence": {}, "cited_doc_ids": [11]}
    path = tmp_path / "claims_train.jsonl"
    inputs, _ = prepare_scifact(corpus, write_jsonl(path, [claim]), split="train", dataset_version="fixture-v1")

    claim["id"] = invalid_claim_id
    with pytest.raises(ValueError, match="claim id must be a non-negative integer ID"):
        prepare_scifact(corpus, write_jsonl(path, [claim]), split="train", dataset_version="fixture-v1")

    inputs["samples"][0]["claim_id"] = invalid_claim_id
    with pytest.raises(ValueError, match="claim_id must be a non-negative integer ID"):
        run_scifact(inputs, "offline", "unused")


def test_scifact_zero_is_still_invalid_as_document_id(tmp_path):
    document = {"doc_id": 11, "abstract": ["Sample size was fixed."]}
    corpus = write_jsonl(tmp_path / "corpus.jsonl", [document])
    claim = {"id": 0, "claim": "Sample size was fixed.", "evidence": {}, "cited_doc_ids": [11]}
    claims = write_jsonl(tmp_path / "claims_train.jsonl", [claim])
    inputs, _ = prepare_scifact(corpus, claims, split="train", dataset_version="fixture-v1")

    with pytest.raises(ValueError, match="cited_doc_ids entry must be a positive integer ID"):
        prepare_scifact(corpus, write_jsonl(claims, [{**claim, "cited_doc_ids": [0]}]),
                        split="train", dataset_version="fixture-v1")

    inputs["documents"][0]["doc_id"] = 0
    with pytest.raises(ValueError, match="doc_id must be a positive integer ID"):
        run_scifact(inputs, "offline", "unused")


def test_scifact_abstract_match_caps_citations_at_three_and_sentence_scoring_requires_full_group():
    good = ["scifact:11:0", "scifact:11:1"]
    extra = ["scifact:11:2", "scifact:11:3", "scifact:11:4"]
    predictions = {"schema_version": 1, "predictions": [{
        "sample_id": "scifact:52:11", "claim_id": "52", "review_status": "completed", "verdict": "supported",
        "evidence_ids": [good[0], *extra, good[1]],
    }]}
    gold = {"schema_version": 1, "labels": [{
        "sample_id": "scifact:52:11", "claim_id": "52", "verdict": "supported", "accepted_evidence_groups": [good],
    }]}
    result = score_scifact(predictions, gold)["scifact_cited_abstracts"]

    assert result["abstract_rationalized"]["precision"]["value"] == 0
    assert result["sentence_label"]["precision"]["value"] == 2 / 5
    assert result["sentence_label"]["recall"]["value"] == 1
