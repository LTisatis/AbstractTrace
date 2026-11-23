import json

import pytest

from abstract_review.scifact import prepare_scifact, run_scifact, score_scifact


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


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
