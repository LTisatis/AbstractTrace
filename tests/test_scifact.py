import json

import pytest

from abstract_review.scifact import prepare_scifact, run_scifact, score_scifact


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_scifact_provider_adapter_preserves_original_sentence_indices_and_separates_gold(tmp_path):
    corpus = write_jsonl(tmp_path / "corpus.jsonl", [
        {"doc_id": 11, "title": "A fixed sample", "abstract": ["Unrelated sentence.", "Sample size was fixed."],
         "structured": False},
        {"doc_id": 13, "title": "Control", "abstract": ["This abstract studies a different problem."],
         "structured": False},
    ])
    claims = write_jsonl(tmp_path / "claims_dev.jsonl", [
        {"id": 52, "claim": "Sample size was fixed.", "evidence": {"11": [{"sentences": [1], "label": "SUPPORT"}]},
         "cited_doc_ids": [11, 13, 13]},
    ])

    inputs, gold = prepare_scifact(corpus, claims, split="dev", dataset_version="fixture-v1")
    assert "SUPPORT" not in json.dumps(inputs)
    assert inputs["source_sha256"]["claims"]
    assert gold["labels"][1]["verdict"] == "insufficient"
    predictions = run_scifact(inputs, "offline", "unused")

    assert [row["sample_id"] for row in predictions] == ["scifact:52:11", "scifact:52:13"]
    assert predictions[0]["evidence_ids"] == ["scifact:11:1"]
    assert predictions[1]["review_status"] == "pending"
    scored = score_scifact({"schema_version": 1, "predictions": predictions}, gold)
    assert scored["scifact_cited_abstracts"]["sentence_label"]["f1"]["value"] == 1
    assert scored["completed_coverage"]["value"] == 0.5
    inputs["samples"][-1]["evidence"] = {"label": "FORBIDDEN"}
    with pytest.raises(ValueError, match="only sample_id"):
        run_scifact(inputs, "openai", "unused", make_model=lambda *_a, **_kw: pytest.fail("started inference"))


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
