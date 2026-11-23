import json

import pymupdf

from abstract_review.claimbench import prepare_claimbench, score_claimbench
from abstract_review.cli import main
from abstract_review.ingest import parse_pdf


def _paper(path):
    with pymupdf.open() as doc:
        for _ in range(2):
            page = doc.new_page()
            page.insert_text((60, 60), "1. Methods")
            page.insert_text((60, 90), "The protocol assigned a fixed sample size.")
        page.insert_text((60, 120), "Every site used the same threshold.")
        doc.save(path)


def test_alignment_uses_exact_contiguous_page_records_and_refuses_ambiguous_quotes(tmp_path):
    pdf = tmp_path / "paper.pdf"
    _paper(pdf)
    annotations = tmp_path / "source.json"
    annotations.write_text(json.dumps({"annotations": [{
        "claim": "The study used a fixed method.",
        "evidences": ["fixed sample size.\nEvery site", "the protocol", "fixed sample size.", "Variable size."],
    }]}), encoding="utf-8")
    inputs, gold = prepare_claimbench(
        pdf, annotations, output_dir=tmp_path, paper_family_id="family", split="dev", dataset_version="fixture",
    )
    assert "Every site" not in json.dumps(inputs)
    assert "verdict" not in json.dumps(gold)
    aligned, case_mismatch, ambiguous, absent = gold["labels"][0]["passages"]
    assert aligned["status"] == "aligned" and len(aligned["evidence_ids"]) == 2
    source = parse_pdf(pdf)
    reference = {item["evidence_id"]: item for item in source.evidence}
    assert gold["pdf_sha256"] == source.document["sha256"]
    for anchor in aligned["anchors"]:
        assert anchor["page_index"] == 1
        assert anchor["bbox"] == reference[anchor["evidence_id"]]["locator"]["bbox"]
    assert [row["status"] for row in (case_mismatch, ambiguous, absent)] == ["unmatched", "ambiguous", "unmatched"]
    assert not ambiguous["anchors"]

    ids = aligned["evidence_ids"]
    predictions = {"schema_version": 1, "predictions": [{
        "sample_id": inputs["samples"][0]["sample_id"], "claim_id": "1", "review_status": "pending",
        "verdict": None, "evidence_ids": ids[:1], "candidate_ids": ["unrelated", *ids],
    }]}
    score = score_claimbench(predictions, gold)
    assert score["alignment_coverage"]["value"] == 1 / 4
    assert score["quoted_passage_recall_at_k"]["1"]["value"] == 0
    assert score["quoted_passage_recall_at_k"]["3"]["value"] == 1
    assert score["cited_passage_recall"]["value"] == 0


def test_flat_claimbench_format_cli_runs_normal_pdf_core_then_localization_score(tmp_path):
    pdf = tmp_path / "paper.pdf"
    _paper(pdf)
    annotations = tmp_path / "source.json"
    annotations.write_text(json.dumps([{
        "claim_id": 42, "claim_text": "Every site used the same threshold.",
        "evidence_text": "Every site used the same threshold.",
        "justification_conclusion": "GOLD ONLY",
    }]), encoding="utf-8")
    manifests = tmp_path / "manifests"
    results = tmp_path / "results"
    assert main([
        "bench", "prepare-claimbench", "--pdf", str(pdf), "--annotations", str(annotations),
        "--paper-family-id", "family", "--split", "dev", "--dataset-version", "fixture",
        "--output-dir", str(manifests),
    ]) == 0
    assert "GOLD ONLY" not in (manifests / "inputs.json").read_text(encoding="utf-8")
    assert main([
        "--data-dir", str(tmp_path / "state"), "bench", "run", "--engine", "offline",
        "--input-manifest", str(manifests / "inputs.json"), "--output-dir", str(results),
    ]) == 0
    score_file = results / "score.json"
    assert main([
        "bench", "score", "--dataset", "claimbench", "--predictions", str(results / "predictions.json"),
        "--gold-manifest", str(manifests / "gold.json"), "--output", str(score_file),
    ]) == 0
    score = json.loads(score_file.read_text(encoding="utf-8"))
    assert score["cited_passage_recall"]["value"] == 1
    assert "accuracy" not in score

    pdf.write_bytes(pdf.read_bytes() + b"\n% changed after annotation preparation\n")
    assert main([
        "--data-dir", str(tmp_path / "state"), "bench", "run", "--engine", "offline", "--resume",
        "--input-manifest", str(manifests / "inputs.json"), "--output-dir", str(results),
    ]) == 1
