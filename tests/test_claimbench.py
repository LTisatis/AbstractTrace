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
    assert score["passages_within_k"]["1"] == {"numerator": 0, "denominator": 1, "value": 0.0}
    assert score["passages_within_k"]["3"] == {"numerator": 1, "denominator": 1, "value": 1.0}
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
    assert score["diagnostics"]["sent_packet_passage_recall"]["value"] == 1
    assert "accuracy" not in score

    pdf.write_bytes(pdf.read_bytes() + b"\n% changed after annotation preparation\n")
    assert main([
        "--data-dir", str(tmp_path / "state"), "bench", "run", "--engine", "offline", "--resume",
        "--input-manifest", str(manifests / "inputs.json"), "--output-dir", str(results),
    ]) == 1


def test_retrieval_diagnostics_distinguish_rank_cutoff_full_packet_and_missing_traces():
    large = [f"long-{i}" for i in range(9)]
    groups = {
        "oversize": large, "complete": ["s1", "s2"], "partial": ["p1", "p2", "p3"],
        "untraced": ["u1"], "no-prediction": ["m1"],
    }
    gold = {"schema_version": 1, "dataset": "claimbench-quoted-passages", "labels": [
        {"sample_id": "paper", "claim_id": claim_id,
         "passages": [{"status": "aligned", "evidence_ids": ids}]
         + ([{"status": "unmatched", "evidence_ids": []}] if claim_id == "complete" else [])}
        for claim_id, ids in groups.items()
    ]}
    traces = {"oversize": ["noise", *large], "complete": ["noise", "s1", "s2"], "partial": ["p1"]}
    predictions = {"schema_version": 1, "predictions": [
        {"sample_id": "paper", "claim_id": claim_id, "review_status": "pending", "verdict": None,
         "evidence_ids": [], "candidate_ids": ids} for claim_id, ids in traces.items()
    ] + [{"sample_id": "paper", "claim_id": "untraced", "review_status": "pending", "verdict": None,
          "evidence_ids": ["u1"]}]}

    score = score_claimbench(predictions, gold)
    assert score["alignment"] == {"aligned": 5, "unmatched": 1, "ambiguous": 0}
    assert score["cited_passage_recall"] == {"numerator": 1, "denominator": 5, "value": 0.2}
    assert score["quoted_passage_recall_at_k"] == dict.fromkeys(("1", "3", "5", "8"))
    assert score["passages_within_k"]["8"] == {"numerator": 4, "denominator": 5, "value": 0.8}
    diagnostics = score["diagnostics"]
    assert diagnostics["candidate_trace_coverage"] == {"numerator": 3, "denominator": 5, "value": 0.6}
    assert diagnostics["candidate_anchor_recall_at_k"]["8"] == {"numerator": 10, "denominator": 14, "value": 10 / 14}
    assert diagnostics["eligible_passage_recall_at_k"]["1"] == {"numerator": 0, "denominator": 0, "value": None}
    assert diagnostics["eligible_passage_recall_at_k"]["8"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert diagnostics["sent_packet_passage_recall"] == {"numerator": 2, "denominator": 3, "value": 2 / 3}
    assert diagnostics["sent_packet_anchor_recall"] == {"numerator": 12, "denominator": 14, "value": 12 / 14}

    cases = {(row["claim_id"], row["passage_number"]): row["diagnostics"] for row in score["cases"]}
    assert cases["oversize", 1] == {
        "candidate_count": 10, "matched_anchor_count_at_k": {"1": 0, "3": 2, "5": 4, "8": 7},
        "eligible_at_k": dict.fromkeys(("1", "3", "5", "8"), False),
        "sent_packet_matched_anchor_count": 9, "sent_packet_hit": True,
    }
    assert cases["complete", 2]["eligible_at_k"] is None
    for key in ("untraced", "no-prediction"):
        assert cases[key, 1]["candidate_count"] is None
        assert cases[key, 1]["sent_packet_hit"] is None


def test_empty_candidate_trace_counts_as_a_known_miss():
    predictions = {"schema_version": 1, "predictions": [
        {"sample_id": "paper", "claim_id": "1", "review_status": "pending", "verdict": None,
         "evidence_ids": [], "candidate_ids": []},
    ]}
    gold = {"schema_version": 1, "dataset": "claimbench-quoted-passages", "labels": [
        {"sample_id": "paper", "claim_id": "1", "passages": [{"status": "aligned", "evidence_ids": ["e1"]}]},
    ]}
    score = score_claimbench(predictions, gold)
    diagnostics = score["diagnostics"]
    assert diagnostics["candidate_trace_coverage"] == {"numerator": 1, "denominator": 1, "value": 1.0}
    assert diagnostics["sent_packet_passage_recall"] == {"numerator": 0, "denominator": 1, "value": 0.0}
    assert score["cases"][0]["diagnostics"]["candidate_count"] == 0


def test_layout_only_dehyphenation_keeps_original_boxes_and_does_not_join_columns(tmp_path):
    pdf = tmp_path / "split.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((60, 60), "1. Methods")
        page.insert_text((60, 90), "The inter-")
        page.insert_text((60, 104), "action estimate.")
        page.insert_text((60, 130), "The mis-")
        page.insert_text((340, 144), "aligned estimate.")
        document.save(pdf)
    annotations = tmp_path / "source.json"
    annotations.write_text(json.dumps({"annotations": [{
        "claim": "The estimate was reported.", "evidences": ["The interaction estimate.", "The misaligned estimate."],
    }]}), encoding="utf-8")

    _, gold = prepare_claimbench(
        pdf, annotations, output_dir=tmp_path, paper_family_id="family", split="dev", dataset_version="fixture",
    )
    found, cross_column = gold["labels"][0]["passages"]
    assert found["status"] == "aligned" and found["normalization"] == "line_wrap_hyphen"
    originals = {item["evidence_id"]: item for item in parse_pdf(pdf).evidence}
    assert [originals[anchor["evidence_id"]]["text"] for anchor in found["anchors"]] == [
        "The inter-", "action estimate.",
    ]
    assert cross_column["status"] == "unmatched"
