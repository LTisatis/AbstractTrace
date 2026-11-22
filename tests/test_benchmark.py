import json
import subprocess
import sys
from pathlib import Path

import pytest

from abstract_review.benchmark import run_pdf_samples, summarize_classification
from abstract_review.review import ReviewService


GENERATOR = Path(__file__).resolve().parents[1] / "examples" / "d02" / "make_sample.py"


def manifest(path, rows):
    path.write_text(json.dumps({"schema_version": 1, "samples": rows}, ensure_ascii=False), encoding="utf-8")
    return path


def finding(claim_id, status="completed", verdict="supported", evidence=None):
    return {"claim_id": claim_id, "review_status": status, "verdict": verdict, "evidence_ids": evidence or []}


class ServiceStub:
    def __init__(self, report, status="report_ready"):
        self.report = report
        self.status = status
        self.requests = []

    def start(self, request):
        self.requests.append(request)
        return {"run_id": "test-run", "report_version_id": "test-version", "run_status": self.status}

    def export(self, run_id, version_id):
        assert (run_id, version_id) == ("test-run", "test-version")
        return self.report


def test_given_claims_use_manifest_relative_paths_and_stable_ids(tmp_path):
    pdf = tmp_path / "材料.pdf"
    pdf.write_bytes(b"stub")
    abstract = tmp_path / "另贴 摘要.txt"
    abstract.write_bytes("提供者摘要。\r\n保留换行。".encode("utf-8"))
    source = manifest(tmp_path / "only-inputs.json", [{
        "sample_id": "paper-a", "pdf": pdf.name, "abstract_file": abstract.name,
        "paper_family_id": "family-a", "split": "dev", "claims": [
            {"claim_id": "known", "text": "原文主张。"},
            {"claim_id": "missing", "text": "未完成。"},
            {"claim_id": "no-status", "text": "也未完成。"},
        ],
    }])
    service = ServiceStub({"findings": [finding("known", evidence=["e1"]), {"claim_id": "no-status", "verdict": "supported"}]})

    rows = run_pdf_samples(service, source, "offline", "unused")
    request, = service.requests
    assert request.pdf == pdf
    assert request.abstract_source == "pasted"
    assert request.pasted_abstract == "提供者摘要。\r\n保留换行。"
    assert request.provided_claims == (("known", "原文主张。"), ("missing", "未完成。"), ("no-status", "也未完成。"))
    assert rows == [
        {"sample_id": "paper-a", **finding("known", evidence=["e1"])},
        {"sample_id": "paper-a", **finding("missing", status="pending", verdict=None)},
        {"sample_id": "paper-a", **finding("no-status", status="pending", verdict=None)},
    ]


def test_without_given_claims_uses_extracted_claims_even_when_review_paused(tmp_path):
    pdf = tmp_path / "p.pdf"
    pdf.touch()
    source = manifest(tmp_path / "inputs.json", [{"sample_id": "sample", "pdf": pdf.name}])
    service = ServiceStub({"claims": [{"claim_id": "extracted"}, {"claim_id": "unfinished"}], "findings": [
        finding("extracted", status="manual_conflict", verdict=None, evidence=["e1", "e2"]),
    ]})
    rows = run_pdf_samples(service, source, "offline", "unused")
    assert service.requests[0].abstract_source == "pdf"
    assert service.requests[0].provided_claims == ()
    assert [row["review_status"] for row in rows] == ["manual_conflict", "pending"]
    assert [row["verdict"] for row in rows] == [None, None]


@pytest.mark.parametrize("invalid_claim", [{"claim_id": "a", "text": "claim", "verdict": "refuted"},
                                            {"claim_id": "a", "text": "claim", "gold_evidence": ["e1"]}])
def test_inference_rejects_gold_fields_before_starting_any_case(tmp_path, invalid_claim):
    pdf = tmp_path / "p.pdf"
    pdf.touch()
    source = manifest(tmp_path / "inputs.json", [
        {"sample_id": "first", "pdf": pdf.name},
        {"sample_id": "invalid", "pdf": pdf.name, "claims": [invalid_claim]},
    ])
    service = ServiceStub({})
    with pytest.raises(ValueError, match="only claim_id and text"):
        run_pdf_samples(service, source, "offline", "unused")
    assert service.requests == []


def test_one_paper_family_cannot_cross_development_and_final_splits(tmp_path):
    for name in ("v1.pdf", "v2.pdf"):
        (tmp_path / name).touch()
    source = manifest(tmp_path / "inputs.json", [
        {"sample_id": "old", "pdf": "v1.pdf", "paper_family_id": "family", "split": "dev"},
        {"sample_id": "new", "pdf": "v2.pdf", "paper_family_id": "family", "split": "test"},
    ])
    service = ServiceStub({})
    with pytest.raises(ValueError, match="appears in both"):
        run_pdf_samples(service, source, "offline", "unused")
    assert service.requests == []


def test_classification_uses_completed_cases_and_all_unsupported_gold():
    predictions = {"schema_version": 1, "predictions": [
        {"sample_id": "paper", **finding("a")}, {"sample_id": "paper", **finding("b")},
        {"sample_id": "paper", **finding("c", status="pending", verdict=None)},
    ]}
    gold = {"schema_version": 1, "labels": [
        {"sample_id": "paper", "claim_id": claim_id, "verdict": label, "accepted_evidence_groups": []}
        for claim_id, label in [("a", "supported"), ("b", "refuted"), ("c", "insufficient"), ("d", "refuted")]
    ]}
    summary = summarize_classification(predictions, gold)
    assert summary["completed_accuracy"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert summary["unsupported_as_support"] == {"numerator": 1, "denominator": 3, "value": 1 / 3}
    assert summary["confusion_matrix"]["refuted"] == {"supported": 1, "refuted": 0, "insufficient": 0}
    assert summary["macro_f1"] == pytest.approx(2 / 9)


def test_recovery_reuses_only_the_exact_sample_inputs(tmp_path):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"first PDF")
    source = manifest(tmp_path / "inputs.json", [{"sample_id": "sample", "pdf": pdf.name, "claims": [
        {"claim_id": "a", "text": "A local claim."},
    ]}])
    service = ServiceStub({"findings": [finding("a")]})
    recovered = {}

    def save(sample_id, input_digest, predictions):
        recovered[sample_id] = {"input_digest": input_digest, "predictions": predictions}

    first = run_pdf_samples(service, source, "offline", "model-1", recovered=recovered, on_sample=save)
    assert run_pdf_samples(service, source, "offline", "model-1", recovered=recovered) == first
    assert len(service.requests) == 1
    run_pdf_samples(service, source, "offline", "model-2", recovered=recovered)
    pdf.write_bytes(b"replaced PDF")
    run_pdf_samples(service, source, "offline", "model-1", recovered=recovered)
    assert len(service.requests) == 3


def test_paused_sample_is_not_saved_as_completed(tmp_path):
    pdf = tmp_path / "p.pdf"
    pdf.touch()
    source = manifest(tmp_path / "inputs.json", [{"sample_id": "sample", "pdf": pdf.name, "claims": [
        {"claim_id": "a", "text": "A local claim."},
    ]}])
    saved = []
    service = ServiceStub({}, status="paused")
    rows = run_pdf_samples(service, source, "offline", "model", on_sample=lambda *args: saved.append(args))
    assert not saved
    assert rows == [{"sample_id": "sample", **finding("a", status="pending", verdict=None)}]


def test_pdf_benchmark_replays_against_independently_supplied_claims(tmp_path):
    pdf = tmp_path / "paper.pdf"
    subprocess.run([sys.executable, str(GENERATOR), str(pdf)], check=True, capture_output=True, timeout=30)
    source = manifest(tmp_path / "inputs.json", [{"sample_id": "fixture", "pdf": pdf.name, "claims": [
        {"claim_id": "stated", "text": "Dataset A uses a fixed test split for the primary evaluation."},
        {"claim_id": "unestablished", "text": "The paper studies material properties on Mars."},
    ]}])
    service = ReviewService(tmp_path / "state.sqlite3")
    rows = run_pdf_samples(service, source, "offline", "unused")

    assert [row["claim_id"] for row in rows] == ["stated", "unestablished"]
    assert rows[0]["review_status"] == "completed" and rows[0]["verdict"] == "supported"
    assert rows[0]["evidence_ids"]
    assert rows[1]["review_status"] == "pending" and rows[1]["verdict"] is None
