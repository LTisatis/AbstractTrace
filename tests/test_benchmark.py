import json
import subprocess
import sys
from pathlib import Path

import pytest

from abstract_review.scoring import score_predictions, semantic_manifest_sha256

from abstract_review.benchmark import benchmark_configuration, run_pdf_samples, summarize_classification
from abstract_review.benchmark_costs import score_costs
from abstract_review.cli import main
from abstract_review.models import ModelUnavailable
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


def test_pdf_benchmark_uses_and_binds_opt_in_local_models(tmp_path):
    pdf = tmp_path / "test.pdf"
    pdf.touch()
    source = manifest(tmp_path / "inputs.json", [{"sample_id": "paper", "pdf": pdf.name}])
    embed = tmp_path / "embed"
    rerank = tmp_path / "rerank"
    for path in (embed, rerank):
        path.mkdir()
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"stub weights")
    (embed / "modules.json").write_text("[]", encoding="utf-8")
    configuration = benchmark_configuration("offline", "test", embed_model=embed, rerank_model=rerank)
    service = ServiceStub({"findings": []})
    run_pdf_samples(service, source, "offline", "test", configuration=configuration,
                    embed_model=embed, rerank_model=rerank)
    request, = service.requests
    assert (request.embed_model, request.rerank_model) == (embed, rerank)
    assert request.expected_retrieval_fingerprints == configuration["retrieval_model_sha256"]
    (rerank / "model.safetensors").write_bytes(b"different defaults")
    with pytest.raises(ValueError, match="does not match"):
        run_pdf_samples(service, source, "offline", "test", configuration=configuration,
                        embed_model=embed, rerank_model=rerank)
    assert len(service.requests) == 1


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
        (tmp_path / name).write_bytes(name.encode())
    source = manifest(tmp_path / "inputs.json", [
        {"sample_id": "old", "pdf": "v1.pdf", "paper_family_id": "family", "split": "dev"},
        {"sample_id": "new", "pdf": "v2.pdf", "paper_family_id": "family", "split": "test"},
    ])
    service = ServiceStub({})
    with pytest.raises(ValueError, match="appears in both"):
        run_pdf_samples(service, source, "offline", "unused")
    assert service.requests == []


@pytest.mark.parametrize("second_split", ["dev", "test"])
def test_copied_pdf_bytes_cannot_cross_splits_under_different_families(tmp_path, second_split):
    for name in ("source.pdf", "copy.pdf"):
        (tmp_path / name).write_bytes(b"identical PDF bytes")
    source = manifest(tmp_path / "inputs.json", [
        {"sample_id": "a", "pdf": "source.pdf", "paper_family_id": "family-one", "split": "dev"},
        {"sample_id": "b", "pdf": "copy.pdf", "paper_family_id": "family-two", "split": second_split},
    ])
    service = ServiceStub({})

    if second_split == "test":
        with pytest.raises(ValueError, match="pdf SHA-256.*appears in both"):
            run_pdf_samples(service, source, "offline", "unused")
        assert service.requests == []
    else:
        run_pdf_samples(service, source, "offline", "unused")
        assert len(service.requests) == 2


def test_declared_pdf_checksum_is_validated_before_reviewing_an_earlier_sample(tmp_path):
    for name in ("valid.pdf", "later.pdf"):
        (tmp_path / name).write_bytes(name.encode())
    source = manifest(tmp_path / "inputs.json", [
        {"sample_id": "valid", "pdf": "valid.pdf"},
        {"sample_id": "invalid", "pdf": "later.pdf", "pdf_sha256": "0" * 64},
    ])
    service = ServiceStub({})

    with pytest.raises(ValueError, match="invalid.*manifest SHA-256"):
        run_pdf_samples(service, source, "openai", "unused")
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


def test_run_record_diagnoses_execution_without_changing_strict_metrics():
    predictions = {"schema_version": 1, "predictions": [
        {"sample_id": "processed", **finding("correct")},
        {"sample_id": "processed", **finding("wrong")},
        {"sample_id": "processed", **finding("pending", "pending", None)},
        {"sample_id": "paused", **finding("pending", "pending", None)},
    ]}
    gold = {"schema_version": 1, "labels": [
        {"sample_id": sample_id, "claim_id": claim_id, "verdict": label, "accepted_evidence_groups": []}
        for sample_id, claim_id, label in [
            ("processed", "correct", "supported"), ("processed", "wrong", "refuted"),
            ("processed", "pending", "refuted"), ("processed", "missing", "refuted"),
            ("paused", "pending", "refuted"), ("untouched", "missing", "refuted"),
            ("outside", "missing", "refuted"),
        ]
    ]}
    record = {"schema_version": 1, "predictions_sha256": semantic_manifest_sha256(predictions),
              "input_manifest_sha256": "1" * 64, "input_sample_ids": ["processed", "paused", "untouched"],
              "prediction_count": 4, "samples": [
                  {"sample_id": "processed", "run_status": "report_ready", "prediction_count": 3},
                  {"sample_id": "paused", "run_status": "paused", "prediction_count": 1},
              ]}

    scored = score_predictions(predictions, gold, run_record=record)
    diagnostics = scored.pop("run_diagnostics")
    assert scored == score_predictions(predictions, gold)
    assert diagnostics["outcome_counts"] == {
        "completed_correct": 1, "completed_incorrect": 1, "unresolved": 1,
        "missing_after_execution": 1, "paused": 1, "unexecuted": 1, "outside_input": 1,
    }
    assert diagnostics["sample_execution_coverage"] == {"numerator": 2, "denominator": 4, "value": 0.5}

    predictions["predictions"][0]["verdict"] = "refuted"
    with pytest.raises(ValueError, match="predictions SHA-256 does not match"):
        score_predictions(predictions, gold, run_record=record)


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
    assert rows[0]["review_status"] == "pending" and rows[0]["verdict"] is None
    assert rows[0]["evidence_ids"] == []
    assert rows[0]["candidate_ids"]
    assert rows[1]["review_status"] == "pending" and rows[1]["verdict"] is None


@pytest.mark.parametrize(("proposal", "blockers", "attempt_status", "stop", "phase"), [
    ("pending", [], "checked", "requires_review", "model_pending"),
    ("refuted", ["missing_direct_contradiction"], "checked", "requires_review", "verdict_blocked"),
    ("refuted", [], "citation_error", "invalid_judge_reference", "citation_validation"),
    (None, [], "model_paused", "budget_exhausted", "execution"),
])
def test_score_distinguishes_decision_traces_with_the_same_final_state(proposal, blockers, attempt_status, stop, phase):
    trace = {"attempts": [{"material_sha256": "2" * 64, "repair": False,
                           "proposed_verdict": proposal, "status": attempt_status}],
             "blocker_codes": blockers, "missing_requirements": ["test scope"]}
    row = {"sample_id": "sample", **finding("claim", status="pending", verdict=None),
           "candidate_ids": ["e1"], "stop_reason": stop, "decision_trace": trace}
    gold = {"schema_version": 1, "labels": [
        {"sample_id": "sample", "claim_id": "claim", "verdict": "refuted", "accepted_evidence_groups": [["e1"]]},
    ]}
    scored = score_predictions({"schema_version": 1, "predictions": [row]}, gold)
    assert scored["errors"][0]["phase"] == phase
    assert scored["cases"][0]["decision_trace"] == trace


def test_predictions_use_sent_order_and_keep_measurement_metadata_on_sidecar(tmp_path):
    pdf = tmp_path / "source.pdf"
    pdf.touch()
    source = manifest(tmp_path / "inputs.json", [{"sample_id": "sample", "pdf": pdf.name}])
    anchor = {"sentence_id": "s1", "start": 0, "end": 4, "proposition": "Test"}
    service = ServiceStub({
        "claims": [{"claim_id": "c1", **anchor}],
        "findings": [
            dict(finding("c1", evidence=["cited"]), candidate_ids=["b", "a", "cited"],
                 sent_candidate_ids=["a", "b", "cited"], stop_reason="verdict"),
        ],
        "usage": {"external_requests": 2, "model_events": [{"total_tokens": 9}, {"total_tokens": 12}]},
    })
    results, recovered = [], {}

    def save(sample_id, digest, predictions):
        recovered[sample_id] = {"input_digest": digest, "predictions": predictions}

    rows = run_pdf_samples(service, source, "offline", "model", on_sample=save, on_result=results.append)
    replay = run_pdf_samples(service, source, "offline", "model", recovered=recovered, on_result=results.append)
    assert rows == replay
    assert rows[0]["candidate_ids"] == ["a", "b", "cited"]
    assert rows[0]["claim_anchor"] == anchor
    assert results[0]["model_requests"] == 2 and results[0]["tokens"]["total_tokens"] == 21
    assert results[0]["run_id"] == "test-run"
    assert results[1]["cache_hit"] is True and "model_requests" not in results[1]


def test_pdf_bench_resume_preserves_the_exact_version_for_independent_scoring(tmp_path):
    pdf = tmp_path / "paper.pdf"
    subprocess.run([sys.executable, str(GENERATOR), str(pdf)], check=True, capture_output=True, timeout=30)
    abstract = tmp_path / "abstract.txt"
    abstract.write_text("On dataset A, accuracy increased by 10 percentage points.", encoding="utf-8")
    inputs = manifest(tmp_path / "inputs.json", [{"sample_id": "sample", "pdf": pdf.name, "abstract_file": abstract.name}])
    output = tmp_path / "benchmark"
    args = ["--data-dir", str(tmp_path / "runs"), "bench", "run", "--engine", "offline",
            "--input-manifest", str(inputs), "--output-dir", str(output), "--resume"]
    assert main(args) == 0
    first, = json.loads((output / "run.json").read_bytes())["samples"]
    review = first["review"]
    change, = review["revisions"]
    assert review["revision_ids"] == [change["revision_id"]]
    assert "8 percentage points" in review["reviewed_text"] and change["status"] == "passed"
    assert review["abstract_review"]["version_id"] == review["version_id"]
    assert {review["evidence_roles"][eid] for eid in review["calculations"][0]["evidence_ids"]} == {"value"}

    assert main(args) == 0
    cached, = json.loads((output / "run.json").read_bytes())["samples"]
    assert cached == {**first, "cache_hit": True}

    annotations = output / "review-gold.json"
    annotations.write_text(json.dumps({"schema_version": 1, "revision_checks": [
        {"sample_id": "sample", "version_id": review["version_id"], "revision_id": change["revision_id"],
         "corrected": True, "facts_preserved": True, "unsupported_added": False},
    ]}), encoding="utf-8")
    pricing = output / "prices.json"
    pricing.write_text(json.dumps({"schema_version": 1, "pricing_id": "synthetic", "models": {}}), encoding="utf-8")
    scores = output / "review-scores.json"
    assert main(["bench", "score-reviews", "--run-record", str(output / "run.json"),
                 "--gold-manifest", str(annotations), "--pricing", str(pricing), "--output", str(scores)]) == 0
    assert json.loads(scores.read_bytes())["estimated_cost"]["known_cost_usd"] == "0"


def test_pdf_bench_resume_keeps_paused_run_and_its_existing_budget(tmp_path, monkeypatch, capsys):
    pdf = tmp_path / "paper.pdf"
    subprocess.run([sys.executable, str(GENERATOR), str(pdf)], check=True, capture_output=True, timeout=30)
    abstract = tmp_path / "abstract.txt"
    abstract.write_text("On dataset A, accuracy increased by 10 percentage points.", encoding="utf-8")
    inputs = manifest(tmp_path / "inputs.json", [
        {"sample_id": "sample", "pdf": pdf.name, "abstract_file": abstract.name, "claims": [
            {"claim_id": "a", "text": "Dataset A uses a fixed test split for the primary evaluation."},
        ]},
    ])
    output = tmp_path / "benchmark"
    args = ["--data-dir", str(tmp_path / "runs"), "bench", "run", "--engine", "openai",
            "--input-manifest", str(inputs), "--output-dir", str(output)]
    attempted = []

    def responses(self, report, **kwargs):
        configuration = report["configuration"]
        attempted.append((report["run_id"], configuration["max_requests"], configuration["max_seconds"]))
        if len(attempted) == 1:
            raise ModelUnavailable("test pause")
        return None

    monkeypatch.setattr(ReviewService, "_responses", responses)
    assert main([*args, "--model", "pinned"]) == 0
    first, = json.loads((output / "run.json").read_bytes())["samples"]
    checkpoints = output / "sample-checkpoints.json"
    paused = json.loads(checkpoints.read_bytes())["sample"]
    assert paused["pause"] == {key: first[key] for key in ("run_id", "run_status", "report_version_id")}
    assert first["run_status"] == "paused"

    assert main([*args, "--model", "different", "--resume"]) == 1
    assert "inputs or configuration changed" in capsys.readouterr().err
    assert len(attempted) == 1 and json.loads(checkpoints.read_bytes())["sample"] == paused

    assert main([*args, "--model", "pinned", "--resume"]) == 0
    completed, = json.loads((output / "run.json").read_bytes())["samples"]
    assert completed["run_status"] == "report_ready" and completed["run_id"] == first["run_id"]
    assert attempted == [(first["run_id"], 80, 900)] * 2
    checkpoint = json.loads(checkpoints.read_bytes())["sample"]
    assert checkpoint["input_digest"] == paused["input_digest"] and "pause" not in checkpoint
    assert checkpoint["result"] == completed


def test_cost_only_prices_recorded_tokens_and_keeps_unknown_requests_visible():
    prices = {"schema_version": 1, "pricing_id": "synthetic-v1", "models": {
        "fixed": {"input_per_million": "2", "cached_input_per_million": "0.5", "output_per_million": "4"},
    }}
    record = {"samples": [
        {"sample_id": "live", "model_requests": 2, "request_usage": [
            {"model": "fixed", "input_tokens": 1000, "cached_input_tokens": 400, "output_tokens": 200},
            {"model": "fixed", "status": "outcome_unknown"},
        ]},
        {"sample_id": "legacy"},
    ]}
    result = score_costs(record, prices)
    assert result["known_cost_usd"] == "0.0022"
    assert (result["priced_requests"], result["unpriced_requests"], result["samples_without_usage"]) == (1, 1, 1)
    assert result["complete"] is False


def test_cost_prices_cache_writes_by_ttl_without_double_counting_input():
    prices = {"schema_version": 1, "pricing_id": "test", "models": {
        "fixed": {"input_per_million": "2", "cached_input_per_million": "0.1", "output_per_million": "10",
                  "cache_write_30m_per_million": "2.5", "cache_write_12h_per_million": "5"},
    }}
    event = {"model": "fixed", "input_tokens": 1000, "cached_input_tokens": 200, "output_tokens": 10,
             "cache_write_input_tokens": 600, "cache_write_30m_input_tokens": 500,
             "cache_write_12h_input_tokens": 100}
    record = {"samples": [{"sample_id": "live", "request_usage": [
        event, {**event, "cache_write_input_tokens": 700},
    ]}]}
    result = score_costs(record, prices)
    assert result["known_cost_usd"] == "0.00227"
    assert (result["priced_requests"], result["unpriced_requests"]) == (1, 1)
    del prices["models"]["fixed"]["cache_write_12h_per_million"]
    assert score_costs(record, prices)["unpriced_requests"] == 2


def test_cost_counts_explicit_zero_and_keeps_legacy_cache_write_usage_unknown():
    prices = {"schema_version": 1, "pricing_id": "test", "models": {
        "fixed": {"input_per_million": "2", "cached_input_per_million": "0.1",
                  "output_per_million": "10", "cache_write_30m_per_million": "2.5"},
    }}
    event = {"model": "fixed", "input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 10}
    record = {"samples": [{"sample_id": "live", "request_usage": [
        event, {**event, "cache_write_input_tokens": 0},
    ]}]}
    result = score_costs(record, prices)
    assert result["known_cost_usd"] == "0.0003"
    assert (result["priced_requests"], result["unpriced_requests"]) == (1, 1)
    assert result["complete"] is False


def test_review_scoring_does_not_report_zero_cost_for_unmeasured_dataset(tmp_path, capsys):
    record = tmp_path / "run.json"
    record.write_text(json.dumps({"schema_version": 1, "prediction_count": 1, "samples": [],
                                  "configuration": {"engine": "openai", "dataset": "scifact-cited-abstracts"}}),
                      encoding="utf-8")
    gold = tmp_path / "gold.json"
    gold.write_text('{"schema_version": 1}', encoding="utf-8")
    pricing = tmp_path / "rates.json"
    pricing.write_text('{"schema_version": 1, "pricing_id": "fixture", "models": {}}', encoding="utf-8")
    output = tmp_path / "scores.json"
    assert main(["bench", "score-reviews", "--run-record", str(record), "--gold-manifest", str(gold),
                 "--pricing", str(pricing), "--output", str(output)]) == 1
    assert "requires a PDF benchmark run record" in capsys.readouterr().err
    assert not output.exists()
