import json
from types import SimpleNamespace

import pytest

from abstract_review.cli import main
from abstract_review.models import ResponsesModel
from abstract_review.scifact import run_scifact, scifact_checkpoint
from abstract_review.scoring import semantic_manifest_sha256


def _manifest():
    return {
        "schema_version": 1, "dataset": "scifact-cited-abstracts", "dataset_version": "budget-fixture",
        "split": "dev", "source_sha256": {"corpus": "fixture", "claims": "fixture"},
        "documents": [{"doc_id": 11, "abstract": ["The study reports recorded observations."]}],
        "samples": [{"sample_id": f"scifact:{i}:11", "claim_id": i, "doc_id": 11,
                     "text": f"The effect for cohort {i} was large."} for i in (1, 2)],
    }


def _provider(calls, *, interrupt=False, invalid_reference=False):
    def parse(**request):
        calls.append(request["input"][1]["content"])
        if interrupt:
            raise KeyboardInterrupt
        decision = request["text_format"].model_validate({
            "verdict": "pending", "reason": "Scope was not settled.",
            "evidence_groups": [{"role": "limit", "evidence_ids": ["E99"]}] if invalid_reference else [],
            "missing_fields": [], "follow_up_queries": [],
        })
        usage = None
        if "cohort 1" in json.loads(calls[-1])["claim"]:
            usage = SimpleNamespace(model_dump=lambda **_: {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})
        return SimpleNamespace(id="fixture", status="completed", model="pinned", usage=usage, output_parsed=decision)

    def make_model(model, **options):
        return ResponsesModel(model, **options, client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))

    return make_model


def test_online_scifact_cli_caps_and_resumes_the_whole_batch(tmp_path, monkeypatch, capsys):
    import abstract_review.scifact as adapter

    manifest_path, output = tmp_path / "input.json", tmp_path / "out"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    calls = []
    monkeypatch.setattr(adapter, "run_scifact", lambda *args, **kwargs: run_scifact(
        *args, **kwargs, make_model=_provider(calls),
    ))
    args = ["bench", "run", "--dataset", "scifact", "--engine", "openai", "--model", "pinned",
            "--input-manifest", str(manifest_path), "--output-dir", str(output)]
    assert main([*args, "--max-requests", "1"]) == 0
    partial = json.loads(capsys.readouterr().out)
    first, = json.loads((output / "predictions.json").read_bytes())["predictions"]
    assert partial["prediction_count"] == len(calls) == 1
    assert partial["batch"]["stop_reason"] == "request_budget_exhausted"

    gold_path, score_path = tmp_path / "gold.json", tmp_path / "score.json"
    gold_path.write_text(json.dumps({"schema_version": 1, "labels": [
        {"sample_id": f"scifact:{i}:11", "claim_id": str(i), "verdict": "refuted",
         "accepted_evidence_groups": [["scifact:11:0"]]} for i in (1, 2)
    ]}), encoding="utf-8")
    score = ["bench", "score", "--dataset", "scifact", "--predictions", str(output / "predictions.json"),
             "--gold-manifest", str(gold_path), "--run-record", str(output / "run.json"),
             "--output", str(score_path)]
    assert main(score) == 0
    diagnostics = json.loads(score_path.read_bytes())["run_diagnostics"]
    assert diagnostics["outcome_counts"] == {"unexecuted": 1, "unresolved": 1}
    assert diagnostics["sample_execution_coverage"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    capsys.readouterr()

    assert main([*args, "--max-requests", "1", "--resume"]) == 0
    assert json.loads(capsys.readouterr().out)["batch"]["requests_used"] == len(calls) == 1
    assert main([*args, "--max-requests", "2", "--resume"]) == 0
    finished = json.loads(capsys.readouterr().out)
    record = json.loads((output / "run.json").read_bytes())
    prediction_file = json.loads((output / "predictions.json").read_bytes())
    predictions = prediction_file["predictions"]
    assert record["input_sample_ids"] == ["scifact:1:11", "scifact:2:11"]
    assert record["predictions_sha256"] == semantic_manifest_sha256(prediction_file)
    assert len(predictions) == len(calls) == finished["batch"]["requests_used"] == 2
    assert predictions[0] == first and record["batch"]["status"] == "completed"
    assert [item["cache_hit"] for item in record["samples"]] == [True, False]
    assert [item["tokens"] for item in record["samples"]] == [
        {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
        {"input_tokens": None, "output_tokens": None, "total_tokens": None},
    ]
    assert all("gold" not in json.loads(call) for call in calls)

    # A real sample pause is distinguishable from the next sample, which was never entered.
    monkeypatch.setattr(adapter, "run_scifact", lambda *args, **kwargs: run_scifact(
        *args, **kwargs, make_model=_provider([], invalid_reference=True),
    ))
    pause_dir = tmp_path / "with-pause"
    args[args.index("--output-dir") + 1] = str(pause_dir)
    assert main([*args, "--max-requests", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["batch"]["last_paused_sample"]["sample_id"] == "scifact:1:11"
    score[score.index("--predictions") + 1] = str(pause_dir / "predictions.json")
    score[score.index("--run-record") + 1] = str(pause_dir / "run.json")
    assert main(score) == 0
    diagnostics = json.loads(score_path.read_bytes())["run_diagnostics"]
    assert diagnostics["outcome_counts"] == {"paused": 1, "unexecuted": 1}


def test_interrupted_scifact_response_reserves_a_request_before_transport(tmp_path):
    path = tmp_path / "sample-checkpoints.sqlite3"
    config = {"engine": "openai", "model": "pinned"}
    calls = []
    with pytest.raises(KeyboardInterrupt), scifact_checkpoint(path, "input", config, resume=False, max_requests=1) as batch:
        run_scifact(_manifest(), "openai", "pinned", batch=batch, make_model=_provider(calls, interrupt=True))
    assert len(calls) == 1

    with scifact_checkpoint(path, "input", config, resume=True, max_requests=1) as batch:
        assert run_scifact(_manifest(), "openai", "pinned", batch=batch, make_model=_provider(calls)) == []
        summary = batch.summary(2)
        assert summary["requests_used"] == 1 and summary["elapsed_seconds"] >= 45
        assert summary["requests_without_sample_checkpoint"] == 1
    assert len(calls) == 1

    with scifact_checkpoint(path, "input", config, resume=True, max_requests=3) as batch:
        assert len(run_scifact(_manifest(), "openai", "pinned", batch=batch, make_model=_provider(calls))) == 2
        assert batch.summary(2)["requests_used"] == 3
    assert len(calls) == 3


def test_scifact_checkpoint_refuses_concurrent_writers_and_mismatched_input(tmp_path):
    path = tmp_path / "sample-checkpoints.sqlite3"
    config = {"engine": "openai", "model": "pinned"}
    with scifact_checkpoint(path, "input", config, resume=False):
        with pytest.raises(ValueError, match="already running"), scifact_checkpoint(path, "input", config, resume=True):
            pytest.fail("second writer acquired the checkpoint")
    for digest, options in (("changed", config), ("input", {**config, "model": "changed"})):
        with pytest.raises(ValueError, match="inputs or configuration changed"), scifact_checkpoint(
            path, digest, options, resume=True,
        ):
            pytest.fail("resumed a stale checkpoint")
    with pytest.raises(ValueError, match="use --resume"), scifact_checkpoint(path, "input", config, resume=False):
        pytest.fail("reset the previous budget")


def test_scifact_elapsed_budget_carries_between_invocations(tmp_path, monkeypatch):
    from abstract_review import scifact

    path = tmp_path / "sample-checkpoints.sqlite3"
    config = {"engine": "openai", "model": "pinned"}
    clock = [0.0]
    monkeypatch.setattr(scifact.time, "monotonic", lambda: clock[0])
    calls = []
    with scifact_checkpoint(path, "input", config, resume=False, max_seconds=1) as batch:
        clock[0] = 1
        assert run_scifact(_manifest(), "openai", "pinned", batch=batch, make_model=_provider(calls)) == []
        assert batch.summary(2)["stop_reason"] == "time_budget_exhausted"
    with scifact_checkpoint(path, "input", config, resume=True, max_seconds=1) as batch:
        assert run_scifact(_manifest(), "openai", "pinned", batch=batch, make_model=_provider(calls)) == []
        assert batch.summary(2)["elapsed_seconds"] == 1
    assert not calls
    with scifact_checkpoint(path, "input", config, resume=True, max_seconds=2) as batch:
        assert len(run_scifact(_manifest(), "openai", "pinned", batch=batch, make_model=_provider(calls))) == 2
    assert len(calls) == 2


def test_scifact_budget_applies_to_reference_repair_calls(tmp_path):
    path = tmp_path / "sample-checkpoints.sqlite3"
    config = {"engine": "openai", "model": "pinned"}
    calls = []
    with scifact_checkpoint(path, "input", config, resume=False, max_requests=1) as batch:
        assert run_scifact(_manifest(), "openai", "pinned", batch=batch,
                           make_model=_provider(calls, invalid_reference=True)) == []
        summary = batch.summary(2)
        assert summary["stop_reason"] == "request_budget_exhausted"
        assert summary["processed_samples"] == 0
        last_pause = summary["last_paused_sample"]
        assert last_pause["sample_id"] == "scifact:1:11"
        assert last_pause["model_requests"] == 1
        assert last_pause["tokens"] == {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
        assert [attempt["status"] for attempt in last_pause["decision_trace"]["attempts"]] == [
            "model_reference_error", "model_paused",
        ]
        assert "Scope was not settled." not in json.dumps(last_pause)
    assert len(calls) == 1
    with scifact_checkpoint(path, "input", config, resume=True, max_requests=1) as batch:
        assert run_scifact(_manifest(), "openai", "pinned", batch=batch, make_model=_provider(calls)) == []
        assert batch.summary(2)["last_paused_sample"] == last_pause
    assert len(calls) == 1
    with scifact_checkpoint(path, "input", config, resume=True, max_requests=3) as batch:
        assert len(run_scifact(_manifest(), "openai", "pinned", batch=batch, make_model=_provider(calls))) == 2
        summary = batch.summary(2)
        assert summary["status"] == "completed" and summary["last_paused_sample"] == last_pause
    assert len(calls) == 3
