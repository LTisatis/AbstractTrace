import json
from types import SimpleNamespace

import httpx
from openai import AuthenticationError

from abstract_review import models
from abstract_review.cli import main
from abstract_review.scoring import semantic_manifest_sha256


def _args(tmp_path, output):
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "dataset": "scitabalign-cells", "dataset_version": "synthetic",
        "source_sha256": "a" * 64,
        "samples": [{
            "sample_id": f"scitabalign:{number}", "claim_id": str(number), "paper_id": "paper", "table_id": "table",
            "claim": f"Variant {number} reached 84% accuracy.", "caption": "Held-out results.",
            "grid": [["Variant", "Accuracy"], [str(number), "84%"]],
        } for number in (1, 2)],
    }), encoding="utf-8")
    return ["bench", "run", "--dataset", "scitabalign", "--model", "pinned",
            "--input-manifest", str(manifest), "--output-dir", str(output)]


def _stub_transport(monkeypatch, calls, *, denied=False):
    from abstract_review.models import ResponsesModel

    def parse(**request):
        calls.append(request)
        if denied:
            response = httpx.Response(401, request=httpx.Request("POST", "https://example.invalid/"))
            raise AuthenticationError("test authentication failure", response=response, body=None)
        usage = SimpleNamespace(model_dump=lambda **_: {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})
        selected = request["text_format"].model_validate({"cells": [{"row": 0, "column": 1}, {"row": 1, "column": 1}]})
        return SimpleNamespace(id="test-response", status="completed", model="pinned-release", usage=usage,
                               output_parsed=selected)

    def make_model(*args, **options):
        client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
        return ResponsesModel(*args, client=client, **options)

    monkeypatch.setattr(models, "ResponsesModel", make_model)


def test_scitabalign_cli_online_budget_saves_and_recovers_actual_rows(tmp_path, monkeypatch, capsys):
    calls = []
    _stub_transport(monkeypatch, calls)
    offline = tmp_path / "offline"
    assert main(_args(tmp_path, offline)) == 0
    assert not calls
    assert json.loads((offline / "run.json").read_bytes())["configuration"]["engine"] == "offline"
    capsys.readouterr()

    online = tmp_path / "online"
    args = [*_args(tmp_path, online), "--engine", "openai"]
    limited = [*args, "--max-requests", "1"]
    assert main(limited) == 0
    partial = json.loads(capsys.readouterr().out)
    first, = json.loads((online / "predictions.json").read_bytes())["predictions"]
    assert first["candidate_ids"] == ["scitabalign:1:cell:0:1", "scitabalign:1:cell:1:1"]
    assert partial["prediction_count"] == partial["batch"]["requests_used"] == len(calls) == 1
    assert partial["batch"]["stop_reason"] == "request_budget_exhausted"
    assert main([*limited, "--table-cell-limit", "16", "--resume"]) == 1
    assert "configuration changed" in capsys.readouterr().err and len(calls) == 1

    assert main(limited) == 1
    assert "use --resume" in capsys.readouterr().err
    assert main([*limited, "--resume"]) == 0
    assert json.loads(capsys.readouterr().out)["batch"]["requests_used"] == len(calls) == 1

    assert main([*args, "--max-requests", "2", "--resume"]) == 0
    finished = json.loads(capsys.readouterr().out)
    predictions = json.loads((online / "predictions.json").read_bytes())
    record = json.loads((online / "run.json").read_bytes())
    assert record["predictions_sha256"] == semantic_manifest_sha256(predictions)
    assert record["input_sample_ids"] == ["scitabalign:1", "scitabalign:2"]
    assert record["configuration"]["engine"] == "openai" and record["configuration"]["max_output_tokens"] == 2048
    assert finished["batch"]["status"] == "completed"
    assert finished["prediction_count"] == finished["batch"]["requests_used"] == len(calls) == 2
    assert predictions["predictions"][0] == first
    assert [row["cache_hit"] for row in record["samples"]] == [True, False]
    for sample, request in zip(record["samples"], calls):
        assert sample["stop_reason"] == "model_cell_selection"
        assert sample["model_requests"] == 1 and sample["tokens"]["total_tokens"] == 110
        assert sample["request_usage"][0]["model"] == "pinned-release"
        assert request["max_output_tokens"] == 2048
        assert "gold" not in json.loads(request["input"][1]["content"])


def test_scitabalign_cli_paused_sample_stays_unscored_and_uses_one_request(tmp_path, monkeypatch, capsys):
    calls = []
    _stub_transport(monkeypatch, calls, denied=True)
    online = tmp_path / "online"
    args = [*_args(tmp_path, online), "--engine", "openai", "--max-requests", "2"]
    assert main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["prediction_count"] == 0
    assert len(calls) == summary["batch"]["requests_used"] == 1
    assert summary["batch"]["status"] == "paused"
    assert summary["batch"]["requests_without_sample_checkpoint"] == 1
    paused = summary["batch"]["last_paused_sample"]
    assert paused["sample_id"] == "scitabalign:1" and paused["model_requests"] == 1
    assert paused["request_usage"][0]["status"] == 401
    assert json.loads((online / "predictions.json").read_bytes())["predictions"] == []
