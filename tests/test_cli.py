import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from abstract_review.scoring import semantic_manifest_sha256


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "d01"


@pytest.fixture
def cli(tmp_path):
    workspace = tmp_path / "离线 审稿"
    workspace.mkdir()
    hooks = tmp_path / "network_hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text(
        "import sys\n"
        "def deny_network(event, args):\n"
        "    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:\n"
        "        raise RuntimeError(f'Offline CLI tried to access the network: {event}')\n"
        "sys.addaudithook(deny_network)\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(hooks), env.get("PYTHONPATH", "")) if path
    )
    env["ABSTRACT_REVIEW_DATA_DIR"] = str(workspace / "审稿 数据")

    inputs = workspace / "审稿 输入.json"
    shutil.copyfile(EXAMPLE / "inputs.json", inputs)

    def invoke(*args, success=True):
        result = subprocess.run(
            [sys.executable, "-m", "abstract_review", *map(str, args)],
            cwd=workspace,
            env=env,
            capture_output=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
        if success:
            assert result.returncode == 0, result.stderr
        else:
            assert result.returncode != 0
            assert "Traceback" not in result.stderr
        return result

    return invoke, workspace, inputs


def test_cli_scores_independent_conjunct_diagnostic(cli):
    run, workspace, _ = cli
    claim = {"sample_id": "s", "claim_id": "whole", "review_status": "completed", "verdict": "refuted",
             "claim_anchor": {"sentence_id": "s1", "start": 0, "end": 37, "proposition": "We screened and accepted all records."},
             "evidence_ids": ["p1"], "candidate_ids": ["p1"]}
    operand = {"sentence_id": "s1", "start": 16, "end": 36, "proposition": "We accepted all records."}
    predictions = {"schema_version": 1, "predictions": [claim]}
    gold = {"schema_version": 1, "extraction_samples": ["s"], "labels": [
        {"sample_id": "s", "claim_id": "accepted", "verdict": "refuted",
         "accepted_evidence_groups": [["p1"]], "claim_anchor": operand},
    ]}
    projections = {
        "schema_version": 1, "annotation_id": "local-review", "annotator_type": "human",
        "predictions_sha256": semantic_manifest_sha256(predictions), "gold_sha256": semantic_manifest_sha256(gold),
        "projections": [{"sample_id": "s", "predicted_claim_id": "whole", "gold_claim_id": "accepted",
                         "relation": "conjunct", "operand_anchor": operand, "evidence_ids": ["p1"],
                         "review_note": "The cited records show that screening excluded some records."}],
    }
    for name, document in (("predictions", predictions), ("gold", gold), ("projections", projections)):
        (workspace / f"{name}.json").write_text(json.dumps(document), encoding="utf-8")
    args = ("bench", "score", "--predictions", "predictions.json", "--gold-manifest", "gold.json",
            "--verdict-projections", "projections.json")
    result = json.loads(run(*args, "--output", "scores.json").stdout)
    assert result["joint_accuracy"]["numerator"] == 0
    assert result["verdict_projection"]["credited_count"] == 1
    assert result == json.loads((workspace / "scores.json").read_text(encoding="utf-8"))
    rejected = run(*args, "--output", "projections.json", success=False)
    assert "score output must differ from scoring input files" in rejected.stderr


def test_cli_reviews_and_exports_after_restart(cli):
    run, workspace, inputs = cli
    diagnosis = json.loads(run("doctor").stdout)
    assert diagnosis["status"] == "ok"
    assert diagnosis["mode"] == "synthetic"
    assert diagnosis["fts5"] is True

    started = json.loads(
        run("review", "start", "--fixture", inputs, "--sample-id", "synthetic-001").stdout
    )
    assert started["run_status"] == "report_ready"
    assert json.loads(run("review", "status", started["run_id"]).stdout) == started

    export_args = (
        "review",
        "export",
        started["run_id"],
        "--version-id",
        started["report_version_id"],
        "--format",
        "json",
    )
    report = json.loads(run(*export_args).stdout)
    assert report["mode"] == "synthetic"
    assert report["sample_id"] == "synthetic-001"
    assert report["run_id"] == started["run_id"]
    assert report["report_version_id"] == report["reviewed_abstract"]["version_id"]
    assert report["report_version_id"] == started["report_version_id"]
    source_abstract = json.loads(inputs.read_text(encoding="utf-8"))["samples"][0]["abstract"]
    assert report["original_abstract"]["text"] == source_abstract
    assert report["reviewed_abstract"]["text"] == source_abstract
    assert {
        finding["claim_id"]: (finding["review_status"], finding["verdict"], finding["evidence_ids"])
        for finding in report["findings"]
    } == {
        "c-a": ("completed", "supported", ["e-a"]),
        "c-b": ("completed", "refuted", ["e-b"]),
        "c-c": ("completed", "insufficient", []),
    }

    report_path = workspace / "审稿 报告.json"
    run(*export_args, "--output", report_path)
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_cli_runs_predictions_before_gold_is_available(cli):
    run, workspace, inputs = cli
    output_dir = workspace / "批跑 结果"
    gold = workspace / "评分 金标.json"
    assert not gold.exists()
    run("bench", "run", "--input-manifest", inputs, "--output-dir", output_dir)
    predictions = output_dir / "predictions.json"
    prediction_rows = json.loads(predictions.read_text(encoding="utf-8"))["predictions"]
    assert len(prediction_rows) == 3
    assert {row["claim_id"]: row["verdict"] for row in prediction_rows} == {
        "c-a": "supported",
        "c-b": "refuted",
        "c-c": "insufficient",
    }

    shutil.copyfile(EXAMPLE / "gold.json", gold)
    score_path = workspace / "评分 结果.json"
    run(
        "bench", "score", "--predictions", predictions,
        "--gold-manifest", gold, "--output", score_path,
    )
    result = json.loads(score_path.read_text(encoding="utf-8"))
    assert result["accuracy"] == result["completed_coverage"] == {
        "numerator": 3,
        "denominator": 3,
        "value": 1.0,
    }
    assert result["evidence_group_recall"] == result["joint_accuracy"] == {
        "numerator": 2,
        "denominator": 2,
        "value": 1.0,
    }


def test_cli_rejects_gold_fields_in_inference_manifest(cli):
    run, workspace, inputs = cli
    contaminant = json.loads(inputs.read_text(encoding="utf-8"))
    contaminant["samples"][0]["claims"][0]["gold_verdict"] = "refuted"
    inputs.write_text(json.dumps(contaminant, ensure_ascii=False), encoding="utf-8")

    output_dir = workspace / "批跑 结果"
    failure = run(
        "bench", "run", "--input-manifest", inputs, "--output-dir", output_dir, success=False,
    )
    assert failure.stderr
    assert not (output_dir / "predictions.json").exists()


def test_cli_unknown_run_and_version_fail_cleanly(cli):
    run, _, inputs = cli
    assert run("review", "status", "missing-run", success=False).stderr

    started = json.loads(
        run("review", "start", "--fixture", inputs, "--sample-id", "synthetic-001").stdout
    )
    failure = run(
        "review", "export", started["run_id"], "--version-id", "missing-version",
        "--format", "json", success=False,
    )
    assert failure.stderr
