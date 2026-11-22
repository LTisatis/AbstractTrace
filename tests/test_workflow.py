import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from abstract_review.models import ModelUnavailable, ResponsesModel, SmokeResult
from abstract_review.reporting import render_markdown
from abstract_review.review import PdfReviewInput, ReviewService


SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "d02" / "make_sample.py"
WRONG = "On dataset A, accuracy increased by 10 percentage points."
RIGHT = "On dataset A, accuracy increased by 8 percentage points."


def test_numeric_refutation_suggested_edit_and_selected_version(tmp_path):
    pdf = tmp_path / "paper.pdf"
    runpy.run_path(str(SAMPLE))["make_paper"](pdf)
    state = tmp_path / "runs" / "state.sqlite3"
    service = ReviewService(state)
    run = service.start(PdfReviewInput(pdf, pasted_abstract=WRONG, abstract_source="pasted", engine="offline"))
    candidate = service.export(run["run_id"], run["report_version_id"])
    assert run["run_status"] == "report_ready"
    assert candidate["findings"][0]["verdict"] == "refuted"
    calculation = candidate["calculations"][0]
    assert (calculation["result"], calculation["unit"], calculation["operands"]) == (
        "8", "percentage_points", ["80%", "88%"],
    )
    evidence = {item["evidence_id"]: item for item in candidate["evidence"]}
    assert all(evidence[eid]["locator"]["page_index"] == 1 for eid in calculation["evidence_ids"])
    assert candidate["reviewed_abstract"]["text"] == RIGHT
    assert candidate["abstract_review"]["sentence_checks"][0]["status"] == "passed"
    assert candidate["abstract_review"]["status"] == "manual_review"
    assert RIGHT in render_markdown(candidate)

    resumed_service = ReviewService(state)
    revision_ids = tuple(candidate["reviewed_abstract"]["revision_ids"])
    selected = resumed_service.submit_selection(run["run_id"], run["report_version_id"], "selection-1", revision_ids)
    assert selected == resumed_service.submit_selection(run["run_id"], run["report_version_id"], "selection-1", revision_ids)
    assert selected["report_version_id"] != run["report_version_id"]
    confirmed = resumed_service.export(run["run_id"], selected["report_version_id"])
    assert confirmed["reviewed_abstract"]["text"] == RIGHT
    assert confirmed["abstract_review"]["version_id"] == selected["report_version_id"]
    assert resumed_service.export(run["run_id"], run["report_version_id"]) == candidate
    with pytest.raises(ValueError, match="stale base version"):
        resumed_service.submit_selection(run["run_id"], run["report_version_id"], "selection-2", (), WRONG)


def test_unconfigured_online_run_keeps_report_and_idempotent_budget_extension(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    source = tmp_path / "paper.pdf"
    runpy.run_path(str(SAMPLE))["make_paper"](source)
    service = ReviewService(tmp_path / "state.db")
    result = service.start(PdfReviewInput(source, engine="openai"))
    assert result["run_status"] == "paused"
    for _ in range(2):
        result = service.resume(result["run_id"], "extension-1", add_requests=3, add_seconds=15)
    report = service.export(result["run_id"], result["report_version_id"])
    assert report["configuration"]["max_requests"] == 83
    assert report["configuration"]["max_seconds"] == 915
    assert report["usage"]["external_requests"] == 0
    assert "OPENAI_API_KEY" in report["runtime_error"]
    with service.store.acquire(result["run_id"]), pytest.raises(ValueError, match="already being processed"):
        service.resume(result["run_id"])


def test_model_requests_use_fresh_context_and_reserve_budget():
    calls = []
    reserved = []

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id="r1", status="completed", model="pinned", usage=None,
                               output_parsed=SmokeResult(phrase="ready"))

    model = ResponsesModel(client=SimpleNamespace(responses=SimpleNamespace(parse=parse)), max_requests=2,
                           before_call=lambda: reserved.append(model.events[-1]["status"]))
    assert model.call("smoke", {"phrase": "ready"}, SmokeResult).phrase == "ready"
    assert model.call("smoke", {"phrase": "ready"}, SmokeResult).phrase == "ready"
    with pytest.raises(ModelUnavailable, match="budget"):
        model.call("smoke", {}, SmokeResult)
    assert reserved == ["outcome_unknown", "outcome_unknown"]
    assert all(call["store"] is False and "previous_response_id" not in call for call in calls)
    assert all([item["role"] for item in call["input"]] == ["developer", "user"] for call in calls)
