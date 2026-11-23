import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from abstract_review.audit import refresh_compound_findings
from abstract_review.followups import FollowUps
from abstract_review.models import ModelPaused, ModelUnavailable, ResponsesModel, SmokeResult
from abstract_review.reporting import render_markdown
from abstract_review.review import PdfReviewInput, ReviewService
from abstract_review.revisions import build_candidate, new_version
from abstract_review.workflow import _refresh_findings


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
    sentence_check, = candidate["abstract_review"]["sentence_checks"]
    assert sentence_check["status"] == "passed"
    revised_calculation, = sentence_check["calculations"]
    assert (revised_calculation["verification"], revised_calculation["result"]) == ("matched", "8")
    assert revised_calculation["evidence_ids"] == calculation["evidence_ids"]
    assert candidate["abstract_review"]["status"] == "manual_review"
    markdown = render_markdown(candidate)
    assert RIGHT in markdown and "## Revised numerical verification" in markdown

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

    manual = RIGHT.replace("8 percentage points", "12 percentage points")
    amended = resumed_service.submit_selection(run["run_id"], selected["report_version_id"], "selection-3",
                                               edited_text=manual)
    manual_report = resumed_service.export(run["run_id"], amended["report_version_id"])
    assert manual_report["reviewed_abstract"]["text"] == manual
    manual_check, = manual_report["abstract_review"]["sentence_checks"]
    assert manual_check["status"] == "returned"
    assert manual_check["calculations"][0]["verification"] == "mismatch"
    assert resumed_service.export(run["run_id"], selected["report_version_id"]) == confirmed


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


def test_late_model_pause_resumes_without_repeating_completed_sentence_check(tmp_path, monkeypatch):
    pdf = tmp_path / "sample.pdf"
    runpy.run_path(str(SAMPLE))["make_paper"](pdf)
    service = ReviewService(tmp_path / "state.db")
    run = service.start(PdfReviewInput(pdf, pasted_abstract=WRONG, abstract_source="pasted", engine="offline"))
    calls = []

    class CheckingModel:
        def __init__(self, *, pause):
            self.pause = pause
            self.events = []
            self.elapsed_seconds = 0

        def call(self, stage, material, schema):
            calls.append(stage)
            ids = [material["evidence"][0]["evidence_id"]]
            if stage == "verify_sentence":
                return schema(status="passed", reason="The sentence was checked.", evidence_ids=ids)
            if self.pause:
                raise ModelPaused("model budget exhausted")
            return schema(status="passed", reason="The abstract was checked.", evidence_ids=ids, issues=[])

    monkeypatch.setattr(service, "_responses", lambda *args, **kwargs: CheckingModel(pause=True))
    selected = service.submit_selection(run["run_id"], run["report_version_id"], "edited",
                                        edited_text=RIGHT + " The evaluation was fixed.")
    assert selected["run_status"] == "paused"
    assert "verify_sentence" in calls and calls[-1] == "verify_abstract"
    calls.clear()

    restarted = ReviewService(tmp_path / "state.db")
    monkeypatch.setattr(restarted, "_responses", lambda *args, **kwargs: CheckingModel(pause=False))
    result = restarted.resume(run["run_id"], "budget-extension", add_requests=3)
    assert result["run_status"] == "report_ready"
    assert result["report_version_id"] == selected["report_version_id"]
    assert calls == ["verify_abstract"]


def test_verification_supplement_updates_saved_compound_verdict(tmp_path, monkeypatch):
    text = "Accuracy increased on dataset A and on dataset B."
    group = {"compound_id": "group", "operator": "and", "member_claim_ids": ["a", "b"],
             "sentence_id": "s1", "text": text, "start": 0, "end": len(text)}
    claims = [{"claim_id": key, "sentence_id": "s1", "text": key.upper(), "compound": group,
               "proposition": f"Accuracy increased on dataset {key.upper()}.",
               "start": text.index(key.upper(), 1), "end": text.index(key.upper(), 1) + 1} for key in ("a", "b")]
    evidence = [{"evidence_id": "new", "doc_id": "paper", "kind": "body_text",
                 "text": "The updated analysis shows an increase on dataset B.", "locator": {"page_index": 1}}]
    report = {
        "original_abstract": {"version_id": "original"}, "claims": claims, "evidence": evidence,
        "documents": [], "calculations": [], "unresolved_items": [], "revisions": [],
        "findings": [
            {**claim, "review_status": "completed", "verdict": verdict, "evidence_ids": [claim["claim_id"]]}
            for claim, verdict in zip(claims, ("supported", "refuted"), strict=True)
        ],
    }
    followups = FollowUps(report)
    followups.rounds["b"] = {"stage": "verify_sentence", "evidence_ids": ["new"]}
    refresh_compound_findings(report)
    assert report["compound_findings"][0]["verdict"] == "refuted"

    def recheck(claim, index, model, **kwargs):
        assert claim["claim_id"] == "b" and kwargs["additional_evidence"] == evidence
        return {**claim, "review_status": "completed", "verdict": "supported", "evidence_ids": ["new"]}, None

    monkeypatch.setattr("abstract_review.workflow.review_claim", recheck)
    assert _refresh_findings(report, object(), followups, None, tmp_path)
    assert report["compound_findings"][0]["verdict"] == "supported"
    assert "new" in report["compound_findings"][0]["evidence_ids"]


def test_refuted_alternative_does_not_trigger_rewriting_a_supported_disjunction():
    text = "Accuracy increased on dataset A or on dataset B."
    group = {"compound_id": "group", "operator": "or", "member_claim_ids": ["a", "b"],
             "sentence_id": "s1", "text": text, "start": 0, "end": len(text)}
    claims = [{"claim_id": key, "sentence_id": "s1", "text": key.upper(), "compound": group,
               "proposition": f"Accuracy increased on dataset {key.upper()}.",
               "start": text.index(key.upper(), 1), "end": text.index(key.upper(), 1) + 1} for key in ("a", "b")]
    report = {
        "original_abstract": new_version(text, "pasted", "", []), "claims": claims,
        "findings": [
            {**claim, "review_status": "completed", "verdict": verdict, "evidence_ids": [claim["claim_id"]]}
            for claim, verdict in zip(claims, ("supported", "refuted"), strict=True)
        ],
        "evidence": [],
    }

    class UnexpectedModelCall:
        def call(self, *_args):
            pytest.fail("an already supported disjunction does not require an edit")

    refresh_compound_findings(report)
    assert report["compound_findings"][0]["verdict"] == "supported"
    assert build_candidate(report, UnexpectedModelCall()) is None
    assert report["revisions"] == []
