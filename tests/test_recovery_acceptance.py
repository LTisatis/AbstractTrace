import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from abstract_review import models
from abstract_review.review import PdfReviewInput, ReviewService


PAPER_FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "d02" / "make_sample.py"
CLAIM = "On dataset A, accuracy increased by 8 percentage points."


class _LostCheckpoint(RuntimeError):
    pass


def test_response_lost_before_checkpoint_consumes_budget_and_recovers_once(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    runpy.run_path(str(PAPER_FIXTURE))["make_paper"](pdf)
    database = tmp_path / "state.sqlite3"
    service = ReviewService(database)
    run = service.start(PdfReviewInput(pdf, provided_claims=(("claim-a", CLAIM),), engine="openai"), execute=False)
    run_id, version_id = run["run_id"], run["report_version_id"]
    seeded = service.export(run_id, version_id)
    seeded["configuration"]["max_requests"] = 2
    service.store.save(seeded)

    calls = []

    def parse(**request):
        schema = request["text_format"]
        context = json.loads(request["input"][1]["content"])
        assert [message["role"] for message in request["input"]] == ["developer", "user"]
        assert request["store"] is False and {"tools", "previous_response_id"}.isdisjoint(request)
        if "claim" in context:
            assert context["claim"] == CLAIM
            ids = context["calculation"]["evidence_ids"]
            parsed = schema(
                verdict="supported", reason="The aligned cells give eight percentage points.",
                evidence_groups=[{"role": "support", "evidence_ids": ids}],
                missing_fields=[], follow_up_queries=[], coverage=[
                    {"requirement": field, "status": "found", "evidence_ids": ids}
                    for field in context["requirements"]
                ],
            )
            calls.append("judge")
        else:
            assert context["version_id"] == version_id
            parsed = schema(status="passed", reason="The source abstract is unchanged.", evidence_ids=[], issues=[])
            calls.append("verify_abstract")
        usage = SimpleNamespace(model_dump=lambda **_kwargs: {"input_tokens": 5, "output_tokens": 2})
        return SimpleNamespace(id=f"fake-response-{len(calls)}", status="completed", model=request["model"],
                               usage=usage, output_parsed=parsed)

    responses_model = models.ResponsesModel
    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(models, "ResponsesModel", lambda *args, **kwargs: responses_model(*args, client=client, **kwargs))
    save = service.store.save

    def lose_completed_checkpoint(report, *args, **kwargs):
        events = report["usage"].get("model_events", ())
        if events and events[-1]["status"] == "completed":
            assert events[-1]["response_id"] == "fake-response-1"
            assert events[-1]["input_tokens"] == 5
            raise _LostCheckpoint("process lost the uncommitted response")
        return save(report, *args, **kwargs)

    monkeypatch.setattr(service.store, "save", lose_completed_checkpoint)
    with pytest.raises(_LostCheckpoint, match="uncommitted response"):
        service.resume(run_id)
    assert calls == ["judge"]

    restarted = ReviewService(database)
    recovered = restarted.export(run_id, version_id)
    assert recovered["findings"] == []
    assert recovered["usage"]["external_requests"] == 1
    unknown, = recovered["usage"]["model_events"]
    assert (unknown["stage"], unknown["status"]) == ("judge", "outcome_unknown")
    assert {"response_id", "input_tokens", "output_tokens"}.isdisjoint(unknown)

    assert restarted.resume(run_id)["run_status"] == "paused"
    paused = restarted.export(run_id, version_id)
    assert paused["runtime_error"] == "model budget exhausted"
    assert paused["usage"]["external_requests"] == 2
    assert calls == ["judge", "judge"]
    assert [(row["claim_id"], row["verdict"]) for row in paused["findings"]] == [("claim-a", "supported")]

    assert restarted.resume(run_id, "extend-once", add_requests=1)["run_status"] == "report_ready"
    assert restarted.resume(run_id, "extend-once", add_requests=1)["run_status"] == "report_ready"
    final = restarted.export(run_id, version_id)
    assert final["configuration"]["max_requests"] == 3
    assert final["budget_extensions"] == {"extend-once": [1, 0]}
    assert final["findings"] == paused["findings"]
    assert final["abstract_review"]["status"] == "passed"
    assert calls == ["judge", "judge", "verify_abstract"]
    assert final["usage"]["external_requests"] == 3
    first, *known = final["usage"]["model_events"]
    assert first == unknown
    assert [(event["response_id"], event["input_tokens"], event["output_tokens"]) for event in known] == [
        ("fake-response-2", 5, 2), ("fake-response-3", 5, 2),
    ]
