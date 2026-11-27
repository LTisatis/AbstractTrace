import os
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError

from abstract_review.models import Extraction, ModelPaused, ResponsesModel, SmokeResult, online_diagnostics


def test_online_diagnostics_exercises_text_and_vision_requests():
    requests = []

    def parse(**payload):
        requests.append(payload)
        phrase = "AbstractTrace ready" if len(requests) == 1 else "W7D42"
        return SimpleNamespace(id="response", status="completed", model="test-model", usage=None,
                               output_parsed=SmokeResult(phrase=phrase))

    model = ResponsesModel("test-model", client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    checked = online_diagnostics("test-model", model=model)

    assert checked["model"] == "test-model"
    assert [event["stage"] for event in checked["attempts"]] == ["smoke", "smoke_image"]
    content = requests[1]["input"][1]["content"]
    assert content[-1]["image_url"].startswith("data:image/png;base64,")
    assert all("W7D42" not in part.get("text", "") for part in content)
    assert all(request["store"] is False for request in requests)


@pytest.mark.parametrize(("budget", "expected"), [(900, 120), (60, 60)])
def test_extraction_timeout_respects_remaining_workflow_budget(monkeypatch, budget, expected):
    monkeypatch.setattr("abstract_review.models.time", SimpleNamespace(monotonic=lambda: 0, sleep=lambda _seconds: None))
    timeouts = []

    def parse(**payload):
        timeouts.append(payload["timeout"])
        raise APITimeoutError(request=httpx.Request("POST", "https://example.invalid/responses"))

    model = ResponsesModel("test-model", max_seconds=budget,
                           client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    with pytest.raises(ModelPaused, match="model request timed out"):
        model.call("extract", {"sentences": []}, Extraction)
    assert timeouts == [expected] * 3
    assert [event["status"] for event in model.events] == ["timeout"] * 3


@pytest.mark.live
def test_live_text_and_vision_diagnostics(request):
    if not request.config.getoption("--run-live"):
        pytest.skip("requires --run-live")
    name = os.environ.get("ABSTRACT_REVIEW_LIVE_MODEL", "gpt-6.2-sol")
    assert online_diagnostics(name)["vision_output"] == "W7D42"
