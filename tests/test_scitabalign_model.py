import json
from types import SimpleNamespace

import pytest

from abstract_review.models import ModelPaused, ModelUnavailable, ResponsesModel
from abstract_review.scitabalign import run_scitabalign


def manifest(*ids: str) -> dict:
    return {
        "schema_version": 1,
        "dataset": "scitabalign-cells",
        "dataset_version": "synthetic",
        "source_sha256": "a" * 64,
        "samples": [
            {
                "sample_id": f"scitabalign:{claim_id}",
                "claim_id": claim_id,
                "paper_id": "synthetic-paper",
                "table_id": "synthetic-table",
                "claim": "Variant Blue improved accuracy over Baseline Red.",
                "caption": "Synthetic results on the validation set.",
                "grid": [["Method", "Size", "Accuracy (%)"], ["Variant Blue", "4", "84%"],
                         ["Baseline Red", "8", "72%"]],
            }
            for claim_id in ids
        ],
    }


class StubModel:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.calls = []

    def call(self, stage, material, schema):
        self.calls.append((stage, material))
        if isinstance(self.coordinates, ModelUnavailable):
            raise self.coordinates
        return schema(cells=[{"row": row, "column": column} for row, column in self.coordinates])


def test_model_sees_entire_input_table_and_preserves_chosen_order():
    inputs = manifest("case")
    model = StubModel([(2, 2), (0, 2), (1, 2)])

    prediction, = run_scitabalign(inputs, model=model)

    stage, material = model.calls[0]
    assert stage == "select_table_cells"
    sample = inputs["samples"][0]
    assert material == {"claim": sample["claim"], "caption": sample["caption"], "grid": sample["grid"],
                        "selection_limit": 8}
    assert prediction == {
        "sample_id": "scitabalign:case", "claim_id": "case", "review_status": "pending", "verdict": None,
        "evidence_ids": [], "candidate_ids": ["scitabalign:case:cell:2:2", "scitabalign:case:cell:0:2",
                                               "scitabalign:case:cell:1:2"],
        "stop_reason": "model_cell_selection",
    }


@pytest.mark.parametrize("coordinates", [
    [(-1, 2)],
    [(3, 2)],
    [(1, 3)],
    [(1, 2), (1, 2)],
    [(row, column) for row in range(3) for column in range(3)],
])
def test_invalid_model_cell_selection_scores_as_empty_without_fallback(coordinates):
    prediction, = run_scitabalign(manifest("case"), model=StubModel(coordinates))

    assert prediction["candidate_ids"] == []
    assert prediction["stop_reason"] == "invalid_model_cell_selection"


@pytest.mark.parametrize(("model_result", "stop_reason"), [
    ([], "model_cell_selection_empty"),
    (ModelUnavailable("failed response"), "model_cell_selection_failed"),
])
def test_empty_and_failed_selections_remain_distinguishable(model_result, stop_reason):
    prediction, = run_scitabalign(manifest("case"), model=StubModel(model_result))

    assert prediction["candidate_ids"] == []
    assert prediction["stop_reason"] == stop_reason


def test_late_invalid_input_rejects_run_before_first_model_call():
    inputs = manifest("first", "second")
    inputs["samples"][1]["grid"][2][1] = 8
    model = StubModel([])

    with pytest.raises(ValueError, match="only text"):
        run_scitabalign(inputs, model=model)
    assert model.calls == []


def test_responses_model_passes_output_limit_and_counts_actual_requests():
    model_calls = []

    def parse(**request):
        model_calls.append(request)
        answer = request["text_format"](cells=[{"row": 1, "column": 2}])
        return SimpleNamespace(id="response-1", status="completed", model="test-model", usage=None,
                               output_parsed=answer)

    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    model = ResponsesModel("test-model", max_requests=1, max_output_tokens=1024, client=client)

    prediction, = run_scitabalign(manifest("case"), model=model)
    request, = model_calls
    assert request["max_output_tokens"] == 1024
    assert request["store"] is False
    assert "selection_limit" in json.loads(request["input"][1]["content"])
    assert prediction["candidate_ids"] == ["scitabalign:case:cell:1:2"]
    assert model.events[0]["stage"] == "select_table_cells"

    with pytest.raises(ModelPaused, match="model budget exhausted"):
        run_scitabalign(manifest("next"), model=model)
    assert len(model_calls) == 1
