import json
from types import SimpleNamespace

import pytest

from abstract_review.models import ModelPaused, ModelUnavailable, ResponsesModel
from abstract_review.scitabalign import run_scitabalign, score_scitabalign


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
                        "selection_limit": 24}
    assert prediction == {
        "sample_id": "scitabalign:case", "claim_id": "case", "review_status": "pending", "verdict": None,
        "evidence_ids": [], "candidate_ids": ["scitabalign:case:cell:2:2", "scitabalign:case:cell:0:2",
                                               "scitabalign:case:cell:1:2"],
        "stop_reason": "model_cell_selection",
    }


def test_wider_comparison_scores_the_complete_group():
    inputs = manifest("case")
    sample, = inputs["samples"]
    sample["claim"] = "Compare the accuracy and recall of Atlas, Beacon and Cedar."
    sample["grid"] = [["Method", "Accuracy", "Recall"],
                      ["Atlas", "82%", "79%"], ["Beacon", "81%", "74%"], ["Cedar", "77%", "70%"]]
    coordinates = [(row, col) for row in range(4) for col in range(3) if (row, col) != (0, 0)]
    model = StubModel(coordinates)

    predictions = run_scitabalign(inputs, model=model)

    assert model.calls[0][1]["selection_limit"] == 24
    assert len(predictions[0]["candidate_ids"]) == len(coordinates)
    gold = {key: inputs[key] for key in ("schema_version", "dataset", "dataset_version", "source_sha256")}
    gold["labels"] = [{"sample_id": sample["sample_id"], "claim_id": sample["claim_id"], "shape": [4, 3],
                       "alignment_status": "aligned", "cells": [list(cell) for cell in coordinates]}]
    scored = score_scitabalign({"schema_version": 1, "predictions": predictions}, gold)
    assert scored["cell_retrieval_at_k"]["8"]["macro_complete_rationale"] == 0
    assert scored["cell_retrieval_at_k"]["16"]["macro_complete_rationale"] == 1
    limited, = run_scitabalign(inputs, model=model, cell_limit=8)
    assert limited["stop_reason"] == "invalid_model_cell_selection"


def test_small_table_uses_eight_cell_ceiling():
    inputs = manifest("case")
    inputs["samples"][0]["grid"] = inputs["samples"][0]["grid"][:2]
    model = StubModel([(1, 2)])

    run_scitabalign(inputs, model=model)

    assert model.calls[0][1]["selection_limit"] == 8


def test_all_metrics_comparison_can_include_more_than_sixteen_required_cells():
    inputs = manifest("case")
    sample, = inputs["samples"]
    sample["claim"] = "Atlas and Beacon are comparable in all aspects."
    sample["grid"] = [["Method", *(f"Metric {column}" for column in range(1, 7))],
                      ["Atlas", *["83"] * 6], ["Beacon", *["82"] * 6]]
    coordinates = [(row, col) for row in range(3) for col in range(7) if (row, col) != (0, 0)]
    model = StubModel(coordinates)

    predictions = run_scitabalign(inputs, model=model)

    assert model.calls[0][1]["selection_limit"] == 24
    gold = {key: inputs[key] for key in ("schema_version", "dataset", "dataset_version", "source_sha256")}
    gold["labels"] = [{"sample_id": sample["sample_id"], "claim_id": sample["claim_id"], "shape": [3, 7],
                       "alignment_status": "aligned", "cells": [list(cell) for cell in coordinates]}]
    scored = score_scitabalign({"schema_version": 1, "predictions": predictions}, gold)["cell_retrieval_at_k"]
    assert scored["16"]["macro_complete_rationale"] == 0
    assert scored["24"]["macro_complete_rationale"] == 1


def test_wide_model_selection_reranks_with_table_context_without_changing_membership():
    inputs = manifest("case")
    sample, = inputs["samples"]
    sample["grid"] += [["Other Cyan", "9", "64%"], ["Other Green", "10", "59%"]]
    sample["claim"] = "Variant Blue reaches 84% accuracy and Baseline Red reaches 72% accuracy."
    chosen = [(4, 1), (3, 1), (4, 2), (3, 2), (4, 0), (3, 0), (1, 1), (2, 1),
              (0, 2), (1, 2), (2, 2), (1, 0), (2, 0)]
    original_ids = {f"scitabalign:case:cell:{row}:{col}" for row, col in chosen}

    prediction, = run_scitabalign(inputs, model=StubModel(chosen))

    assert set(prediction["candidate_ids"]) == original_ids
    assert prediction["candidate_ids"][:3] == ["scitabalign:case:cell:1:2", "scitabalign:case:cell:1:0",
                                               "scitabalign:case:cell:0:2"]


def test_explicit_limit_can_return_and_score_more_than_twenty_four_cells():
    inputs = manifest("case")
    sample, = inputs["samples"]
    sample["grid"] += [[f"Variant {row}", "4", "82%"] for row in range(3, 12)]
    chosen = [(row, col) for row in range(12) for col in (0, 2) if row] + [(0, 2), (0, 1), (1, 1)]
    assert len(chosen) == 25
    model = StubModel(chosen)

    prediction, = run_scitabalign(inputs, model=model, cell_limit=48)

    assert model.calls[0][1]["selection_limit"] == 48
    assert len(prediction["candidate_ids"]) == 25
    metadata = {key: value for key, value in inputs.items() if key != "samples"}
    gold = {**metadata, "labels": [{"sample_id": sample["sample_id"], "claim_id": sample["claim_id"], "shape": [12, 3],
                                  "alignment_status": "aligned", "cells": [list(cell) for cell in chosen]}]}
    scored = score_scitabalign({"schema_version": 1, "predictions": [prediction]}, gold)["cell_retrieval_at_k"]
    assert scored["24"]["macro_complete_rationale"] == 0
    assert scored["48"]["macro_complete_rationale"] == 1


@pytest.mark.parametrize("coordinates", [
    [(-1, 2)],
    [(3, 2)],
    [(1, 3)],
    [(1, 2), (1, 2)],
    [(row, column) for row in range(3) for column in range(3)],
])
def test_invalid_model_cell_selection_scores_as_empty_without_fallback(coordinates):
    prediction, = run_scitabalign(manifest("case"), model=StubModel(coordinates), cell_limit=8)

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
