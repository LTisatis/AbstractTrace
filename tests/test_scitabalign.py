import json
from copy import deepcopy

import pytest

from abstract_review.scitabalign import prepare_scitabalign, run_scitabalign, score_scitabalign


def annotation(claim_id, cells):
    return {
        "id": claim_id, "paper_id": "synthetic-paper", "table_id": "synthetic-table",
        "claim": "Variant Blue reached 84% accuracy.", "table_caption": "Synthetic evaluation results.",
        "table_column_names": ["Method", "Size", "Accuracy (%)"],
        "table_content_values": [["Variant Blue", 4, "84%"], ["Baseline Red", 8, "72%"]],
        "label": "Supported", "human_label": "Good", "explanation_cells": cells,
        "explanation_cot_cells": "Reserved scoring annotation",
    }


def prepare(tmp_path, rows):
    path = tmp_path / "release.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return prepare_scitabalign(path, dataset_version="synthetic-release")


def prediction(claim_id, coordinates):
    sample_id = f"scitabalign:{claim_id}"
    return {"sample_id": sample_id, "claim_id": claim_id, "review_status": "pending", "verdict": None,
            "evidence_ids": [], "candidate_ids": [f"{sample_id}:cell:{row}:{col}" for row, col in coordinates]}


def test_release_coordinate_origin_and_input_isolation(tmp_path):
    inputs, gold = prepare(tmp_path, [annotation("claim-1", [[1, 3], [2, 3]])])

    sample, = inputs["samples"]
    assert set(sample) == {"sample_id", "claim_id", "paper_id", "table_id", "claim", "caption", "grid"}
    assert sample["grid"][1] == ["Variant Blue", "4", "84%"]
    assert "Reserved scoring annotation" not in json.dumps(inputs)
    label, = gold["labels"]
    assert label["alignment_status"] == "aligned"
    assert label["cells"] == [[0, 2], [1, 2]]
    assert label["shape"] == [3, 3]
    assert inputs["source_sha256"] == gold["source_sha256"]

    first, = run_scitabalign(inputs)
    assert first["verdict"] is None and first["evidence_ids"] == []
    assert first["candidate_ids"][0] == "scitabalign:claim-1:cell:1:2"
    assert run_scitabalign(inputs) == [first]

    altered = deepcopy(inputs)
    altered["samples"][0]["explanation_cells"] = [[0, 2], [1, 2]]
    with pytest.raises(ValueError, match="required fields"):
        run_scitabalign(altered)


def test_score_uses_macro_cutoffs_and_preserves_excluded_case_counts(tmp_path):
    inputs, gold = prepare(tmp_path, [
        annotation("first", [[1, 3], [2, 3]]), annotation("missing", [[2, 3]]),
        annotation("empty", []), annotation("malformed", [[2, 3], [0, 1]]),
    ])
    assert len(run_scitabalign(inputs)) == 4
    predictions = {"schema_version": 1, "predictions": [prediction("first", [(1, 2), (1, 1), (0, 2)])]}

    result = score_scitabalign(predictions, gold)

    assert result["task"] == "given_table_cell_ranking"
    assert result["alignment"] == {"aligned": 2, "empty_explanation": 1, "invalid_coordinates": 1}
    assert result["alignment_coverage"] == {"numerator": 2, "denominator": 4, "value": 0.5}
    assert result["prediction_coverage"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert result["cell_retrieval_at_k"]["3"] == pytest.approx({
        "macro_precision": 1 / 3, "macro_recall": 1 / 2, "macro_f1": 0.4, "macro_complete_rationale": 0.5,
    })
    assert result["cases"][1]["scores_at_k"]["3"]["f1"] == 0
    assert result["cases"][2]["scores_at_k"] is None
    assert result["cases"][3]["gold_cells"] == []


def test_capacity_diagnostics_separate_oversize_groups_and_missing_predictions(tmp_path):
    grid = [(row, col) for row in range(3) for col in range(3)]
    _, gold = prepare(tmp_path, [
        annotation("oversize", [[row + 1, col + 1] for row, col in grid]),
        annotation("complete", [[1, 1], [1, 2]]),
        annotation("missing", [[1, 1], [1, 2]]),
        annotation("empty", []),
    ])
    predictions = {"schema_version": 1, "predictions": [
        prediction("oversize", grid[:8]), prediction("complete", [(0, 0), (0, 1)]),
    ]}

    result = score_scitabalign(predictions, gold)
    by_capacity = result["rationale_capacity_at_k"]
    assert by_capacity["1"]["eligible_complete_rationale_recall"] == {
        "numerator": 0, "denominator": 0, "value": None,
    }
    assert by_capacity["1"]["eligible_macro_recall"] is None
    assert by_capacity["8"]["complete_rationale_ceiling"] == {
        "numerator": 2, "denominator": 3, "value": 2 / 3,
    }
    assert by_capacity["8"]["eligible_complete_rationale_recall"] == {
        "numerator": 1, "denominator": 2, "value": 0.5,
    }
    assert by_capacity["8"]["eligible_macro_recall"] == 0.5
    assert by_capacity["8"]["over_capacity_macro_recall"] == 8 / 9
    assert by_capacity["8"]["macro_recall_ceiling"] == pytest.approx((2 + 8 / 9) / 3)
    assert result["cell_retrieval_at_k"]["8"]["macro_complete_rationale"] == 1 / 3


@pytest.mark.parametrize("cell_id", ["scitabalign:other:cell:1:2", "scitabalign:first:cell:1:3"])
def test_scoring_rejects_foreign_or_out_of_range_candidate(tmp_path, cell_id):
    _, gold = prepare(tmp_path, [annotation("first", [[1, 3]])])
    row = prediction("first", [])
    row["candidate_ids"] = [cell_id]
    with pytest.raises(ValueError, match="outside the given table"):
        score_scitabalign({"schema_version": 1, "predictions": [row]}, gold)


def test_release_rejects_duplicate_claims_and_ragged_rows(tmp_path):
    row = annotation("first", [[1, 3]])
    with pytest.raises(ValueError, match="duplicate SciTabAlign claim"):
        prepare(tmp_path, [row, row])

    row["table_content_values"][0].pop()
    with pytest.raises(ValueError, match="match the header width"):
        prepare(tmp_path, [row])
