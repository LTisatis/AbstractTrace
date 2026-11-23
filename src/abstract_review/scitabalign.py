"""Given-table cell ranking over a local SciTabAlign release.

Source: https://github.com/Alab-NII/SciTabAlign. The released explanation
coordinates use one-based rows and columns, including the column-name row.
The authors' runner adjusts only rows. We normalize both axes to the actual
table grid and report retrieval at fixed cutoffs separately from the paper's
free-form evidence selection task.
"""

import hashlib
import json
import math
import re
from collections import Counter
from contextlib import closing
from pathlib import Path

from .evidence import EvidenceIndex
from .scoring import _DEFAULT_RANK_CUTOFFS, _identifier, _parse_predictions, _ratio, _rows


_DATASET = "scitabalign-cells"
_STATUSES = ("aligned", "empty_explanation", "invalid_coordinates")
_COORDINATE = re.compile(r"(0|[1-9][0-9]*):(0|[1-9][0-9]*)")


def _sample_id(claim_id: str) -> str:
    return f"scitabalign:{claim_id}"


def _cell_id(sample_id: str, row: int, column: int) -> str:
    return f"{sample_id}:cell:{row}:{column}"


def _shape(grid: object) -> tuple[int, int]:
    if not isinstance(grid, list) or len(grid) < 2 or not isinstance(grid[0], list) or not grid[0]:
        raise ValueError("SciTabAlign requires a header and at least one table row")
    width = len(grid[0])
    if any(not isinstance(row, list) or len(row) != width for row in grid):
        raise ValueError("SciTabAlign table rows must match the header width")
    return len(grid), width


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"SciTabAlign {name} must be nonempty text")
    return value


def _release_grid(row: dict) -> list[list[str]]:
    headers, values = row.get("table_column_names"), row.get("table_content_values")
    if not isinstance(headers, list) or not all(isinstance(cell, str) for cell in headers):
        raise ValueError("SciTabAlign table_column_names must list text headers")
    if not isinstance(values, list):
        raise ValueError("SciTabAlign table_content_values must list rows")
    _shape([headers, *values])
    if any(type(cell) not in {str, int, float} or (type(cell) is float and not math.isfinite(cell))
           for cells in values for cell in cells):
        raise ValueError("SciTabAlign table cells must be text or finite numbers")
    return [list(headers), *[[str(cell) for cell in cells] for cells in values]]


def _within(cell: object, shape: tuple[int, int], *, first: int) -> bool:
    return isinstance(cell, list) and len(cell) == 2 and all(
        type(index) is int and first <= index < size + first for index, size in zip(cell, shape)
    )


def _rationale(cells: object, shape: tuple[int, int]) -> tuple[str, list[list[int]]]:
    if cells == []:
        return "empty_explanation", []
    if not isinstance(cells, list) or any(not _within(cell, shape, first=1) for cell in cells):
        return "invalid_coordinates", []
    return "aligned", [list(cell) for cell in sorted({(row - 1, col - 1) for row, col in cells})]


def prepare_scitabalign(annotations: Path, *, dataset_version: str) -> tuple[dict, dict]:
    """Create separate retrieval inputs and gold from the author's JSON array."""
    version = _identifier(dataset_version, "SciTabAlign dataset_version")
    raw = annotations.read_bytes()
    try:
        source = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{annotations}: invalid UTF-8 SciTabAlign JSON") from exc
    if not isinstance(source, list) or not source:
        raise ValueError("expected a nonempty SciTabAlign release array")
    seen, samples, labels = set(), [], []
    for index, row in enumerate(source):
        if not isinstance(row, dict):
            raise ValueError(f"SciTabAlign source row {index + 1} must be an object")
        claim_id = _identifier(row.get("id"), f"SciTabAlign source row {index + 1} id")
        if claim_id in seen:
            raise ValueError(f"duplicate SciTabAlign claim ID {claim_id!r}")
        seen.add(claim_id)
        grid = _release_grid(row)
        shape = _shape(grid)
        key = {"sample_id": _sample_id(claim_id), "claim_id": claim_id}
        samples.append({
            **key, "paper_id": _identifier(row.get("paper_id"), "SciTabAlign paper_id"),
            "table_id": _identifier(row.get("table_id"), "SciTabAlign table_id"),
            "claim": _text(row.get("claim"), "claim"),
            "caption": _text(row.get("table_caption"), "caption"), "grid": grid,
        })
        status, cells = _rationale(row.get("explanation_cells"), shape)
        labels.append({**key, "shape": list(shape), "alignment_status": status, "cells": cells})
    metadata = {"schema_version": 1, "dataset": _DATASET, "dataset_version": version,
                "source_sha256": hashlib.sha256(raw).hexdigest()}
    return {**metadata, "samples": samples}, {**metadata, "labels": labels}


def _table_cells(sample: dict, grid: list[list[str]]) -> list[dict]:
    cells = []
    for row, values in enumerate(grid):
        for column, value in enumerate(values):
            if not value.strip():
                continue
            cells.append({
                "evidence_id": _cell_id(sample["sample_id"], row, column), "doc_id": sample["sample_id"],
                "kind": "table_cell", "text": value,
                "locator": {"page_index": 0, "section": "Table", "table_index": 0,
                            "row_index": row, "column_index": column},
                "row_header": values[0] if row and column else "",
                "column_header": grid[0][column] if row else "",
            })
    return cells


def _metadata(document: object) -> None:
    if not isinstance(document, dict) or document.get("dataset") != _DATASET:
        raise ValueError("expected a SciTabAlign cell-ranking manifest")
    _identifier(document.get("dataset_version"), "SciTabAlign dataset_version")
    digest = document.get("source_sha256")
    if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
        raise ValueError("invalid SciTabAlign source SHA-256")


def run_scitabalign(manifest: object) -> list[dict]:
    """Rank given-table cells offline through the shared evidence index."""
    _metadata(manifest)
    fields = {"sample_id", "claim_id", "paper_id", "table_id", "claim", "caption", "grid"}
    metadata = {"dataset", "dataset_version", "source_sha256"}
    predictions = []
    for (sample_id, claim_id), sample, _ in _rows(manifest, "samples", fields, set(), metadata):
        if sample_id != _sample_id(claim_id):
            raise ValueError("invalid SciTabAlign sample ID")
        _identifier(sample["paper_id"], "SciTabAlign paper_id")
        _identifier(sample["table_id"], "SciTabAlign table_id")
        _text(sample["caption"], "caption")
        claim, grid = _text(sample["claim"], "claim"), sample["grid"]
        _shape(grid)
        if any(not isinstance(cell, str) for row in grid for cell in row):
            raise ValueError("SciTabAlign input grid may contain only text")
        with closing(EvidenceIndex(_table_cells(sample, grid))) as index:
            result = index.search(claim, limit=max(_DEFAULT_RANK_CUTOFFS))
        predictions.append({
            "sample_id": sample_id, "claim_id": claim_id, "review_status": "pending", "verdict": None,
            "evidence_ids": [], "candidate_ids": [item["evidence_id"] for item in result["evidence"]],
            "stop_reason": "ranking_only",
        })
    if not predictions:
        raise ValueError("SciTabAlign input manifest has no samples")
    return predictions


def _predicted_cells(sample_id: str, ids: tuple[str, ...], shape: tuple[int, int]) -> list[tuple[int, int]]:
    cells = []
    for eid in ids:
        prefix = f"{sample_id}:cell:"
        match = _COORDINATE.fullmatch(eid.removeprefix(prefix)) if eid.startswith(prefix) else None
        if match is None or not _within(cell := [int(part) for part in match.groups()], shape, first=0):
            raise ValueError(f"SciTabAlign {sample_id}: evidence ID {eid!r} is outside the given table")
        cells.append(tuple(cell))
    return cells


def _cell_scores(ranked: list[tuple[int, int]], expected: set[tuple[int, int]], k: int) -> dict:
    selected = set(ranked[:k])
    overlap = len(selected & expected)
    return {"precision": overlap / len(selected) if selected else 0.0,
            "recall": overlap / len(expected), "f1": 2 * overlap / (len(selected) + len(expected)),
            "complete_rationale": int(expected <= selected)}


def score_scitabalign(predictions: object, gold: object) -> dict:
    """Macro-average fixed-cutoff cell retrieval over aligned gold examples."""
    _metadata(gold)
    predicted = _parse_predictions(predictions)
    fields = {"sample_id", "claim_id", "shape", "alignment_status", "cells"}
    metadata = {"dataset", "dataset_version", "source_sha256"}
    metrics = {k: Counter() for k in _DEFAULT_RANK_CUTOFFS}
    alignment = Counter(dict.fromkeys(_STATUSES, 0))
    cases, seen, ranked_count = [], set(), 0
    for key, label, _ in _rows(gold, "labels", fields, set(), metadata):
        sample_id, claim_id = key
        if sample_id != _sample_id(claim_id):
            raise ValueError("invalid SciTabAlign gold sample ID")
        seen.add(key)
        shape, status, cells = label["shape"], label["alignment_status"], label["cells"]
        if (not isinstance(shape, list) or len(shape) != 2 or any(type(size) is not int for size in shape)
                or shape[0] < 2 or shape[1] < 1 or status not in _STATUSES or not isinstance(cells, list)
                or bool(cells) != (status == "aligned")):
            raise ValueError("invalid SciTabAlign gold shape or alignment status")
        grid_shape = (shape[0], shape[1])
        if any(not _within(cell, grid_shape, first=0) for cell in cells):
            raise ValueError("invalid normalized SciTabAlign gold coordinates")
        expected = set(map(tuple, cells))
        if len(expected) != len(cells):
            raise ValueError("repeated normalized SciTabAlign gold coordinates")
        alignment[status] += 1
        prediction = predicted.get(key)
        if prediction is not None and prediction.candidate_ids is None:
            raise ValueError("SciTabAlign cell predictions require ordered candidate IDs")
        ranked = _predicted_cells(sample_id, prediction.candidate_ids, grid_shape) if prediction is not None else []
        per_case = None
        if status == "aligned":
            ranked_count += int(prediction is not None)
            per_case = {str(k): _cell_scores(ranked, expected, k) for k in metrics}
            for k, totals in metrics.items():
                totals.update(per_case[str(k)])
        cases.append({"sample_id": sample_id, "claim_id": claim_id, "alignment_status": status,
                      "gold_cells": cells, "ranked_cells": [list(cell) for cell in ranked], "scores_at_k": per_case})
    if predicted.keys() - seen:
        raise ValueError("some predictions have no SciTabAlign gold claim")
    if not cases:
        raise ValueError("SciTabAlign gold manifest has no samples")
    aligned = alignment["aligned"]
    return {"schema_version": 1, "dataset": _DATASET, "dataset_version": gold["dataset_version"],
            "source_sha256": gold["source_sha256"], "task": "given_table_cell_ranking", "alignment": dict(alignment),
            "alignment_coverage": _ratio(aligned, len(cases)), "prediction_coverage": _ratio(ranked_count, aligned),
            "cell_retrieval_at_k": {str(k): {f"macro_{metric}": totals[metric] / aligned if aligned else None
                                             for metric in ("precision", "recall", "f1", "complete_rationale")}
                                    for k, totals in metrics.items()},
            "cases": cases}
