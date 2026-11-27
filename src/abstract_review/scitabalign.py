"""Given-table cell ranking and model selection over a local SciTabAlign release.

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

from pydantic import StrictInt

from .evidence import EvidenceIndex, _MAX_CANDIDATES
from .models import ModelPaused, ModelUnavailable, Schema
from .scoring import _DEFAULT_RANK_CUTOFFS, _identifier, _parse_predictions, _ratio, _rows, semantic_manifest_sha256
from .table_ranking import _label_terms, rank_table_cells


_DATASET = "scitabalign-cells"
_STATUSES = ("aligned", "empty_explanation", "invalid_coordinates")
_COORDINATE = re.compile(r"(0|[1-9][0-9]*):(0|[1-9][0-9]*)")
_SELECTION_LIMIT = max(_DEFAULT_RANK_CUTOFFS)
_WIDE_SELECTION_LIMIT = 16
_TABLE_CUTOFFS = (*_DEFAULT_RANK_CUTOFFS, _WIDE_SELECTION_LIMIT)
_BROAD_SCOPE = re.compile(
    r"\b(?:all|each|every|across|throughout|highest|lowest|best|worst|in general)\b", re.IGNORECASE,
)
_COMPARISON = re.compile(
    r"\b(?:than|compared?|comparable|baselines?|outperform\w*|dominat\w*|improv\w*|gap|"
    r"difference|margin|differ\w*|increas\w*|decreas\w*|greater|lower|higher|no effect|negligible)\b",
    re.IGNORECASE,
)


class _CellCoordinate(Schema):
    row: StrictInt
    column: StrictInt


class _CellSelection(Schema):
    cells: list[_CellCoordinate]


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


def _load_inputs(manifest: object) -> list[dict]:
    _metadata(manifest)
    fields = {"sample_id", "claim_id", "paper_id", "table_id", "claim", "caption", "grid"}
    metadata = {"dataset", "dataset_version", "source_sha256"}
    samples = []
    for (sample_id, claim_id), sample, _ in _rows(manifest, "samples", fields, set(), metadata):
        if sample_id != _sample_id(claim_id):
            raise ValueError("invalid SciTabAlign sample ID")
        _identifier(sample["paper_id"], "SciTabAlign paper_id")
        _identifier(sample["table_id"], "SciTabAlign table_id")
        _text(sample["caption"], "caption")
        _text(sample["claim"], "claim")
        grid = sample["grid"]
        _shape(grid)
        if any(not isinstance(cell, str) for row in grid for cell in row):
            raise ValueError("SciTabAlign input grid may contain only text")
        samples.append(sample)
    if not samples:
        raise ValueError("SciTabAlign input manifest has no samples")
    return samples


def _selection_limit(sample: dict) -> int:
    """Expand the budget when named rows and columns imply a larger evidence group."""
    grid = sample["grid"]
    claim = sample["claim"]
    if len(grid) * len(grid[0]) <= _SELECTION_LIMIT:
        return _SELECTION_LIMIT
    query = _label_terms(claim)

    def matched(labels):
        terms = [_label_terms(label) for label in labels]
        frequency = Counter(term for label in terms for term in label)
        return sum(any(frequency[term] == 1 for term in label & query) for label in terms)

    rows = matched([row[0] for row in grid[1:]])
    columns = matched(grid[0][1:])
    if _BROAD_SCOPE.search(claim):
        if columns and rows < 2:
            rows = len(grid) - 1
        elif rows and not columns:
            columns = len(grid[0]) - 1
        elif not rows and not columns:
            rows = len(grid) - 1
    elif _COMPARISON.search(claim) and rows < 2 and columns < 2:
        # An unnamed comparator can refer to several eligible rows or columns.
        return _WIDE_SELECTION_LIMIT
    rows, columns = max(1, rows), max(1, columns)
    needed = rows * columns + rows + columns  # values, row labels, and column headers
    return _WIDE_SELECTION_LIMIT if needed > _SELECTION_LIMIT else _SELECTION_LIMIT


def _offline_selection(sample: dict, limit: int) -> tuple[list[str], str]:
    cells = _table_cells(sample, sample["grid"])
    with closing(EvidenceIndex(cells)) as index:
        seeds = [item["evidence_id"] for item in index.search(sample["claim"], limit=_MAX_CANDIDATES)["evidence"]]
    return rank_table_cells(sample["claim"], cells, seeds, limit=limit), "ranking_only"


def _model_selection(sample: dict, model, limit: int) -> tuple[list[str], str]:
    material = {key: sample[key] for key in ("claim", "caption", "grid")}
    choice = model.call("select_table_cells", {**material, "selection_limit": limit}, _CellSelection)
    if not isinstance(choice, _CellSelection):
        return [], "invalid_model_cell_selection"
    coordinates = [[cell.row, cell.column] for cell in choice.cells]
    shape = _shape(sample["grid"])
    if (len(coordinates) > limit or any(not _within(cell, shape, first=0) for cell in coordinates)
            or len(set(map(tuple, coordinates))) != len(coordinates)):
        return [], "invalid_model_cell_selection"
    reason = "model_cell_selection" if coordinates else "model_cell_selection_empty"
    return [_cell_id(sample["sample_id"], row, column) for row, column in coordinates], reason


def run_scitabalign(manifest: object, *, model=None, batch=None, cell_limit: int | None = None) -> list[dict]:
    """Select ranked evidence from the given table under a fixed or adaptive budget."""
    if batch is not None and model is None:
        raise ValueError("SciTabAlign batch checkpoints are for model selection")
    if cell_limit is not None and (type(cell_limit) is not int or cell_limit not in (_SELECTION_LIMIT, _WIDE_SELECTION_LIMIT)):
        raise ValueError("SciTabAlign cell limit must be 8 or 16")
    samples = _load_inputs(manifest)
    predictions = []
    for sample in samples:
        sample_id, claim_id = sample["sample_id"], sample["claim_id"]
        if batch is not None:
            input_digest = semantic_manifest_sha256([batch.identity, sample])
            if previous := batch.saved(sample_id, input_digest):
                predictions.append(previous)
                continue
            batch.stop_reason = batch.exhaustion()
            if batch.stop_reason is not None:
                break
            before, event_offset = batch.requests_used, len(model.events)
        try:
            if model is None:
                selected, stop_reason = _offline_selection(sample, cell_limit or _SELECTION_LIMIT)
            else:
                selected, stop_reason = _model_selection(sample, model, cell_limit or _selection_limit(sample))
        except ModelUnavailable as exc:
            if batch is None and isinstance(exc, ModelPaused):
                raise
            events = model.events[event_offset:] if batch is not None else []
            request_failed = events and type(events[-1].get("status")) is int
            if batch is not None and (isinstance(exc, ModelPaused) or request_failed):
                reason = "model_cell_selection_paused" if isinstance(exc, ModelPaused) else "model_cell_selection_failed"
                batch.pause(sample_id, input_digest, batch.requests_used - before, events, finding={"stop_reason": reason})
                break
            selected, stop_reason = [], "model_cell_selection_failed"
        prediction = {
            "sample_id": sample_id, "claim_id": claim_id, "review_status": "pending", "verdict": None,
            "evidence_ids": [], "candidate_ids": selected, "stop_reason": stop_reason,
        }
        if batch is not None:
            batch.record(sample_id, input_digest, prediction, batch.requests_used - before, model.events[event_offset:])
        predictions.append(prediction)
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
    metrics = {k: Counter() for k in _TABLE_CUTOFFS}
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
                scores = per_case[str(k)]
                totals.update(scores)
                totals["recall_ceiling"] += min(k, len(expected)) / len(expected)
                if len(expected) <= k:
                    totals["eligible"] += 1
                    totals["eligible_recall"] += scores["recall"]
                else:
                    totals["over_capacity_recall"] += scores["recall"]
        cases.append({"sample_id": sample_id, "claim_id": claim_id, "alignment_status": status,
                      "gold_cells": cells, "ranked_cells": [list(cell) for cell in ranked], "scores_at_k": per_case})
    if predicted.keys() - seen:
        raise ValueError("some predictions have no SciTabAlign gold claim")
    if not cases:
        raise ValueError("SciTabAlign gold manifest has no samples")
    aligned = alignment["aligned"]
    capacity = {}
    for k, totals in metrics.items():
        eligible = totals["eligible"]
        over_capacity = aligned - eligible
        capacity[str(k)] = {
            "complete_rationale_ceiling": _ratio(eligible, aligned),
            "eligible_complete_rationale_recall": _ratio(totals["complete_rationale"], eligible),
            "eligible_macro_recall": totals["eligible_recall"] / eligible if eligible else None,
            "over_capacity_macro_recall": (
                totals["over_capacity_recall"] / over_capacity if over_capacity else None
            ),
            "macro_recall_ceiling": totals["recall_ceiling"] / aligned if aligned else None,
        }
    return {"schema_version": 1, "dataset": _DATASET, "dataset_version": gold["dataset_version"],
            "source_sha256": gold["source_sha256"], "task": "given_table_cell_ranking", "alignment": dict(alignment),
            "alignment_coverage": _ratio(aligned, len(cases)), "prediction_coverage": _ratio(ranked_count, aligned),
            "cell_retrieval_at_k": {str(k): {f"macro_{metric}": totals[metric] / aligned if aligned else None
                                             for metric in ("precision", "recall", "f1", "complete_rationale")}
                                    for k, totals in metrics.items()},
            "rationale_capacity_at_k": capacity,
            "cases": cases}
