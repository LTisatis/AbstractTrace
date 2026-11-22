"""Reproducible calculations on scoped table operands."""

import re
import unicodedata
from collections import defaultdict
from decimal import Decimal, localcontext


def _normal(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _number(text: str) -> tuple[Decimal, str] | None:
    normalized = _normal(text)
    match = re.fullmatch(r"([+−-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))\s*(%)?", normalized)
    if not match:
        return None
    value = Decimal(match.group(1).replace("−", "-"))
    return value, "percent" if match.group(2) else "number"


def calculate(baseline: dict, target: dict, operation: str) -> dict:
    """Require operands from the same row of the same versioned table."""
    result = {
        "operation": operation,
        "evidence_ids": [baseline["evidence_id"], target["evidence_id"]],
        "operands": [baseline["text"], target["text"]],
    }
    a, b = baseline.get("locator", {}), target.get("locator", {})
    scope = ("page_index", "table_index", "row_index")
    if (
        baseline.get("kind") != "table_cell" or target.get("kind") != "table_cell"
        or baseline.get("doc_id") != target.get("doc_id")
        or not all(key in a and key in b and a[key] == b[key] for key in scope)
        or a.get("column_index") == b.get("column_index")
    ):
        return {**result, "status": "not_calculable", "reason": "operands do not share a comparable table row"}
    left, right = _number(baseline["text"]), _number(target["text"])
    if left is None or right is None or left[1] != right[1]:
        return {**result, "status": "not_calculable", "reason": "operand units or values are ambiguous"}
    original, unit = left
    revised = right[0]
    with localcontext() as context:
        context.prec = 28
        delta = revised - original
        if operation == "absolute_difference":
            formula, value, result_unit = "target - baseline", delta, "percentage_points" if unit == "percent" else unit
        elif operation == "percentage_points" and unit == "percent":
            formula, value, result_unit = "target - baseline", delta, "percentage_points"
        elif operation == "relative_change" and original != 0:
            formula, value, result_unit = "(target - baseline) / baseline * 100", delta / original * 100, "percent"
        else:
            return {**result, "status": "not_calculable", "reason": "operation is undefined for these operands"}
    return {**result, "status": "calculated", "formula": formula, "result": format(value.normalize(), "f"), "unit": result_unit}


def comparison_for_claim(text: str, candidates: list[dict]) -> tuple[dict, Decimal] | None:
    """Recognize explicit percentage-point/relative comparisons on named table rows."""
    points = re.search(r"([+−-]?\d+(?:\.\d+)?)\s*(?:percentage points?|百分点)\b", text, re.I)
    relative = re.search(r"(?:relative(?:ly)?\s+(?:increase\s+of\s+)?|相对(?:提高|增加)?)\s*(\d+(?:\.\d+)?)\s*%", text, re.I)
    change = points or relative
    if change is None:
        return None
    subjects = defaultdict(dict)
    normalized = _normal(text)
    for evidence in candidates:
        locator = evidence.get("locator", {})
        row = evidence.get("row_header")
        col = _normal(evidence.get("column_header") or "")
        if evidence.get("kind") != "table_cell" or not row or not col:
            continue
        row_key = _normal(row)
        subject_pattern = (r"(?:\bdataset\s+|\bdata set\s+|数据集\s*)" if len(row_key) == 1 else r"(?<!\w)")
        if not re.search(subject_pattern + re.escape(row_key) + r"(?!\w)", normalized):
            continue
        key = (evidence["doc_id"], locator["page_index"], locator["table_index"], locator["row_index"])
        if col in {"baseline", "baseline accuracy", "基线"}:
            subjects[key]["baseline"] = evidence
        if col in {"final", "final accuracy", "target", "ours", "result", "最终结果", "本方法"}:
            subjects[key]["target"] = evidence
    paired = [row for row in subjects.values() if {"baseline", "target"} <= row.keys()]
    if len(paired) != 1:
        return None
    pair = paired[0]
    calculation = calculate(pair["baseline"], pair["target"], "percentage_points" if points else "relative_change")
    if calculation["status"] != "calculated":
        return None
    expected = Decimal(change.group(1).replace("−", "-"))
    if expected > 0 and re.search(r"\b(?:decreas\w*|drop\w*|reduc\w*|fell)\b|降低|下降", normalized):
        expected = -expected
    return calculation, expected
