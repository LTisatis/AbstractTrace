"""Reproducible calculations on scoped table operands."""

import re
import unicodedata
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP, localcontext


_VALUE = r"[+−-]?(?:\d+(?:\.\d+)?|\.\d+)"
_QUANTITY = re.compile(rf"(?<![a-z0-9.])(?P<value>{_VALUE})\s*(?P<unit>percentage points?\b|个?百分点|pp\b|%)", re.I)
_LEVELS = re.compile(rf"(?:\bfrom\s+|从)({_VALUE})\s*(%)?\s*(?:to\s+|到|至)({_VALUE})\s*(%)?", re.I)
_CLAUSE_BREAK = re.compile(r"[;；。]|\.(?=\s+[a-z])|\b(?:and|but|whereas|while)\b|[，,]|并且|而")
_WIDE_SCOPE = re.compile(r"\b(?:all|each|every|both)\s+(?:data\s*sets?|splits?|groups?|samples?)\b|所有(?:数据集|样本)|各数据集", re.I)
_SPLIT = re.compile(r"\b(?:(?:fixed|held[- ]out|validation|test|train(?:ing)?|development|dev)[- ]+)+(?:split|set|partition)\b|(?:验证|测试|训练)集", re.I)
_COMPARE_TO = re.compile(r"\b(?:against|versus|vs\.?|compared\s+(?:to|with)|relative\s+to)\s+([^,;:。；，.]+)", re.I)
_DECREASE = re.compile(r"\b(?:decreas\w*|declin\w*|drop\w*|reduc\w*|lower\w*|fell|fall\w*)\b|降低|下降|减少", re.I)
_INCREASE = re.compile(r"\b(?:increas\w*|improv\w*|gain\w*|ris\w*|rose|higher|grow\w*|grew)\b|提高|上升|增加|提升", re.I)
_RELATIVE = re.compile(r"\b(?:relative(?:ly)?|by)\b|相对", re.I)
_LEVEL = re.compile(r"\b(?:was|were|is|of|reached|achieved|reported|at)\s*$|(?:为|达到|是)\s*$", re.I)
_METRIC = re.compile(r"\b(?:accuracy|f[- ]?1|precision|recall|auc|rmse|mae|loss|latency|throughput)\b|准确率|召回率|精确率|延迟|损失", re.I)
_TABLE = re.compile(r"\btable\s+[a-z]?\d+(?:\.\d+)?\b|表\s*[a-z]?\d+(?:\.\d+)?", re.I)
_PLAIN_CHANGE = re.compile(rf"\bby\s+({_VALUE})(?!\d|\.\d)")
_PLAIN_LEVEL = re.compile(rf"\b(?:was|were|reached|achieved|reported|at)\s+({_VALUE})(?!\d|\.\d)")


def _normal(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _number(text: str) -> tuple[Decimal, str] | None:
    normalized = _normal(text)
    match = re.fullmatch(r"([+−-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))\s*(%)?", normalized)
    if not match:
        return None
    value = Decimal(match.group(1).replace("−", "-"))
    return value, "percent" if match.group(2) else "number"


def _row_key(evidence: dict) -> tuple | None:
    location = evidence.get("locator", {})
    scope = (location.get("page_index"), location.get("table_index"), location.get("row_index"))
    if evidence.get("kind") != "table_cell" or not evidence.get("doc_id") or None in scope:
        return None
    return evidence["doc_id"], evidence.get("document_version"), evidence.get("version"), *scope


def calculate(baseline: dict, target: dict, operation: str) -> dict:
    """Require operands from the same row of the same versioned table."""
    result = {
        "operation": operation,
        "evidence_ids": [baseline["evidence_id"], target["evidence_id"]],
        "operands": [baseline["text"], target["text"]],
    }
    a, b = baseline.get("locator", {}), target.get("locator", {})
    if (_row_key(baseline) is None or _row_key(baseline) != _row_key(target)
            or None in (a.get("column_index"), b.get("column_index")) or a["column_index"] == b["column_index"]):
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
        elif operation in {"baseline", "target"}:
            formula, value, result_unit = operation, original if operation == "baseline" else revised, unit
        else:
            return {**result, "status": "not_calculable", "reason": "operation is undefined for these operands"}
    return {**result, "status": "calculated", "formula": formula, "result": format(value.normalize(), "f"), "unit": result_unit}


def _matches_subject(text: str, header: str) -> bool:
    subject = re.split(r"\s*[\[(]", header, maxsplit=1)[0]
    split = _SPLIT.search(subject)
    if split:
        subject = subject[:split.start()].rstrip(" -/,")
    subject = re.sub(r"^data\s*set\s+|^数据集\s*", "", subject)
    if not subject:
        return False
    prefix = r"(?:\bdata\s*sets?\s+|数据集\s*)" if len(subject) == 1 else r"(?<!\w)"
    return re.search(prefix + re.escape(subject) + r"(?![a-z0-9_-])", text) is not None


def _column_role(header: str) -> str | None:
    if re.match(r"(?:baseline\b|基线)", header):
        return "baseline"
    if re.match(r"(?:(?:final|target|ours|result|our method)\b|最终结果|本方法)", header):
        return "target"
    return None


def _comparison_baseline(text: str, fallback: str) -> str | None:
    candidate = _COMPARE_TO.search(text) or _COMPARE_TO.search(fallback)
    if candidate:
        label = re.split(r"\s+(?:on|in|for|under|using|with|accuracy|improved|increased)\b", candidate[1], maxsplit=1)[0]
    else:
        # Explicit named baselines also occur without a comparison preposition.
        named = re.search(r"\bbaseline\s+([a-z]\b|[0-9]+\b)", text)
        if not named:
            return None
        label = named[0]
    return re.sub(r"^(?:(?:the|same|reported|original)\s+)+", "", _normal(label)).strip()


def _scoped_pair(text: str, sentence: str, source: str, candidates: list[dict]) -> tuple[tuple[dict, dict, list[str]] | None, str]:
    by_row = defaultdict(list)
    for item in candidates:
        if key := _row_key(item):
            by_row[key].append(item)

    if _WIDE_SCOPE.search(text):
        return None, "the numerical comparison covers multiple scopes"
    expected_table = _TABLE.search(text) or _TABLE.search(sentence)
    expected_split = _SPLIT.search(text) or _SPLIT.search(sentence)
    if not expected_split and _SPLIT.search(source):
        return None, "the revision removed an explicit split from the comparison"
    metric_match = _METRIC.search(text) or _METRIC.search(sentence)
    metric = metric_match[0] if metric_match else None
    source_metric = _METRIC.search(source)
    baseline_name = _comparison_baseline(text, sentence)
    pairs = {}
    matching_rows = 0
    for row in by_row.values():
        subject = _normal(next((item.get("row_header", "") for item in row if item.get("row_header")), ""))
        if not _matches_subject(text, subject) and (
            re.search(r"\b(?:data\s*sets?|on|for|under)\b|数据集", text)
            or not _matches_subject(sentence, subject)
        ):
            continue
        if expected_table and not any(_normal(item["locator"].get("table_label", "")) == expected_table[0] for item in row):
            continue
        split = _SPLIT.search(subject)
        split_evidence = []
        if expected_split:
            if split and split[0] != expected_split[0]:
                continue
            if not split:
                split_evidence = [item["evidence_id"] for item in candidates
                                  if item.get("kind") == "body_text" and item.get("doc_id") == row[0]["doc_id"]
                                  and expected_split[0] in _normal(item["text"])
                                  and _matches_subject(_normal(item["text"]), subject)]
                if not split_evidence:
                    continue
        baseline = []
        target = []
        for item in row:
            header = _normal(item.get("column_header") or "")
            heading_metric = _METRIC.search(header)
            if metric and heading_metric and heading_metric[0] != metric:
                continue
            if _column_role(header) == "baseline" and (baseline_name in {None, "baseline", "基线"} or baseline_name == header):
                baseline.append(item)
            if _column_role(header) == "target":
                target.append(item)
        if metric and source_metric and source_metric[0] != metric and not all(
            _METRIC.search(_normal(item.get("column_header") or "")) for item in [*baseline, *target]
        ):
            continue
        if baseline or target:
            matching_rows += 1
        for first in baseline:
            for second in target:
                pairs[first["evidence_id"], second["evidence_id"]] = (first, second, split_evidence)
    if len(pairs) != 1 or matching_rows != 1:
        return None, "the revised dataset, split, metric, or baseline did not identify one comparable table pair"
    return next(iter(pairs.values())), ""


def _clauses(text: str) -> list[str]:
    clauses = []
    start = 0
    for separator in _CLAUSE_BREAK.finditer(text):
        if _stated_values(text[start:separator.start()], text) and _stated_values(text[separator.end():], text):
            clauses.append(text[start:separator.start()])
            start = separator.end()
    return [*clauses, text[start:]]


def _direction(text: str) -> int:
    directions = [(match.start(), sign) for pattern, sign in ((_DECREASE, -1), (_INCREASE, 1))
                  for match in pattern.finditer(text)]
    return max(directions)[1] if directions else 0


def _stated_values(clause: str, sentence: str) -> list[tuple[str | None, Decimal, str, bool]]:
    values = {}
    for match in _LEVELS.finditer(clause):
        if not (match[2] or match[4] or _METRIC.search(clause)):
            continue
        for operation, group in (("baseline", 1), ("target", 3)):
            unit = "percent" if match[group + 1] else "number"
            values[match.start(group)] = (operation, Decimal(match[group].replace("−", "-")), unit, False)
    for token in _QUANTITY.finditer(clause):
        if token.start() in values:
            continue
        value = Decimal(token["value"].replace("−", "-"))
        prefix = clause[:token.start()]
        suffix = clause[token.end():]
        if re.match(r"\s*(?:ci\b|confidence (?:interval|level)\b|置信区间)", suffix):
            continue
        direction = _direction(prefix)
        if not direction and prefix.strip() in {"", "by", "相对"}:
            direction = _direction(sentence[:sentence.find(clause)])
        if token["unit"] != "%":
            operation = "percentage_points"
        elif re.match(r"\s*(?:baseline\b|基线)", suffix) or re.search(r"\bfrom\s+(?:a\s+|the\s+)?$", prefix):
            operation = "baseline"
        elif _LEVEL.search(prefix):
            operation = "baseline" if re.search(r"\bbaseline\b|基线", prefix) else "target"
        elif _RELATIVE.search(prefix) or direction:
            operation = "relative_change"
        else:
            operation = None
        relative = operation in {"percentage_points", "relative_change"}
        conflict = relative and (direction > 0 and value < 0 or direction < 0 and token["value"].startswith("+"))
        if relative and direction < 0 and value > 0:
            value = -value
        values[token.start()] = (operation, value, "percent" if token["unit"] == "%" else "percentage_points", conflict)
    for match in _PLAIN_CHANGE.finditer(clause):
        if match.start(1) in values or not _METRIC.search(sentence):
            continue
        direction = _direction(clause[:match.start()])
        if direction:
            value = Decimal(match[1].replace("−", "-"))
            conflict = direction > 0 and value < 0 or direction < 0 and match[1].startswith("+")
            values[match.start(1)] = ("absolute_difference", value if direction > 0 else -abs(value), "number", conflict)
    for match in _PLAIN_LEVEL.finditer(clause):
        prefix = clause[:match.start()]
        if match.start(1) not in values and _METRIC.search(prefix):
            operation = "baseline" if re.search(r"\bbaseline\b|基线", prefix) else "target"
            values[match.start(1)] = (operation, Decimal(match[1].replace("−", "-")), "number", False)
    return [values[index] for index in sorted(values)]


def check_revision_numbers(source: str, replacement: str, evidence: list[dict]) -> list[dict]:
    """Recompute recognized numeric claims and preserve findings that need manual review."""
    source, replacement = _normal(source), _normal(replacement)
    records = []
    for clause in _clauses(replacement):
        values = _stated_values(clause, replacement)
        if not values:
            continue
        pair, reason = _scoped_pair(clause, replacement, source, evidence)
        for operation, stated, unit, direction_conflict in values:
            if operation is None or pair is None:
                result = {
                    "status": "not_calculable", "operation": operation or "unparsed_percentage",
                    "operands": [], "evidence_ids": [], "reason": reason or "the stated percentage has no clear operation",
                }
            else:
                baseline, target, scope_ids = pair
                result = calculate(baseline, target, operation)
                if scope_ids:
                    result["scope_evidence_ids"] = scope_ids
                if result.get("status") == "calculated" and result["unit"] != unit:
                    result.update(status="not_calculable", reason="the stated value and the calculation use incompatible units")
            verification = "unresolved"
            if result["status"] == "calculated":
                calculated = Decimal(result["result"])
                with localcontext() as context:
                    context.prec = max(28, len(calculated.as_tuple().digits) + abs(stated.as_tuple().exponent) + 1)
                    rounded = calculated.quantize(Decimal(1).scaleb(stated.as_tuple().exponent), rounding=ROUND_HALF_UP)
                verification = "matched" if rounded == stated and not direction_conflict else "mismatch"
            if direction_conflict:
                result["reason"] = "the revised wording gives conflicting directions for the same change"
            records.append({**result, "stated": format(stated, "f"), "verification": verification})
    return records


def comparison_for_claim(text: str, candidates: list[dict]) -> tuple[dict, Decimal] | None:
    """Recognize explicit percentage-point/relative comparisons on named table rows."""
    records = check_revision_numbers("", text, candidates)
    if len(records) != 1 or records[0]["status"] != "calculated" or records[0].get("reason"):
        return None
    record = records[0]
    if record["operation"] not in {"percentage_points", "relative_change"}:
        return None
    return {key: value for key, value in record.items() if key not in {"stated", "verification"}}, Decimal(record["stated"])
