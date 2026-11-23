"""Score human checks against the exact review version recorded by a benchmark."""

from decimal import Decimal, InvalidOperation


_HEADER_ROLES = {"column_header", "row_header"}
_OPERAND_ROLES = {"value", "table_cell"}
_REVISION_FLAGS = ("corrected", "facts_preserved", "unsupported_added")


def _versioned(document: object, label: str) -> dict:
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError(f"{label} requires schema_version 1")
    return document


def _records(document: dict, key: str, *, optional: bool = False) -> list[dict]:
    rows = document.get(key, [] if optional else None)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{key} must be a list of objects")
    return rows


def _identifier(row: dict, field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _indexed(rows: list[dict], *fields: str) -> dict:
    result = {}
    for row in rows:
        key = tuple(_identifier(row, field) for field in fields)
        if key in result:
            raise ValueError(f"duplicate {'/'.join(fields)}: {'/'.join(key)}")
        result[key] = row
    return result


def _ids(row: dict, field: str, *, optional: bool = False) -> list[str]:
    items = row.get(field, [] if optional else None)
    if (not isinstance(items, list) or any(not isinstance(item, str) or not item.strip() for item in items)
            or len(items) != len(set(items))):
        raise ValueError(f"{field} must contain unique nonempty evidence IDs")
    return items


def _ratio(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def _fraction(rows: list[dict], flag: str) -> dict:
    known = [row[flag] for row in rows if row[flag] is not None]
    return _ratio(sum(known), len(known))


def _number(value: object) -> Decimal | None:
    if type(value) not in {str, int, float}:
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _current_review(samples: dict, key: tuple[str, ...]) -> dict:
    sample_id, version_id = key[:2]
    sample = samples.get((sample_id,))
    if sample is None:
        raise ValueError(f"unknown quality sample {sample_id!r}")
    review = sample.get("review")
    if not isinstance(review, dict) or review.get("version_id") != version_id:
        raise ValueError(f"quality version {version_id!r} does not match the recorded review for {sample_id!r}")
    return review


def _score_revision(label: dict, review: dict) -> dict:
    revision_id = label["revision_id"]
    revision = _indexed(_records(review, "revisions"), "revision_id").get((revision_id,))
    if revision is None:
        raise ValueError(f"unknown revision {revision_id!r} for the recorded version")
    flags = {}
    for key in _REVISION_FLAGS:
        if key not in label or (label[key] is not None and type(label[key]) is not bool):
            raise ValueError(f"{key} must be true, false or null")
        flags[key] = label[key]
    unsafe = flags["corrected"] is False or flags["facts_preserved"] is False or flags["unsupported_added"] is True
    human_safe = False if unsafe else True if None not in flags.values() else None
    status = _identifier(revision, "status")
    return {**flags, "status": status, "human_safe": human_safe,
            "wrong_pass": (status == "passed") if human_safe is False else None,
            "wrong_return": (status == "returned") if human_safe is True else None,
            "deferred": status not in {"passed", "returned"}}


def _score_abstract(label: dict, review: dict) -> dict:
    expected = label.get("has_conflict")
    if type(expected) is not bool:
        raise ValueError("has_conflict must be a boolean")
    verification = review.get("abstract_review")
    if not isinstance(verification, dict) or verification.get("version_id", review["version_id"]) != review["version_id"]:
        raise ValueError("abstract verification does not match the recorded review version")
    status = _identifier(verification, "status")
    issues = _records(verification, "issues", optional=True)
    detected = True if status == "returned" and issues else False if status == "passed" and not issues else None
    return {"status": status, "expected_conflict": expected, "detected_conflict": detected, "issue_count": len(issues),
            "correct": detected is expected, "deferred": detected is None,
            "conflict_found": (detected is True) if expected else None,
            "false_conflict": (detected is True) if not expected else None}


def _score_table(label: dict, review: dict) -> dict:
    operands, headers, notes = (_ids(label, name) for name in ("operand_ids", "header_ids", "note_ids"))
    expected_result, expected_unit = _number(label.get("result")), _identifier(label, "unit")
    if not operands or len(set(operands + headers + notes)) != len(operands + headers + notes) or expected_result is None:
        raise ValueError("table checks need distinct operand, header and note roles and a finite result")
    calculations = _indexed(_records(review, "calculations"), "claim_id", "operation")
    result = calculations.get((label["claim_id"], label["operation"]))
    if result is None:
        return dict.fromkeys(("operands_correct", "headers_correct", "notes_correct", "calculation_correct",
                              "joint_correct", "calculation_present"), False)
    roles = review.get("evidence_roles")
    if not isinstance(roles, dict) or any(not isinstance(value, str) for value in roles.values()):
        raise ValueError("evidence_roles must map evidence IDs to role names")
    actual_operands = _ids(result, "evidence_ids")
    actual_scope = _ids(result, "scope_evidence_ids", optional=True)
    actual_headers = {eid for eid in actual_scope if roles.get(eid) in _HEADER_ROLES}
    actual_notes = {eid for eid in actual_scope if roles.get(eid) == "table_note"}
    operands_correct = actual_operands == operands and all(roles.get(eid) in _OPERAND_ROLES for eid in actual_operands)
    headers_correct = actual_headers == set(headers)
    notes_correct = actual_notes == set(notes)
    calculation_correct = (result.get("status", "calculated") == "calculated"
                           and _number(result.get("result")) == expected_result and result.get("unit") == expected_unit)
    return {"operands_correct": operands_correct, "headers_correct": headers_correct, "notes_correct": notes_correct,
            "calculation_correct": calculation_correct, "calculation_present": True,
            "joint_correct": all((operands_correct, headers_correct, notes_correct, calculation_correct)),
            "evidence_ids": actual_operands, "scope_evidence_ids": actual_scope}


def score_review_quality(run_record: object, gold: object) -> dict:
    """Score independently annotated revision, whole-abstract and table checks.

    A null annotation is unassessed. Manual review is reported as deferred rather
    than as a final return. Table scope is compared by role because the same
    calculation may also cite a caption or a methods paragraph.
    """
    run_record = _versioned(run_record, "run record")
    gold = _versioned(gold, "review quality gold")
    samples = _indexed(_records(run_record, "samples"), "sample_id")
    cases = {}
    for name, extra, scorer in (
        ("revision_checks", ("revision_id",), _score_revision),
        ("abstract_checks", (), _score_abstract),
        ("table_checks", ("claim_id", "operation"), _score_table),
    ):
        fields = ("sample_id", "version_id", *extra)
        cases[name] = [{**dict(zip(fields, key)), **scorer(row, _current_review(samples, key))}
                       for key, row in _indexed(_records(gold, name, optional=True), *fields).items()]
    revisions, abstracts, tables = (cases[key] for key in ("revision_checks", "abstract_checks", "table_checks"))
    return {
        "schema_version": 1,
        "revisions": {name: _fraction(revisions, flag) for name, flag in (
            ("correction_rate", "corrected"), ("fact_preservation_rate", "facts_preserved"),
            ("unsupported_addition_rate", "unsupported_added"), ("wrong_pass_rate", "wrong_pass"),
            ("wrong_return_rate", "wrong_return"), ("deferred_rate", "deferred"),
        )},
        "abstracts": {name: _fraction(abstracts, flag) for name, flag in (
            ("accuracy", "correct"), ("conflict_recall", "conflict_found"),
            ("false_conflict_rate", "false_conflict"), ("deferred_rate", "deferred"),
        )},
        "tables": {name: _fraction(tables, flag) for name, flag in (
            ("operand_accuracy", "operands_correct"), ("header_accuracy", "headers_correct"),
            ("note_accuracy", "notes_correct"), ("calculation_accuracy", "calculation_correct"),
            ("joint_accuracy", "joint_correct"), ("calculation_coverage", "calculation_present"),
        )},
        "cases": cases,
        "errors": [
            *({**row, "phase": "revision"} for row in revisions if row["human_safe"] is False or row["wrong_return"]),
            *({**row, "phase": "version_verification"} for row in abstracts if not row["correct"]),
            *({**row, "phase": "table_calculation"} for row in tables if not row["joint_correct"]),
        ],
    }
