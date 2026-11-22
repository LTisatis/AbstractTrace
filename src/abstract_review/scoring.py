from collections.abc import Iterator
from dataclasses import dataclass


_VERDICTS = ("supported", "refuted", "insufficient")
_REVIEW_STATUSES = ("completed", "pending", "manual_conflict")
type _CaseKey = tuple[str, str]
type _Ratio = dict[str, int | float | None]


@dataclass(frozen=True)
class _Prediction:
    status: str
    verdict: str | None
    evidence_ids: frozenset[str]


@dataclass(frozen=True)
class _GoldLabel:
    verdict: str
    evidence_groups: tuple[frozenset[str], ...]


def score_predictions(predictions: object, gold: object) -> dict[str, object]:
    """Score stored predictions against a separately supplied gold manifest."""
    predicted = _parse_predictions(predictions)
    gold_labels = _parse_gold(gold)
    unexpected = sorted(predicted.keys() - gold_labels.keys())
    if unexpected:
        sample_id, claim_id = unexpected[0]
        raise ValueError(f"Prediction has no gold label for {sample_id}/{claim_id}")

    completed_count = correct_count = evidence_count = evidence_hits = joint_hits = 0
    cases: list[dict[str, object]] = []

    for key, label in gold_labels.items():
        prediction = predicted.get(key)
        status = prediction.status if prediction is not None else None
        verdict = prediction.verdict if prediction is not None else None
        completed = status == "completed"
        correct = completed and verdict == label.verdict
        completed_count += int(completed)
        correct_count += int(correct)

        evidence_hit = None
        if label.evidence_groups:
            submitted_ids = prediction.evidence_ids if prediction is not None else frozenset()
            evidence_hit = any(group <= submitted_ids for group in label.evidence_groups)
            evidence_count += 1
            evidence_hits += int(evidence_hit)
            joint_hits += int(correct and evidence_hit)

        cases.append({
            "sample_id": key[0],
            "claim_id": key[1],
            "review_status": status,
            "predicted_verdict": verdict,
            "gold_verdict": label.verdict,
            "label_correct": correct,
            "evidence_group_hit": evidence_hit,
        })

    return {
        "schema_version": 1,
        "accuracy": _ratio(correct_count, len(gold_labels)),
        "evidence_group_recall": _ratio(evidence_hits, evidence_count),
        "joint_accuracy": _ratio(joint_hits, evidence_count),
        "completed_coverage": _ratio(completed_count, len(gold_labels)),
        "cases": cases,
    }


def _parse_predictions(document: object) -> dict[_CaseKey, _Prediction]:
    fields = {"sample_id", "claim_id", "review_status", "verdict", "evidence_ids"}
    predictions = {}
    for key, row, location in _rows(document, "predictions", fields):
        status = row["review_status"]
        verdict = row["verdict"]
        if not isinstance(status, str) or status not in _REVIEW_STATUSES:
            raise ValueError(f"{location}.review_status is invalid")
        if status == "completed":
            if not isinstance(verdict, str) or verdict not in _VERDICTS:
                raise ValueError(f"{location}.verdict must be valid for a completed prediction")
        elif verdict is not None:
            raise ValueError(f"{location}.verdict must be null for an incomplete prediction")
        evidence_ids = _evidence_ids(row["evidence_ids"], f"{location}.evidence_ids")
        predictions[key] = _Prediction(status, verdict, evidence_ids)
    return predictions


def _parse_gold(document: object) -> dict[_CaseKey, _GoldLabel]:
    fields = {"sample_id", "claim_id", "verdict", "accepted_evidence_groups"}
    labels = {}
    for key, row, location in _rows(document, "labels", fields):
        verdict = row["verdict"]
        if not isinstance(verdict, str) or verdict not in _VERDICTS:
            raise ValueError(f"{location}.verdict is invalid")

        group_location = f"{location}.accepted_evidence_groups"
        group_rows = row["accepted_evidence_groups"]
        if not isinstance(group_rows, list):
            raise ValueError(f"{group_location} must be a list")
        groups = tuple(
            _evidence_ids(group, f"{group_location}[{index}]")
            for index, group in enumerate(group_rows)
        )
        if any(not group for group in groups):
            raise ValueError(f"{group_location} must not contain an empty group")
        labels[key] = _GoldLabel(verdict, groups)
    return labels


def _rows(
    document: object, collection: str, fields: set[str]
) -> Iterator[tuple[_CaseKey, dict[str, object], str]]:
    if not isinstance(document, dict) or set(document) != {"schema_version", collection}:
        raise ValueError(f"Input must contain only schema_version and {collection}")
    version = document["schema_version"]
    if type(version) is not int or version != 1:
        raise ValueError(f"Unsupported {collection} schema_version")
    rows = document[collection]
    if not isinstance(rows, list):
        raise ValueError(f"{collection} must be a list")

    seen: set[_CaseKey] = set()
    for index, row in enumerate(rows):
        location = f"{collection}[{index}]"
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError(f"{location} must contain exactly {', '.join(sorted(fields))}")
        key = (
            _identifier(row["sample_id"], f"{location}.sample_id"),
            _identifier(row["claim_id"], f"{location}.claim_id"),
        )
        if key in seen:
            raise ValueError(f"{location} repeats {key[0]}/{key[1]}")
        seen.add(key)
        yield key, row, location


def _evidence_ids(value: object, location: str) -> frozenset[str]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    ids = frozenset(_identifier(item, location) for item in value)
    if len(ids) != len(value):
        raise ValueError(f"{location} contains duplicate evidence IDs")
    return ids


def _identifier(value: object, location: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{location} must be a non-empty ID without surrounding whitespace")
    return value


def _ratio(numerator: int, denominator: int) -> _Ratio:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }
