import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass


_VERDICTS = ("supported", "refuted", "insufficient")
_REVIEW_STATUSES = ("completed", "pending", "manual_conflict")
_DEFAULT_RANK_CUTOFFS = (1, 3, 5, 8)
type _CaseKey = tuple[str, str]
type _Ratio = dict[str, int | float | None]


@dataclass(frozen=True)
class _Anchor:
    sentence_id: str
    start: int
    end: int
    proposition: str


@dataclass(frozen=True)
class _Prediction:
    status: str
    verdict: str | None
    evidence_ids: frozenset[str]
    candidate_ids: tuple[str, ...] | None
    anchor: _Anchor | None
    stop_reason: str | None


@dataclass(frozen=True)
class _GoldLabel:
    verdict: str
    evidence_groups: tuple[frozenset[str], ...]
    anchor: _Anchor | None
    split: str | None


def score_predictions(
    predictions: object, gold: object, *, rank_cutoffs: tuple[int, ...] = _DEFAULT_RANK_CUTOFFS,
) -> dict[str, object]:
    """Score stored predictions against a separately supplied gold manifest."""
    if not rank_cutoffs or any(type(k) is not int or k <= 0 for k in rank_cutoffs):
        raise ValueError("rank cutoffs must be positive integers")
    cutoffs = tuple(sorted(set(rank_cutoffs)))
    predicted = _parse_predictions(predictions)
    gold_labels = _parse_gold(gold)
    extraction_samples = _extraction_samples(gold, gold_labels)
    matches = _match_extractions(predicted, gold_labels, extraction_samples)
    unexpected = sorted(key for key in predicted.keys() - gold_labels.keys() if key[0] not in extraction_samples)
    if unexpected:
        sample_id, claim_id = unexpected[0]
        raise ValueError(f"Prediction has no gold label for {sample_id}/{claim_id}")

    completed_count = correct_count = evidence_count = evidence_hits = joint_hits = known_ranks = 0
    unknown_ranks = False
    rank_hits = dict.fromkeys(cutoffs, 0)
    cases: list[dict[str, object]] = []
    errors = []

    for key, label in gold_labels.items():
        prediction_key = matches.get(key) if key[0] in extraction_samples else key
        prediction = predicted.get(prediction_key) if prediction_key is not None else None
        status = prediction.status if prediction is not None else None
        verdict = prediction.verdict if prediction is not None else None
        completed = status == "completed"
        correct = completed and verdict == label.verdict
        completed_count += int(completed)
        correct_count += int(correct)

        evidence_hit = ranked_hits = None
        submitted_ids = prediction.evidence_ids if prediction is not None else frozenset()
        candidates = prediction.candidate_ids if prediction is not None else ()
        if label.evidence_groups:
            evidence_hit = any(group <= submitted_ids for group in label.evidence_groups)
            evidence_count += 1
            evidence_hits += int(evidence_hit)
            joint_hits += int(correct and evidence_hit)
            if candidates is None:
                unknown_ranks = True
            else:
                known_ranks += int(prediction is not None)
                ranked_hits = {k: _contains_evidence(label.evidence_groups, candidates[:k]) for k in cutoffs}
                for k, hit in ranked_hits.items():
                    rank_hits[k] += int(hit)

        row = {
            "sample_id": key[0],
            "claim_id": key[1],
            "predicted_claim_id": prediction_key[1] if prediction is not None else None,
            "review_status": status,
            "predicted_verdict": verdict,
            "gold_verdict": label.verdict,
            "label_correct": correct,
            "evidence_group_hit": evidence_hit,
            "evidence_ids": sorted(submitted_ids),
            "candidate_ids": list(candidates) if prediction is not None and candidates is not None else None,
            "ranked_evidence_group_hit": {str(k): hit for k, hit in ranked_hits.items()} if ranked_hits is not None else None,
        }
        if label.split is not None:
            row["split"] = label.split
        if prediction is not None and prediction.stop_reason is not None:
            row["stop_reason"] = prediction.stop_reason
        cases.append(row)
        phase = _failure_phase(prediction, label, correct, evidence_hit)
        if phase is not None:
            errors.append({**row, "phase": phase})

    result = {
        "schema_version": 1,
        "accuracy": _ratio(correct_count, len(gold_labels)),
        "evidence_group_recall": _ratio(evidence_hits, evidence_count),
        "joint_accuracy": _ratio(joint_hits, evidence_count),
        "completed_coverage": _ratio(completed_count, len(gold_labels)),
        "ranked_evidence_coverage": _ratio(known_ranks, evidence_count),
        "evidence_group_recall_at_k": {
            str(k): None if unknown_ranks or not known_ranks else _ratio(rank_hits[k], evidence_count)
            for k in cutoffs
        },
        "cases": cases,
        "errors": errors,
        **classification_metrics(cases),
    }
    if extraction_samples:
        result["extraction"] = _extraction_metrics(predicted, gold_labels, extraction_samples, matches)
    splits = sorted({label.split for label in gold_labels.values() if label.split is not None})
    if splits:
        result["classification_by_split"] = {
            split: classification_metrics(row for row in cases if row.get("split") == split) for split in splits
        }
    return result


def classification_metrics(cases: Iterable[dict[str, object]]) -> dict[str, object]:
    confusion = {actual: dict.fromkeys(_VERDICTS, 0) for actual in _VERDICTS}
    gold_counts = dict.fromkeys(_VERDICTS, 0)
    correct = completed = unsupported = false_support = 0
    for row in cases:
        actual, predicted = row["gold_verdict"], row["predicted_verdict"]
        gold_counts[actual] += 1
        unsupported += int(actual != "supported")
        if row["review_status"] == "completed":
            confusion[actual][predicted] += 1
            false_support += int(actual != "supported" and predicted == "supported")
            correct += int(actual == predicted)
            completed += 1

    by_class = {}
    for label in _VERDICTS:
        actual_count = sum(confusion[label].values())
        predicted_count = sum(confusion[row][label] for row in _VERDICTS)
        f1 = _ratio(2 * confusion[label][label], actual_count + predicted_count)
        by_class[label] = {
            "gold_count": gold_counts[label], "completed_gold_count": actual_count,
            "predicted_count": predicted_count, "f1": f1,
        }
    macro = sum(item["f1"]["value"] or 0.0 for item in by_class.values()) / len(_VERDICTS)
    return {
        "completed_accuracy": _ratio(correct, completed),
        "confusion_matrix": confusion,
        "macro_f1": macro if completed else None,
        "classification_by_label": by_class,
        "unsupported_as_support": _ratio(false_support, unsupported),
    }


def _failure_phase(prediction: _Prediction | None, gold: _GoldLabel,
                   correct: bool, evidence_hit: bool | None) -> str | None:
    if prediction is None:
        return "extraction" if gold.anchor is not None else "missing_prediction"
    if correct and evidence_hit is not False:
        return None
    if prediction.stop_reason in {"unreadable", "parse_error", "key_source_unreadable"}:
        return "parsing"
    if prediction.stop_reason in {"budget_exhausted", "model_unavailable", "interrupted", "time_limit"}:
        return "execution"
    if prediction.candidate_ids is not None and gold.evidence_groups:
        if not _contains_evidence(gold.evidence_groups, prediction.candidate_ids):
            return "retrieval"
    if prediction.status != "completed":
        return "verification_pending"
    if gold.evidence_groups and prediction.candidate_ids is None and not evidence_hit:
        return "unknown"
    return "judgment"


def _contains_evidence(groups: tuple[frozenset[str], ...], candidates: tuple[str, ...]) -> bool:
    present = set(candidates)
    return any(group <= present for group in groups)


def _parse_predictions(document: object) -> dict[_CaseKey, _Prediction]:
    fields = {"sample_id", "claim_id", "review_status", "verdict", "evidence_ids"}
    predictions = {}
    for key, row, location in _rows(document, "predictions", fields, {"candidate_ids", "claim_anchor", "stop_reason"}):
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
        candidates = None
        if "candidate_ids" in row:
            candidates = _ordered_ids(row["candidate_ids"], f"{location}.candidate_ids")
        anchor = _anchor(row["claim_anchor"], f"{location}.claim_anchor") if "claim_anchor" in row else None
        reason = _identifier(row["stop_reason"], f"{location}.stop_reason") if "stop_reason" in row else None
        predictions[key] = _Prediction(status, verdict, evidence_ids, candidates, anchor, reason)
    return predictions


def _parse_gold(document: object) -> dict[_CaseKey, _GoldLabel]:
    fields = {"sample_id", "claim_id", "verdict", "accepted_evidence_groups"}
    labels = {}
    for key, row, location in _rows(document, "labels", fields, {"claim_anchor", "split"}, {"extraction_samples"}):
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
        anchor = _anchor(row["claim_anchor"], f"{location}.claim_anchor") if "claim_anchor" in row else None
        split = _identifier(row["split"], f"{location}.split") if "split" in row else None
        labels[key] = _GoldLabel(verdict, groups, anchor, split)
    return labels


def _rows(
    document: object, collection: str, fields: set[str], optional_fields: set[str],
    optional_document_fields: set[str] | frozenset[str] = frozenset(),
) -> Iterator[tuple[_CaseKey, dict[str, object], str]]:
    required = {"schema_version", collection}
    if not isinstance(document, dict) or not required <= document.keys() <= required | optional_document_fields:
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
        if not isinstance(row, dict) or not fields <= row.keys() <= fields | optional_fields:
            raise ValueError(f"{location} must contain exactly the required fields {', '.join(sorted(fields))} "
                             f"and may include {', '.join(sorted(optional_fields))}")
        key = (
            _identifier(row["sample_id"], f"{location}.sample_id"),
            _identifier(row["claim_id"], f"{location}.claim_id"),
        )
        if key in seen:
            raise ValueError(f"{location} repeats {key[0]}/{key[1]}")
        seen.add(key)
        yield key, row, location


def _evidence_ids(value: object, location: str) -> frozenset[str]:
    return frozenset(_ordered_ids(value, location))


def _ordered_ids(value: object, location: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    ids = tuple(_identifier(item, location) for item in value)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{location} contains duplicate IDs")
    return ids


def _anchor(value: object, location: str) -> _Anchor:
    required = {"sentence_id", "start", "end", "proposition"}
    if not isinstance(value, dict) or value.keys() != required:
        raise ValueError(f"{location} requires sentence_id, start, end, and proposition")
    start, end = value["start"], value["end"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError(f"{location} has an invalid source interval")
    proposition = _identifier(value["proposition"], f"{location}.proposition")
    normalized = " ".join(unicodedata.normalize("NFKC", proposition).casefold().split())
    return _Anchor(_identifier(value["sentence_id"], f"{location}.sentence_id"), start, end, normalized)


def _extraction_samples(gold: dict, labels: dict[_CaseKey, _GoldLabel]) -> frozenset[str]:
    samples = frozenset(_ordered_ids(gold.get("extraction_samples", []), "extraction_samples"))
    for key, label in labels.items():
        if (key[0] in samples) != (label.anchor is not None):
            raise ValueError(f"{key[0]}/{key[1]}: only extraction samples require gold claim anchors")
    return samples


def _match_extractions(predicted: dict, labels: dict, samples: frozenset[str]) -> dict[_CaseKey, _CaseKey]:
    # Candidate links require the same sample, sentence, and normalized proposition,
    # as well as overlapping original offsets. Labels and evidence never affect the assignment.
    gold_by_sentence = {}
    for gold_key, label in labels.items():
        if label.anchor is not None:
            key = gold_key[0], label.anchor.sentence_id, label.anchor.proposition
            gold_by_sentence.setdefault(key, []).append((gold_key, label.anchor))

    links = {}
    for pred_key, prediction in predicted.items():
        if pred_key[0] not in samples or prediction.anchor is None:
            continue
        anchor = prediction.anchor
        key = pred_key[0], anchor.sentence_id, anchor.proposition
        overlaps = []
        for gold_key, gold_anchor in gold_by_sentence.get(key, ()):
            overlap = min(anchor.end, gold_anchor.end) - max(anchor.start, gold_anchor.start)
            if overlap > 0:
                distance = abs(anchor.start - gold_anchor.start) + abs(anchor.end - gold_anchor.end)
                overlaps.append((distance, gold_key))
        links[pred_key] = [key for _, key in sorted(overlaps)]

    matches = {}

    def assign(pred_key, seen):
        for gold_key in links[pred_key]:
            if gold_key not in seen:
                seen.add(gold_key)
                if gold_key not in matches or assign(matches[gold_key], seen):
                    matches[gold_key] = pred_key
                    return True
        return False

    for pred_key in sorted(links, key=lambda key: (len(links[key]), key)):
        assign(pred_key, set())
    return matches


def _extraction_metrics(predicted: dict, labels: dict, samples: frozenset[str], matches: dict) -> dict:
    gold_keys = {key for key in labels if key[0] in samples}
    pred_keys = {key for key in predicted if key[0] in samples}
    matched = len(matches)
    return {
        "precision": _ratio(matched, len(pred_keys)),
        "recall": _ratio(matched, len(gold_keys)),
        "f1": _ratio(2 * matched, len(pred_keys) + len(gold_keys)),
        "missing": [dict(zip(("sample_id", "claim_id"), key)) for key in sorted(gold_keys - matches.keys())],
        "extra": [dict(zip(("sample_id", "claim_id"), key)) for key in sorted(pred_keys - set(matches.values()))],
    }


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
