import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass


_VERDICTS = ("supported", "refuted", "insufficient")
_REVIEW_STATUSES = ("completed", "pending", "manual_conflict")
_DEFAULT_RANK_CUTOFFS = (1, 3, 5, 8)
_JUDGE_ATTEMPT_STATUSES = {"checked", "model_reference_error", "citation_error", "model_paused", "model_unavailable"}
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
    decision_trace: dict | None


@dataclass(frozen=True)
class _GoldLabel:
    verdict: str
    evidence_groups: tuple[frozenset[str], ...]
    anchor: _Anchor | None
    split: str | None


def score_predictions(
    predictions: object, gold: object, *, rank_cutoffs: tuple[int, ...] = _DEFAULT_RANK_CUTOFFS,
    semantic_matches: object = None, verdict_projections: object = None,
) -> dict[str, object]:
    """Score stored predictions, with optional independent alignment and projection reviews."""
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

    result = _score_aligned(predicted, gold_labels, extraction_samples, matches, cutoffs)
    combined = matches
    if semantic_matches is not None:
        annotations, provenance = _parse_semantic_matches(
            semantic_matches, predictions, gold, predicted, gold_labels, extraction_samples, matches,
        )
        combined = matches | annotations
        result["semantic_extraction"] = {
            **_extraction_metrics(predicted, gold_labels, extraction_samples, combined),
            "matches": [
                {"sample_id": key[0], "gold_claim_id": key[1], "predicted_claim_id": prediction_key[1],
                 "source": "exact" if key in matches else "annotation"}
                for key, prediction_key in sorted(combined.items())
            ],
            "provenance": provenance,
        }
        result["semantic_scoring"] = _score_aligned(predicted, gold_labels, extraction_samples, combined, cutoffs)
    if verdict_projections is not None:
        result["verdict_projection"] = _review_verdict_projections(
            verdict_projections, predictions, gold, predicted, gold_labels, extraction_samples, combined,
        )
    return result


def _score_aligned(
    predicted: dict[_CaseKey, _Prediction], gold_labels: dict[_CaseKey, _GoldLabel],
    extraction_samples: frozenset[str], matches: dict[_CaseKey, _CaseKey], cutoffs: tuple[int, ...],
) -> dict[str, object]:
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
        if prediction is not None and prediction.decision_trace is not None:
            row["decision_trace"] = prediction.decision_trace
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
    if prediction.stop_reason == "invalid_judge_reference":
        return "citation_validation"
    if prediction.candidate_ids is not None and gold.evidence_groups:
        if not _contains_evidence(gold.evidence_groups, prediction.candidate_ids):
            return "retrieval"
    if prediction.status != "completed":
        if prediction.decision_trace:
            if prediction.decision_trace["blocker_codes"]:
                return "verdict_blocked"
            if prediction.decision_trace["attempts"][-1]["proposed_verdict"] == "pending":
                return "model_pending"
        return "verification_pending"
    if gold.evidence_groups and prediction.candidate_ids is None and not evidence_hit:
        return "unknown"
    return "judgment"


def _contains_evidence(groups: tuple[frozenset[str], ...], candidates: tuple[str, ...]) -> bool:
    present = set(candidates)
    return any(group <= present for group in groups)


def _decision_trace(value: object, location: str) -> dict:
    if not isinstance(value, dict) or value.keys() != {"attempts", "blocker_codes", "missing_requirements"}:
        raise ValueError(f"{location} must contain the structured decision trace")
    attempts = value["attempts"]
    if not isinstance(attempts, list) or not attempts:
        raise ValueError(f"{location}.attempts must be a nonempty list")
    for attempt in attempts:
        if not isinstance(attempt, dict) or attempt.keys() != {"material_sha256", "repair", "proposed_verdict", "status"}:
            raise ValueError(f"{location}.attempts contains an invalid decision attempt")
        digest, proposed, status = attempt["material_sha256"], attempt["proposed_verdict"], attempt["status"]
        if (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or type(attempt["repair"]) is not bool or not isinstance(status, str) or status not in _JUDGE_ATTEMPT_STATUSES
                or proposed not in (None, *_VERDICTS, "pending", "manual_conflict")
                or (proposed is not None) != (status in {"checked", "citation_error"})):
            raise ValueError(f"{location}.attempts contains an invalid decision attempt")
    _ordered_ids(value["blocker_codes"], f"{location}.blocker_codes")
    _ordered_ids(value["missing_requirements"], f"{location}.missing_requirements")
    return value


def _parse_predictions(document: object) -> dict[_CaseKey, _Prediction]:
    fields = {"sample_id", "claim_id", "review_status", "verdict", "evidence_ids"}
    predictions = {}
    for key, row, location in _rows(document, "predictions", fields,
                                   {"candidate_ids", "claim_anchor", "stop_reason", "decision_trace"}):
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
        trace = _decision_trace(row["decision_trace"], f"{location}.decision_trace") if "decision_trace" in row else None
        predictions[key] = _Prediction(status, verdict, evidence_ids, candidates, anchor, reason, trace)
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


def semantic_manifest_sha256(document: object) -> str:
    """Fingerprint parsed JSON using sorted keys, UTF-8, and compact separators."""
    canonical = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _review_provenance(document: dict, predictions: object, gold: object, description: str) -> dict[str, str]:
    annotation_id = _identifier(document["annotation_id"], "annotation_id")
    annotator_type = document["annotator_type"]
    if annotator_type not in ("human", "ai"):
        raise ValueError("annotator_type must be human or ai")
    provenance = {
        "annotation_id": annotation_id,
        "annotator_type": annotator_type,
        "predictions_sha256": semantic_manifest_sha256(predictions),
        "gold_sha256": semantic_manifest_sha256(gold),
        "annotation_sha256": semantic_manifest_sha256(document),
    }
    for field in ("predictions_sha256", "gold_sha256"):
        if document[field] != provenance[field]:
            raise ValueError(f"{description} {field} does not match the supplied manifest")
    return provenance


def _parse_semantic_matches(
    document: object, predictions: object, gold: object, predicted: dict[_CaseKey, _Prediction],
    labels: dict[_CaseKey, _GoldLabel], samples: frozenset[str], exact: dict[_CaseKey, _CaseKey],
) -> tuple[dict[_CaseKey, _CaseKey], dict[str, str]]:
    fields = {"schema_version", "annotation_id", "annotator_type", "predictions_sha256", "gold_sha256", "matches"}
    if not isinstance(document, dict) or document.keys() != fields:
        raise ValueError(f"Semantic matches must contain exactly {', '.join(sorted(fields))}")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("Unsupported semantic matches schema_version")
    if not samples:
        raise ValueError("Semantic matches require gold extraction samples")
    provenance = _review_provenance(document, predictions, gold, "Semantic matches")
    rows = document["matches"]
    if not isinstance(rows, list):
        raise ValueError("Semantic matches must be a list")

    exact_by_prediction = {prediction_key: gold_key for gold_key, prediction_key in exact.items()}
    matches = {}
    used_predictions: set[_CaseKey] = set()
    for index, row in enumerate(rows):
        location = f"matches[{index}]"
        if not isinstance(row, dict) or row.keys() != {"sample_id", "predicted_claim_id", "gold_claim_id"}:
            raise ValueError(f"{location} must contain exactly sample_id, predicted_claim_id, and gold_claim_id")
        sample = _identifier(row["sample_id"], f"{location}.sample_id")
        gold_key = sample, _identifier(row["gold_claim_id"], f"{location}.gold_claim_id")
        prediction_key = sample, _identifier(row["predicted_claim_id"], f"{location}.predicted_claim_id")
        if sample not in samples or gold_key not in labels or prediction_key not in predicted:
            raise ValueError(f"{location} must reference an existing prediction and gold claim in one extraction sample")
        if gold_key in matches or prediction_key in used_predictions:
            raise ValueError(f"{location} violates one-to-one matching by repeating a prediction or gold claim")
        gold_anchor, prediction_anchor = labels[gold_key].anchor, predicted[prediction_key].anchor
        if gold_anchor is None or prediction_anchor is None or not _same_source_span(gold_anchor, prediction_anchor):
            raise ValueError(f"{location} must link overlapping source intervals in the same sentence")
        if (exact.get(gold_key, prediction_key) != prediction_key
                or exact_by_prediction.get(prediction_key, gold_key) != gold_key):
            raise ValueError(f"{location} conflicts with an existing exact match")
        matches[gold_key] = prediction_key
        used_predictions.add(prediction_key)
    return matches, provenance


def _review_verdict_projections(
    document: object, predictions: object, gold: object, predicted: dict[_CaseKey, _Prediction],
    labels: dict[_CaseKey, _GoldLabel], samples: frozenset[str], aligned: dict[_CaseKey, _CaseKey],
) -> dict[str, object]:
    """Validate evidence and provenance for each independently reviewed conjunct."""
    fields = {"schema_version", "annotation_id", "annotator_type", "predictions_sha256", "gold_sha256", "projections"}
    if not isinstance(document, dict) or document.keys() != fields:
        raise ValueError(f"Verdict projections must contain exactly {', '.join(sorted(fields))}")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("Unsupported verdict projections schema_version")
    if not samples:
        raise ValueError("Verdict projections require gold extraction samples")
    provenance = _review_provenance(document, predictions, gold, "Verdict projections")
    rows = document["projections"]
    if not isinstance(rows, list):
        raise ValueError("Verdict projections must be a list")

    row_fields = {"sample_id", "predicted_claim_id", "gold_claim_id", "relation", "operand_anchor",
                  "evidence_ids", "review_note"}
    used_predictions = set(aligned.values())
    used_labels = set(aligned)
    reviewed = []
    for index, row in enumerate(rows):
        location = f"projections[{index}]"
        if not isinstance(row, dict) or row.keys() != row_fields:
            raise ValueError(f"{location} must contain exactly {', '.join(sorted(row_fields))}")
        sample = _identifier(row["sample_id"], f"{location}.sample_id")
        gold_key = sample, _identifier(row["gold_claim_id"], f"{location}.gold_claim_id")
        prediction_key = sample, _identifier(row["predicted_claim_id"], f"{location}.predicted_claim_id")
        if sample not in samples or gold_key not in labels or prediction_key not in predicted:
            raise ValueError(f"{location} must reference an existing prediction and gold claim in one extraction sample")
        if gold_key in used_labels or prediction_key in used_predictions:
            raise ValueError(f"{location} conflicts with an existing alignment or projection")
        if row["relation"] != "conjunct":
            raise ValueError(f"{location}.relation must be conjunct")
        note = row["review_note"]
        if not isinstance(note, str) or not note.strip():
            raise ValueError(f"{location}.review_note must explain why the evidence refutes this operand")

        prediction, label = predicted[prediction_key], labels[gold_key]
        source = prediction.anchor
        operand = _anchor(row["operand_anchor"], f"{location}.operand_anchor")
        if operand != label.anchor:
            raise ValueError(f"{location}.operand_anchor must identify the exact gold operand")
        if (source is None or source.sentence_id != operand.sentence_id
                or not source.start <= operand.start < operand.end <= source.end
                or (source.start, source.end) == (operand.start, operand.end)
                or source.proposition == operand.proposition):
            raise ValueError(f"{location} must link a distinct, strictly contained operand in the same sentence")
        if prediction.status != "completed" or prediction.verdict != "refuted" or label.verdict != "refuted":
            raise ValueError(f"{location} requires a completed refutation and a refuted gold operand")

        ids = _evidence_ids(row["evidence_ids"], f"{location}.evidence_ids")
        if (not ids or prediction.candidate_ids is None or not ids <= prediction.evidence_ids
                or not ids <= set(prediction.candidate_ids)):
            raise ValueError(f"{location}.evidence_ids must be both cited and present in the prediction's candidates")
        if not any(group <= ids for group in label.evidence_groups):
            raise ValueError(f"{location}.evidence_ids must include a complete accepted evidence group for this operand")
        used_labels.add(gold_key)
        used_predictions.add(prediction_key)
        reviewed.append({**row, "operand_anchor": dict(row["operand_anchor"]), "evidence_ids": sorted(ids)})

    reviewed.sort(key=lambda row: (row["sample_id"], row["gold_claim_id"], row["predicted_claim_id"]))
    return {"provenance": provenance, "credited_count": len(reviewed), "projections": reviewed}


def _same_source_span(left: _Anchor, right: _Anchor) -> bool:
    return left.sentence_id == right.sentence_id and max(left.start, right.start) < min(left.end, right.end)


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
            if _same_source_span(anchor, gold_anchor):
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
