"""Input-only PDF benchmark runs and separately validated classification summaries."""

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .review import PdfReviewInput
from .scoring import score_predictions


_LABELS = ("supported", "refuted", "insufficient")
_SAMPLE_FIELDS = {"sample_id", "pdf", "claims", "abstract_file", "paper_family_id", "split"}


@dataclass(frozen=True, slots=True)
class _Sample:
    sample_id: str
    pdf: Path
    pasted_abstract: str | None
    abstract_digest: str | None
    claims: tuple[tuple[str, str], ...]


def _identifier(value: object, location: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{location} must be a nonempty string without surrounding whitespace")
    return value


def _source_path(value: object, base: Path, location: str) -> Path:
    path = (base / _identifier(value, location)).resolve()
    if not path.is_file():
        raise ValueError(f"{location} is not a local file: {path}")
    return path


def _claims(value: object, location: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a nonempty list when specified")
    claims = {}
    for number, row in enumerate(value):
        item = f"{location}[{number}]"
        if not isinstance(row, dict) or set(row) != {"claim_id", "text"}:
            raise ValueError(f"{item} must contain only claim_id and text")
        claim_id = _identifier(row["claim_id"], f"{item}.claim_id")
        if claim_id in claims:
            raise ValueError(f"{item} repeats claim_id {claim_id!r}")
        text = row["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{item}.text must be a nonempty string")
        claims[claim_id] = text
    return tuple(claims.items())


def _check_split(key: str | Path, split: str, seen: dict, location: str) -> None:
    previous = seen.setdefault(key, split)
    if previous != split:
        raise ValueError(f"{location}: {key!s} appears in both {previous!r} and {split!r} splits")


def _load_samples(path: Path) -> list[_Sample]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid UTF-8 JSON ({exc})") from exc
    if (not isinstance(data, dict) or set(data) != {"schema_version", "samples"}
            or type(data["schema_version"]) is not int or data["schema_version"] != 1
            or not isinstance(data["samples"], list) or not data["samples"]):
        raise ValueError("PDF benchmark input requires schema_version 1 and a nonempty samples list")

    samples, seen, family_splits, source_splits = [], set(), {}, {}
    base = path.parent
    for number, row in enumerate(data["samples"]):
        location = f"samples[{number}]"
        if not isinstance(row, dict) or not {"sample_id", "pdf"} <= row.keys() <= _SAMPLE_FIELDS:
            raise ValueError(f"{location} requires sample_id and pdf and may contain only input fields")
        sample_id = _identifier(row["sample_id"], f"{location}.sample_id")
        if sample_id in seen:
            raise ValueError(f"{location} repeats sample_id {sample_id!r}")
        seen.add(sample_id)
        pdf = _source_path(row["pdf"], base, f"{location}.pdf")
        family = _identifier(row["paper_family_id"], f"{location}.paper_family_id") if "paper_family_id" in row else None
        if "split" in row:
            split = _identifier(row["split"], f"{location}.split")
            if family is None:
                raise ValueError(f"{location}.split requires paper_family_id")
            _check_split(family, split, family_splits, location)
            _check_split(pdf, split, source_splits, location)

        abstract = abstract_digest = None
        if "abstract_file" in row:
            abstract_path = _source_path(row["abstract_file"], base, f"{location}.abstract_file")
            if abstract_path in {path.resolve(), pdf}:
                raise ValueError(f"{location}.abstract_file must be separate from the manifest and PDF")
            try:
                raw_abstract = abstract_path.read_bytes()
                abstract = raw_abstract.decode("utf-8-sig")
            except UnicodeError as exc:
                raise ValueError(f"{location}.abstract_file is not UTF-8") from exc
            if not abstract.strip():
                raise ValueError(f"{location}.abstract_file is empty")
            abstract_digest = hashlib.sha256(raw_abstract).hexdigest()
        claims = _claims(row["claims"], f"{location}.claims") if "claims" in row else ()
        samples.append(_Sample(sample_id, pdf, abstract, abstract_digest, claims))
    return samples


def _claim_ids(sample: _Sample, report: dict, findings: Iterable[dict]) -> list[str]:
    if sample.claims:
        return [claim_id for claim_id, _ in sample.claims]
    return list(dict.fromkeys(row["claim_id"] for row in [*report.get("claims", []), *findings]))


def _prediction(sample_id: str, claim_id: str, finding: dict) -> dict:
    status, verdict = finding.get("review_status"), finding.get("verdict")
    if status == "completed" and verdict in _LABELS:
        evidence = finding.get("evidence_ids", [])
    elif status in ("pending", "manual_conflict"):
        verdict = None
        evidence = finding.get("evidence_ids", [])
    else:
        status, verdict, evidence = "pending", None, []
    return {"sample_id": sample_id, "claim_id": claim_id, "review_status": status,
            "verdict": verdict, "evidence_ids": list(dict.fromkeys(evidence))}


def _input_digest(sample: _Sample, engine: str, model: str) -> tuple[str, str]:
    from .ingest import PARSER_VERSION
    from .models import ROLE_PROMPTS

    with sample.pdf.open("rb") as source:
        pdf_hash = hashlib.file_digest(source, "sha256").hexdigest()
    material = [1, PARSER_VERSION, ROLE_PROMPTS, sample.sample_id, pdf_hash, sample.abstract_digest,
                sample.claims, engine, model]
    digest = hashlib.sha256(json.dumps(material, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return digest, pdf_hash


def _checkpoint_rows(sample: _Sample, input_digest: str, entry: object) -> list[dict] | None:
    if not isinstance(entry, dict) or entry.get("input_digest") != input_digest:
        return None
    rows = entry.get("predictions")
    if not isinstance(rows, list):
        return None
    claim_ids = []
    for row in rows:
        if not isinstance(row, dict) or row.get("sample_id") != sample.sample_id:
            return None
        claim_id = row.get("claim_id")
        evidence = row.get("evidence_ids")
        if not isinstance(claim_id, str) or not claim_id.strip() or not isinstance(evidence, list):
            return None
        if not all(isinstance(item, str) and item.strip() for item in evidence):
            return None
        if _prediction(sample.sample_id, claim_id, row) != row:
            return None
        claim_ids.append(claim_id)
    if len(set(claim_ids)) != len(claim_ids):
        return None
    if sample.claims and claim_ids != [claim_id for claim_id, _ in sample.claims]:
        return None
    return rows


def run_pdf_samples(
    service, manifest_path: Path, engine: str, model: str, *,
    recovered: dict | None = None, on_sample: Callable[[str, str, list[dict]], None] | None = None,
) -> list[dict]:
    """Run input rows through the normal review service without reading gold."""
    if engine not in {"offline", "openai"}:
        raise ValueError("PDF benchmark engine must be offline or openai")
    samples = _load_samples(manifest_path)
    checkpoints = recovered or {}
    predictions = []
    for sample in samples:
        try:
            digest, pdf_hash = _input_digest(sample, engine, model)
        except OSError as exc:
            raise ValueError(f"benchmark sample {sample.sample_id!r}: {exc}") from exc
        cached = _checkpoint_rows(sample, digest, checkpoints.get(sample.sample_id))
        if cached is not None:
            predictions.extend(cached)
            continue
        request = PdfReviewInput(
            pdf=sample.pdf, pasted_abstract=sample.pasted_abstract,
            abstract_source="pasted" if sample.pasted_abstract is not None else "pdf",
            engine=engine, model=model, provided_claims=sample.claims,
        )
        try:
            run = service.start(request)
            report = service.export(run["run_id"], run["report_version_id"])
        except (ValueError, OSError) as exc:
            raise ValueError(f"benchmark sample {sample.sample_id!r}: {exc}") from exc

        documents = report.get("documents", [])
        if documents and documents[0]["sha256"] != pdf_hash:
            raise ValueError(f"benchmark sample {sample.sample_id!r}: PDF changed during review")
        findings = {item["claim_id"]: item for item in report.get("findings", [])}
        rows = [_prediction(sample.sample_id, claim_id, findings.get(claim_id, {}))
                for claim_id in _claim_ids(sample, report, findings.values())]
        predictions.extend(rows)
        if on_sample is not None and run.get("run_status") == "report_ready":
            on_sample(sample.sample_id, digest, rows)
    return predictions


def _ratio(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def summarize_classification(predictions: object, gold: object) -> dict:
    """Add completed-only scores and count false support against all unsupported gold."""
    scored = score_predictions(predictions, gold)
    confusion = {actual: {predicted: 0 for predicted in _LABELS} for actual in _LABELS}
    correct = completed = unsupported = false_support = 0
    for case in scored["cases"]:
        actual, predicted = case["gold_verdict"], case["predicted_verdict"]
        if actual != "supported":
            unsupported += 1
            false_support += int(predicted == "supported" and case["review_status"] == "completed")
        if case["review_status"] == "completed":
            confusion[actual][predicted] += 1
            correct += int(actual == predicted)
            completed += 1

    class_f1 = []
    for label in _LABELS:
        denominator = sum(confusion[label].values()) + sum(confusion[row][label] for row in _LABELS)
        class_f1.append(2 * confusion[label][label] / denominator if denominator else 0.0)
    return {"completed_accuracy": _ratio(correct, completed), "confusion_matrix": confusion,
            "macro_f1": sum(class_f1) / len(_LABELS) if completed else None,
            "unsupported_as_support": _ratio(false_support, unsupported)}
