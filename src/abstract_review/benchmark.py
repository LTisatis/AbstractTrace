"""Input-only PDF benchmark runs and separately validated classification summaries."""

import hashlib
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .review import PdfReviewInput
from .scoring import score_predictions


_LABELS = ("supported", "refuted", "insufficient")
_SAMPLE_FIELDS = {"sample_id", "pdf", "pdf_sha256", "claims", "abstract_file", "paper_family_id", "split"}
_ANCHOR_FIELDS = ("sentence_id", "start", "end", "proposition")
_BENCHMARK_PIPELINE_FILES = ("audit.py", "benchmark.py", "calculations.py", "compound.py", "evidence.py",
                             "followups.py", "ingest.py", "models.py", "review.py", "revisions.py",
                             "run_store.py", "scifact.py", "scitabalign.py", "workflow.py")


@dataclass(frozen=True, slots=True)
class _Sample:
    sample_id: str
    pdf: Path
    pasted_abstract: str | None
    abstract_digest: str | None
    claims: tuple[tuple[str, str], ...]
    paper_family_id: str | None
    split: str | None
    pdf_sha256: str | None


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
        pdf_sha256 = None
        if "pdf_sha256" in row:
            pdf_sha256 = _identifier(row["pdf_sha256"], f"{location}.pdf_sha256")
            if len(pdf_sha256) != 64 or any(character not in "0123456789abcdef" for character in pdf_sha256):
                raise ValueError(f"{location}.pdf_sha256 must be a lowercase SHA-256 digest")
        family = _identifier(row["paper_family_id"], f"{location}.paper_family_id") if "paper_family_id" in row else None
        split = None
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
        samples.append(_Sample(sample_id, pdf, abstract, abstract_digest, claims, family, split, pdf_sha256))
    return samples


def _claim_ids(sample: _Sample, report: dict, findings: Iterable[dict]) -> list[str]:
    if sample.claims:
        return [claim_id for claim_id, _ in sample.claims]
    return list(dict.fromkeys(row["claim_id"] for row in [*report.get("claims", []), *findings]))


def _prediction(sample_id: str, claim_id: str, finding: dict, claim: dict | None = None) -> dict:
    status, verdict = finding.get("review_status"), finding.get("verdict")
    if status == "completed" and verdict in _LABELS:
        evidence = finding.get("evidence_ids", [])
    elif status in ("pending", "manual_conflict"):
        verdict = None
        evidence = finding.get("evidence_ids", [])
    else:
        status, verdict, evidence = "pending", None, []
    row = {"sample_id": sample_id, "claim_id": claim_id, "review_status": status,
           "verdict": verdict, "evidence_ids": list(dict.fromkeys(evidence))}
    candidates = finding.get("sent_candidate_ids", finding.get("candidate_ids"))
    if candidates is not None:
        row["candidate_ids"] = list(dict.fromkeys(candidates))
    if claim is not None and all(field in claim for field in _ANCHOR_FIELDS):
        row["claim_anchor"] = {key: claim[key] for key in _ANCHOR_FIELDS}
    if finding.get("stop_reason"):
        row["stop_reason"] = finding["stop_reason"]
    return row


def benchmark_configuration(engine: str, model: str) -> dict:
    from .ingest import PARSER_VERSION
    from .models import ROLE_PROMPTS

    source_dir = Path(__file__).parent
    pipeline = hashlib.sha256()
    for name in _BENCHMARK_PIPELINE_FILES:
        source = source_dir / name
        if not source.is_file():
            continue
        pipeline.update(name.encode("ascii"))
        pipeline.update(hashlib.sha256(source.read_bytes()).digest())
    prompts = json.dumps(ROLE_PROMPTS, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {
        "engine": engine, "model": model, "parser_version": PARSER_VERSION,
        "prompts_sha256": hashlib.sha256(prompts).hexdigest(), "pipeline_sha256": pipeline.hexdigest(),
    }


def _input_digest(sample: _Sample, configuration: dict) -> tuple[str, str]:
    with sample.pdf.open("rb") as source:
        pdf_hash = hashlib.file_digest(source, "sha256").hexdigest()
    if sample.pdf_sha256 is not None and pdf_hash != sample.pdf_sha256:
        raise ValueError(f"benchmark sample {sample.sample_id!r}: PDF does not match the manifest SHA-256")
    material = [1, configuration, sample.sample_id, pdf_hash, sample.abstract_digest,
                sample.claims, sample.paper_family_id, sample.split]
    encoded = json.dumps(material, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
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
        candidates = row.get("candidate_ids", [])
        if not isinstance(candidates, list) or not all(isinstance(item, str) and item.strip() for item in candidates):
            return None
        anchor = row.get("claim_anchor")
        if anchor is not None and (
            not isinstance(anchor, dict) or anchor.keys() != set(_ANCHOR_FIELDS)
            or not isinstance(anchor["sentence_id"], str) or not anchor["sentence_id"].strip()
            or not isinstance(anchor["proposition"], str) or not anchor["proposition"].strip()
            or type(anchor["start"]) is not int or type(anchor["end"]) is not int
            or not 0 <= anchor["start"] < anchor["end"]
        ):
            return None
        if "stop_reason" in row and (not isinstance(row["stop_reason"], str) or not row["stop_reason"].strip()):
            return None
        if _prediction(sample.sample_id, claim_id, row, anchor) != row:
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
    on_result: Callable[[dict], None] | None = None, configuration: dict | None = None,
) -> list[dict]:
    """Run input rows through the normal review service without reading gold."""
    if engine not in {"offline", "openai"}:
        raise ValueError("PDF benchmark engine must be offline or openai")
    samples = _load_samples(manifest_path)
    configuration = configuration or benchmark_configuration(engine, model)
    checkpoints = recovered or {}
    predictions = []
    for sample in samples:
        started = time.perf_counter()
        try:
            digest, pdf_hash = _input_digest(sample, configuration)
        except OSError as exc:
            raise ValueError(f"benchmark sample {sample.sample_id!r}: {exc}") from exc
        result = {"sample_id": sample.sample_id, "paper_family_id": sample.paper_family_id,
                  "split": sample.split, "pdf_sha256": pdf_hash, "input_sha256": digest}
        entry = checkpoints.get(sample.sample_id)
        cached = _checkpoint_rows(sample, digest, entry)
        if cached is not None:
            predictions.extend(cached)
            if on_result is not None:
                saved_result = entry.get("result")
                metadata = saved_result if isinstance(saved_result, dict) and all(
                    saved_result.get(field) == value for field, value in result.items()
                ) and saved_result.get("prediction_count") == len(cached) else {}
                on_result({**metadata, **result, "cache_hit": True, "prediction_count": len(cached)})
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
        extracted = {item["claim_id"]: item for item in report.get("claims", [])} if not sample.claims else {}
        rows = [_prediction(sample.sample_id, claim_id, findings.get(claim_id, {}), extracted.get(claim_id))
                for claim_id in _claim_ids(sample, report, findings.values())]
        predictions.extend(rows)
        if on_result is not None:
            on_result(_sample_result(result, run, report, rows, time.perf_counter() - started))
        if on_sample is not None and run.get("run_status") == "report_ready":
            on_sample(sample.sample_id, digest, rows)
    return predictions


def _sample_result(result: dict, run: dict, report: dict, rows: list[dict], elapsed: float) -> dict:
    usage = report.get("usage", {})
    events = usage.get("model_events", [])
    tokens = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        values = [entry[key] for entry in events if type(entry.get(key)) is int]
        tokens[key] = sum(values) if values else None
    request_usage = []
    for event in events:
        input_details = event.get("input_tokens_details") or {}
        write_details = input_details.get("cache_write_tokens_details") or {}
        request_usage.append({key: event.get(key) for key in ("stage", "model", "status", "input_tokens", "output_tokens")}
                             | {
                                 "cached_input_tokens": input_details.get("cached_tokens"),
                                 "cache_write_input_tokens": input_details.get("cache_write_tokens"),
                                 "cache_write_30m_input_tokens": write_details.get("ttl_30m_tokens"),
                                 "cache_write_12h_input_tokens": write_details.get("ttl_12h_tokens"),
                             })
    summary = {
        **result, "cache_hit": False, "run_id": run["run_id"], "run_status": run.get("run_status"),
        "prediction_count": len(rows), "wall_seconds": round(elapsed, 6),
        "model_requests": usage.get("external_requests"), "tokens": tokens, "request_usage": request_usage,
        "follow_up_count": sum(bool(row.get("follow_up_performed")) for row in report.get("findings", [])),
        "follow_up_new_evidence": sum(len(row.get("follow_up_evidence_ids", ())) for row in report.get("findings", [])),
        "unreadable_pages": sum(page.get("status") == "unreadable" for document in report.get("documents", [])
                                for page in document.get("pages", [])),
        "revision_count": len(report.get("revisions", [])),
        "abstract_verification_status": report.get("abstract_review", {}).get("status"),
    }
    actual = report.get("reviewed_abstract")
    review = report.get("abstract_review")
    version_id = actual.get("version_id") if actual else None
    if version_id and review and version_id == review.get("version_id") == run.get("report_version_id"):
        calculations = report.get("calculations", [])
        scope_ids = {eid for row in calculations for eid in [*row.get("evidence_ids", ()),
                                                               *row.get("scope_evidence_ids", ())]}
        summary["review"] = {
            "version_id": version_id,
            "original_text": report["original_abstract"]["text"], "reviewed_text": actual["text"],
            "revision_ids": actual["revision_ids"],
            "revisions": report.get("revisions", []), "abstract_review": review,
            "calculations": calculations,
            "evidence_roles": {item["evidence_id"]: item.get("cell_role") or item["kind"] for item in report["evidence"]
                               if item["evidence_id"] in scope_ids},
        }
    return summary


def summarize_classification(predictions: object, gold: object) -> dict:
    """Add completed-only scores and count false support against all unsupported gold."""
    scored = score_predictions(predictions, gold)
    return {key: scored[key] for key in ("completed_accuracy", "confusion_matrix", "macro_f1", "unsupported_as_support")}
