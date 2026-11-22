"""Synthetic review input, deterministic model, and persisted JSON reports."""

import hashlib
import json
import sqlite3
import unicodedata
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, Protocol
from uuid import uuid4


Verdict = Literal["supported", "refuted", "insufficient"]
ReviewStatus = Literal["completed", "pending", "manual_conflict"]


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    text: str
    subject: str
    metric: str
    value: str


@dataclass(frozen=True, slots=True)
class Locator:
    section: str
    page_index: int


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    text: str
    subject: str
    metric: str
    value: str
    locator: Locator


@dataclass(frozen=True, slots=True)
class ReviewInput:
    sample_id: str
    abstract: str
    claims: tuple[Claim, ...]
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True, slots=True)
class PdfReviewInput:
    pdf: Path
    supplements: tuple[Path, ...] = ()
    pasted_abstract: str | None = None
    abstract_source: Literal["pdf", "pasted"] = "pdf"


def _object(value: object, keys: set[str], path: str) -> dict[str, object]:
    if not isinstance(value, dict) or value.keys() != keys:
        raise ValueError(f"{path} must contain exactly these fields: {', '.join(sorted(keys))}")
    return value


def _items(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _identifier(value: object, path: str) -> str:
    text = _text(value, path)
    if text != text.strip():
        raise ValueError(f"{path} must not have surrounding whitespace")
    return text


def _parse_locator(value: object, path: str) -> Locator:
    fields = _object(value, {"section", "page_index"}, path)
    page_index = fields["page_index"]
    if type(page_index) is not int or page_index < 0:
        raise ValueError(f"{path}.page_index must be a non-negative integer")
    return Locator(section=_text(fields["section"], f"{path}.section"), page_index=page_index)


def parse_sample(raw: object) -> ReviewInput:
    """Parse only inputs available to the model, rejecting extra fields at every level."""
    fields = _object(raw, {"sample_id", "abstract", "claims", "evidence"}, "sample")
    sample_id = _identifier(fields["sample_id"], "sample.sample_id")
    abstract = _text(fields["abstract"], "sample.abstract")
    fact_keys = {"text", "subject", "metric", "value"}
    claims = []
    for index, item in enumerate(_items(fields["claims"], "sample.claims")):
        path = f"sample.claims[{index}]"
        source = _object(item, fact_keys | {"claim_id"}, path)
        claim = Claim(
            claim_id=_identifier(source["claim_id"], f"{path}.claim_id"),
            **{key: _text(source[key], f"{path}.{key}") for key in fact_keys},
        )
        if claim.text not in abstract:
            raise ValueError(f"{path}.text must appear in sample.abstract")
        claims.append(claim)
    if not claims:
        raise ValueError("sample.claims must contain at least one claim")
    if len({claim.claim_id for claim in claims}) != len(claims):
        raise ValueError("sample.claims contains duplicate claim_id values")

    evidence = []
    for index, item in enumerate(_items(fields["evidence"], "sample.evidence")):
        path = f"sample.evidence[{index}]"
        source = _object(item, fact_keys | {"evidence_id", "locator"}, path)
        locator = _parse_locator(source["locator"], f"{path}.locator")
        evidence.append(
            Evidence(
                evidence_id=_identifier(source["evidence_id"], f"{path}.evidence_id"),
                **{key: _text(source[key], f"{path}.{key}") for key in fact_keys},
                locator=locator,
            )
        )
    if len({item.evidence_id for item in evidence}) != len(evidence):
        raise ValueError("sample.evidence contains duplicate evidence_id values")
    return ReviewInput(sample_id, abstract, tuple(claims), tuple(evidence))


@dataclass(frozen=True, slots=True)
class EvidenceGroup:
    role: Literal["support", "refute"]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Decision:
    review_status: ReviewStatus
    verdict: Verdict | None
    reason: str
    evidence_groups: tuple[EvidenceGroup, ...] = ()


class ReviewModel(Protocol):
    def judge(self, claim: Claim, evidence: tuple[Evidence, ...]) -> Decision: ...


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class FakeModel:
    """Compare synthetic subject/metric/value triples without any external inputs."""

    def judge(self, claim: Claim, evidence: tuple[Evidence, ...]) -> Decision:
        subject, metric, expected = map(_normalize, (claim.subject, claim.metric, claim.value))
        support = []
        refute = []
        observed = set()
        for item in evidence:
            if (_normalize(item.subject), _normalize(item.metric)) != (subject, metric):
                continue
            value = _normalize(item.value)
            observed.add(value)
            if value == expected:
                support.append(EvidenceGroup("support", (item.evidence_id,)))
            else:
                refute.append(EvidenceGroup("refute", (item.evidence_id,)))
        if len(observed) > 1:
            return Decision(
                "manual_conflict", None, "Synthetic evidence has conflicting values.", tuple(support + refute)
            )
        if support:
            return Decision(
                "completed", "supported", "Synthetic evidence has the same value.", tuple(support)
            )
        if refute:
            return Decision(
                "completed", "refuted", "Synthetic evidence has a different value.", tuple(refute)
            )
        return Decision(
            "completed", "insufficient", "No synthetic evidence matches the subject and metric."
        )


class ReviewService:
    """Run the synthetic fixture path and store each immutable report in SQLite."""

    def __init__(self, database: Path, model: ReviewModel | None = None) -> None:
        self.database = database
        self.model = model if model is not None else FakeModel()

    def start(self, review_input: ReviewInput | PdfReviewInput) -> dict:
        if isinstance(review_input, PdfReviewInput):
            return self._start_pdf(review_input)
        run_id = uuid4().hex
        abstract = review_input.abstract
        version_key = json.dumps((review_input.sample_id, abstract), ensure_ascii=False)
        version_id = hashlib.sha256(version_key.encode("utf-8")).hexdigest()
        version = {"version_id": version_id, "text": abstract, "revision_ids": []}
        allowed_ids = {item.evidence_id for item in review_input.evidence}
        findings = []
        unresolved = []

        for claim in review_input.claims:
            decision = self.model.judge(claim, review_input.evidence)
            ids = list(
                dict.fromkeys(eid for group in decision.evidence_groups for eid in group.evidence_ids)
            )
            if not allowed_ids.issuperset(ids):
                raise ValueError(f"model referenced evidence outside sample {review_input.sample_id}")
            if decision.review_status == "completed":
                if decision.verdict not in {"supported", "refuted", "insufficient"}:
                    raise ValueError("completed finding requires a verdict")
            elif (
                decision.review_status not in {"pending", "manual_conflict"}
                or decision.verdict is not None
            ):
                raise ValueError("unresolved finding must have a valid status and no verdict")
            offset = abstract.index(claim.text)
            finding = {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "start": offset,
                "end": offset + len(claim.text),
                "review_status": decision.review_status,
                "verdict": decision.verdict,
                "reason": decision.reason,
                "evidence_ids": ids,
                "evidence_groups": [asdict(group) for group in decision.evidence_groups],
            }
            findings.append(finding)
            if decision.review_status != "completed" or decision.verdict == "insufficient":
                unresolved.append({"claim_id": claim.claim_id, "reason": decision.reason})

        report = {
            "schema_version": 1,
            "mode": "synthetic",
            "sample_id": review_input.sample_id,
            "run_id": run_id,
            "report_version_id": version_id,
            "original_abstract": version,
            "reviewed_abstract": version,
            "findings": findings,
            "revisions": [],
            "verifications": [],
            "abstract_review": {"version_id": version_id, "status": "pending"},
            "evidence": [asdict(item) for item in review_input.evidence],
            "calculations": [],
            "omissions": [],
            "external_checks": [],
            "unresolved_items": unresolved,
            "usage": {"external_requests": 0, "local_model_calls": len(review_input.claims)},
        }
        return self._persist_report(report)

    def _start_pdf(self, request: PdfReviewInput) -> dict:
        from .ingest import parse_pdf

        if request.pasted_abstract is not None and not request.pasted_abstract.strip():
            raise ValueError("--abstract-file is empty")
        if request.abstract_source == "pasted" and request.pasted_abstract is None:
            raise ValueError("--abstract-source pasted requires --abstract-file")
        primary = parse_pdf(request.pdf)
        versions = {}
        for source, abstract in (("pdf", primary.abstract), ("pasted", request.pasted_abstract)):
            if abstract is None:
                continue
            version_key = json.dumps((primary.document["sha256"], source, abstract), ensure_ascii=False)
            versions[source] = {
                "version_id": hashlib.sha256(version_key.encode("utf-8")).hexdigest(),
                "kind": f"{source}_original",
                "text": abstract,
                "revision_ids": [],
            }
        if request.abstract_source not in versions:
            raise ValueError("could not identify a PDF abstract; supply --abstract-file and --abstract-source pasted")
        version = versions[request.abstract_source]

        parsed_pdfs = [primary]
        seen_hashes = {primary.document["sha256"]}
        diagnostics = []
        for path in request.supplements:
            try:
                supplement = parse_pdf(path, role="supplement")
            except (OSError, ValueError) as exc:
                diagnostics.append({"code": "supplement_unreadable", "name": path.name, "reason": str(exc)})
                continue
            sha256 = supplement.document["sha256"]
            if sha256 not in seen_hashes:
                parsed_pdfs.append(supplement)
                seen_hashes.add(sha256)

        documents = []
        evidence = []
        for parsed in parsed_pdfs:
            blob_path = self._store_pdf(parsed.data, parsed.document["sha256"])
            documents.append({**parsed.document, "blob_path": blob_path})
            evidence.extend(parsed.evidence)

        return self._persist_report({
            "schema_version": 1,
            "mode": "pdf",
            "run_id": uuid4().hex,
            "report_version_id": version["version_id"],
            "original_abstract": version,
            "reviewed_abstract": version,
            "abstract_versions": list(versions.values()),
            "documents": documents,
            "diagnostics": diagnostics,
            "findings": [],
            "revisions": [],
            "verifications": [],
            "abstract_review": {"version_id": version["version_id"], "status": "pending"},
            "evidence": evidence,
            "calculations": [],
            "omissions": [],
            "external_checks": [],
            "unresolved_items": [{"code": "review_pending", "reason": "PDF evidence has been parsed for review."}],
            "usage": {"external_requests": 0, "local_model_calls": 0},
        })

    def _store_pdf(self, data: bytes, sha256: str) -> str:
        relative_path = Path("blobs") / f"{sha256}.pdf"
        path = self.database.parent / relative_path
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = None
            try:
                with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
                    temporary_path = Path(temporary.name)
                    temporary.write(data)
                temporary_path.replace(path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        return relative_path.as_posix()

    def _persist_report(self, report: dict) -> dict:
        run_id = report["run_id"]
        version_id = report["report_version_id"]
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    report_version_id TEXT NOT NULL,
                    report_json TEXT NOT NULL
                )"""
            )
            connection.execute(
                "INSERT INTO runs (run_id, report_version_id, report_json) VALUES (?, ?, ?)",
                (run_id, version_id, json.dumps(report, ensure_ascii=False)),
            )
        return self._run_ref(run_id, version_id)

    def status(self, run_id: str) -> dict:
        row = self._get_run(run_id)
        return self._run_ref(run_id, row[0])

    def export(self, run_id: str, version_id: str) -> dict:
        stored_version, report = self._get_run(run_id)
        if stored_version != version_id:
            raise ValueError(f"report version {version_id!r} does not belong to run {run_id!r}")
        return json.loads(report)

    def _get_run(self, run_id: str) -> tuple[str, str]:
        if self.database.is_file():
            with closing(sqlite3.connect(self.database)) as connection:
                table = connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'runs'")
                if table.fetchone() is not None:
                    row = connection.execute(
                        "SELECT report_version_id, report_json FROM runs WHERE run_id = ?", (run_id,)
                    ).fetchone()
                    if row is not None:
                        return row
        raise ValueError(f"unknown review run {run_id!r}")

    @staticmethod
    def _run_ref(run_id: str, version_id: str) -> dict:
        return {"run_id": run_id, "report_version_id": version_id, "run_status": "report_ready"}
