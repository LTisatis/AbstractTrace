"""SciFact cited-abstract evaluation with original document and sentence IDs."""

import hashlib
import json
import sqlite3
import time
from contextlib import closing, contextmanager
from pathlib import Path

from .audit import review_claim, supplied_claims
from .benchmark import _prediction
from .evidence import EvidenceIndex
from .models import ModelPaused, ModelUnavailable, ResponsesModel
from .scoring import _ratio, score_predictions


_DATASET = "scifact-cited-abstracts"
_VERDICTS = {"SUPPORT": "supported", "CONTRADICT": "refuted"}
_BATCH_DEFAULT_REQUESTS = 8
_BATCH_DEFAULT_SECONDS = 180
_BATCH_SAMPLE_SECONDS = 45
_TOKEN_COUNT_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


def _jsonl(path: Path, source: bytes):
    for index, line in enumerate(source.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{index}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index}: expected a JSON object")
        yield row


def _positive_id(value: object, location: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{location} must be a positive integer ID")
    return value


def _claim_id(value: object, location: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{location} must be a non-negative integer ID")
    return value


def _evidence_id(doc_id: int, index: int) -> str:
    return f"scifact:{doc_id}:{index}"


def _sample_id(claim_id: int, doc_id: int) -> str:
    return f"scifact:{claim_id}:{doc_id}"


def _document(row: dict) -> dict:
    doc_id = _positive_id(row.get("doc_id"), "doc_id")
    sentences = row.get("abstract")
    if not isinstance(sentences, list) or not sentences or not all(isinstance(item, str) for item in sentences):
        raise ValueError(f"SciFact abstract {doc_id}: expected a list of original sentence strings")
    return {"doc_id": doc_id, "abstract": sentences}


def _gold(row: dict, doc_id: int, count: int) -> tuple[str, list[list[str]]]:
    if str(doc_id) not in row:
        return "insufficient", []
    rationales = row[str(doc_id)]
    if not isinstance(rationales, list) or not rationales:
        raise ValueError(f"SciFact document {doc_id}: annotated evidence is empty")
    groups, labels = [], set()
    for item in rationales:
        label = item.get("label") if isinstance(item, dict) else None
        if not isinstance(label, str) or label not in _VERDICTS:
            raise ValueError(f"SciFact document {doc_id}: invalid rationale label")
        indexes = item.get("sentences")
        if (not isinstance(indexes, list) or not indexes
                or any(type(index) is not int or not 0 <= index < count for index in indexes)
                or len(set(indexes)) != len(indexes)):
            raise ValueError(f"SciFact document {doc_id}: invalid original sentence indexes")
        groups.append([_evidence_id(doc_id, index) for index in indexes])
        labels.add(_VERDICTS[label])
    if len(labels) != 1:
        raise ValueError(f"SciFact document {doc_id}: contradictory annotations for one claim-document pair")
    return labels.pop(), groups


def prepare_scifact(corpus: Path, claims: Path, *, split: str, dataset_version: str) -> tuple[dict, dict]:
    """Adapt labeled source JSONL. Candidate documents come from the cited IDs."""
    if not isinstance(dataset_version, str) or not dataset_version.strip() or split not in {"train", "dev"}:
        raise ValueError("SciFact requires an explicit dataset version and a labeled train or dev split")
    try:
        raw_corpus, raw_claims = corpus.read_bytes(), claims.read_bytes()
        documents = {}
        for row in _jsonl(corpus, raw_corpus):
            document = _document(row)
            doc_id = document["doc_id"]
            if doc_id in documents:
                raise ValueError(f"duplicate SciFact doc_id {doc_id}")
            documents[doc_id] = document

        selected, samples, labels, seen_claims = {}, [], [], set()
        for row in _jsonl(claims, raw_claims):
            claim_id = _claim_id(row.get("id"), "claim id")
            text, cited, evidence = row.get("claim"), row.get("cited_doc_ids"), row.get("evidence")
            if claim_id in seen_claims:
                raise ValueError(f"duplicate SciFact claim id {claim_id}")
            seen_claims.add(claim_id)
            if (not isinstance(text, str) or not text.strip() or not isinstance(cited, list) or not cited
                    or not isinstance(evidence, dict)):
                raise ValueError(f"SciFact claim {claim_id}: missing claim text, cited IDs, or labeled evidence")
            doc_ids = list(dict.fromkeys(_positive_id(doc_id, "cited_doc_ids entry") for doc_id in cited))
            if not evidence.keys() <= {str(doc_id) for doc_id in doc_ids}:
                raise ValueError(f"SciFact claim {claim_id}: inconsistent annotated or cited documents")
            for doc_id in doc_ids:
                if doc_id not in documents:
                    raise ValueError(f"SciFact document {doc_id} is missing from the corpus")
                doc = selected[doc_id] = documents[doc_id]
                sample_id = _sample_id(claim_id, doc_id)
                samples.append({"sample_id": sample_id, "claim_id": claim_id, "doc_id": doc_id, "text": text})
                verdict, groups = _gold(evidence, doc_id, len(doc["abstract"]))
                labels.append({"sample_id": sample_id, "claim_id": str(claim_id), "verdict": verdict,
                               "accepted_evidence_groups": groups, "split": split})
    except UnicodeError as exc:
        raise ValueError("SciFact source JSONL is not UTF-8") from exc
    if not samples:
        raise ValueError("SciFact input has no claim-document pairs")
    manifest = {
        "schema_version": 1, "dataset": _DATASET, "dataset_version": dataset_version.strip(), "split": split,
        "source_sha256": {"corpus": hashlib.sha256(raw_corpus).hexdigest(), "claims": hashlib.sha256(raw_claims).hexdigest()},
        "documents": list(selected.values()), "samples": samples,
    }
    return manifest, {"schema_version": 1, "labels": labels}


def _load_inputs(manifest: object) -> tuple[dict, list[dict]]:
    required = {"schema_version", "dataset", "dataset_version", "split", "source_sha256", "documents", "samples"}
    if (not isinstance(manifest, dict) or manifest.keys() != required
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["dataset"] != _DATASET or not isinstance(manifest["documents"], list)
            or not isinstance(manifest["samples"], list) or not manifest["samples"]):
        raise ValueError("expected a SciFact cited-abstract input manifest")
    documents = {}
    for row in manifest["documents"]:
        if not isinstance(row, dict) or row.keys() != {"doc_id", "abstract"}:
            raise ValueError("a SciFact input document may contain only doc_id and abstract")
        doc = _document(row)
        if doc["doc_id"] in documents:
            raise ValueError(f"duplicate SciFact doc_id {doc['doc_id']}")
        documents[doc["doc_id"]] = doc["abstract"]
    samples, seen = [], set()
    for row in manifest["samples"]:
        if not isinstance(row, dict) or row.keys() != {"sample_id", "claim_id", "doc_id", "text"}:
            raise ValueError("a SciFact input sample may contain only sample_id, claim_id, doc_id, and text")
        claim_id = _claim_id(row["claim_id"], "claim_id")
        doc_id = _positive_id(row["doc_id"], "doc_id")
        sample_id = row["sample_id"]
        if (sample_id != _sample_id(claim_id, doc_id) or sample_id in seen or doc_id not in documents
                or not isinstance(row["text"], str) or not row["text"].strip()):
            raise ValueError("invalid SciFact sample ID, document, or claim")
        seen.add(sample_id)
        samples.append(row)
    return documents, samples


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _token_totals(requests: int, events: list[dict]) -> dict:
    tokens = {}
    for field in _TOKEN_COUNT_FIELDS:
        known = [event[field] for event in events if type(event.get(field)) is int and event[field] >= 0]
        tokens[field] = sum(known) if len(known) == requests else None
    return tokens


class _SciFactBatch:
    def __init__(self, db: sqlite3.Connection, identity: str, used: int, seconds: float,
                 max_requests: int, max_seconds: int) -> None:
        self.db = db
        self.identity = identity
        self.requests_used = used
        self.previous_seconds = seconds
        self.max_requests = max_requests
        self.max_seconds = max_seconds
        self.started = time.monotonic()
        self.stop_reason = None
        self.results = []

    @property
    def elapsed(self) -> float:
        return self.previous_seconds + (time.monotonic() - self.started)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed)

    def exhaustion(self) -> str | None:
        if self.requests_used >= self.max_requests:
            return "request_budget_exhausted"
        if self.remaining_seconds <= 0:
            return "time_budget_exhausted"
        return None

    def reserve(self) -> None:
        if reason := self.exhaustion():
            raise ModelPaused(reason)
        with self.db:
            # The sample can stop before its outcome is known. Reserve its remaining processing
            # window before transport, then settle to actual elapsed time when the sample returns.
            self.db.execute("UPDATE scifact_batch SET requests = ?, seconds = ?",
                            (self.requests_used + 1, min(self.max_seconds, self.elapsed + _BATCH_SAMPLE_SECONDS)))
        self.requests_used += 1
        if self.remaining_seconds <= 0:
            raise ModelPaused("time_budget_exhausted")

    def _settle(self) -> None:
        seconds = max(self.previous_seconds, min(self.max_seconds, self.elapsed))
        self.db.execute("UPDATE scifact_batch SET seconds = ?", (seconds,))

    def saved(self, sample_id: str, digest: str) -> dict | None:
        row = self.db.execute("SELECT prediction, result FROM scifact_samples WHERE sample_id = ? AND input_digest = ?",
                              (sample_id, digest)).fetchone()
        if row is None:
            return None
        prediction, result = map(json.loads, row)
        self.results.append({**result, "cache_hit": True})
        return prediction

    def record(self, sample_id: str, digest: str, prediction: dict, requests: int, events: list[dict]) -> None:
        result = {"sample_id": sample_id, "input_digest": digest, "model_requests": requests,
                  "tokens": _token_totals(requests, events), "cache_hit": False}
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO scifact_samples VALUES (?, ?, ?, ?)",
                            (sample_id, digest, json.dumps(prediction, ensure_ascii=False), json.dumps(result)))
            self._settle()
        self.results.append(result)

    def pause(self, sample_id: str, digest: str, requests: int, events: list[dict], *, finding: dict | None = None) -> None:
        self.stop_reason = self.exhaustion() or "model_paused"
        diagnostics = {"sample_id": sample_id, "input_digest": digest, "model_requests": requests,
                       "tokens": _token_totals(requests, events), "stop_reason": self.stop_reason}
        if finding is not None:
            diagnostics["finding_stop_reason"] = finding["stop_reason"]
            if "decision_trace" in finding:
                diagnostics["decision_trace"] = finding["decision_trace"]
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO scifact_pause VALUES (1, ?)",
                            (json.dumps(diagnostics, ensure_ascii=False),))
            self._settle()

    def summary(self, sample_count: int) -> dict:
        with self.db:
            self._settle()
        seconds, = self.db.execute("SELECT seconds FROM scifact_batch").fetchone()
        last_pause = self.db.execute("SELECT diagnostics FROM scifact_pause WHERE slot = 1").fetchone()
        return {"status": "paused" if self.stop_reason else "completed", "stop_reason": self.stop_reason,
                "sample_count": sample_count, "processed_samples": len(self.results),
                "max_requests": self.max_requests, "requests_used": self.requests_used,
                "requests_without_sample_checkpoint": self.requests_used - sum(row["model_requests"] for row in self.results),
                "max_seconds": self.max_seconds, "elapsed_seconds": round(seconds, 3),
                "last_paused_sample": json.loads(last_pause[0]) if last_pause else None}


@contextmanager
def scifact_checkpoint(path: Path, manifest_sha256: str, configuration: dict, *, resume: bool,
                       max_requests: int = _BATCH_DEFAULT_REQUESTS, max_seconds: int = _BATCH_DEFAULT_SECONDS):
    """Open a single-writer checkpoint and its cumulative budget for this exact batch."""
    if any(type(value) is not int or value <= 0 for value in (max_requests, max_seconds)):
        raise ValueError("SciFact batch limits must be positive integers")
    identity = _digest([1, manifest_sha256, configuration])
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=0)) as db:
        try:
            # Exclusive locking mode retains the file lock across commits and releases it on exit.
            db.execute("PRAGMA locking_mode = EXCLUSIVE")
            db.execute("PRAGMA synchronous = FULL")
            db.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            if getattr(exc, "sqlite_errorcode", None) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                raise ValueError("this SciFact benchmark is already running in this output directory") from exc
            raise
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS scifact_batch (identity TEXT PRIMARY KEY, requests INTEGER NOT NULL, seconds REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS scifact_samples (sample_id TEXT PRIMARY KEY, input_digest TEXT NOT NULL, prediction TEXT NOT NULL, result TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS scifact_pause (slot INTEGER PRIMARY KEY, diagnostics TEXT NOT NULL)")
            stored = db.execute("SELECT identity, requests, seconds FROM scifact_batch").fetchone()
            if stored is None:
                db.execute("INSERT INTO scifact_batch VALUES (?, 0, 0)", (identity,))
                stored = identity, 0, 0.0
            elif not resume:
                raise ValueError("this output directory has SciFact checkpoints; use --resume or another output directory")
            elif stored[0] != identity:
                raise ValueError("SciFact checkpoint inputs or configuration changed; use another output directory")
        yield _SciFactBatch(db, *stored, max_requests, max_seconds)


def run_scifact(manifest: object, engine: str, model_name: str, *, make_model=ResponsesModel,
                batch: _SciFactBatch | None = None) -> list[dict]:
    """Apply the shared claim retrieval and judge to input-only cited abstracts."""
    if engine not in {"offline", "openai"}:
        raise ValueError("SciFact benchmark engine must be offline or openai")
    if batch is not None and engine != "openai":
        raise ValueError("SciFact batch checkpoints are for the online engine")
    documents, samples = _load_inputs(manifest)
    predictions = []
    for sample in samples:
        claim_id, doc_id = str(sample["claim_id"]), sample["doc_id"]
        sample_id = sample["sample_id"]
        if batch is not None:
            input_digest = _digest([batch.identity, sample, documents[doc_id]])
            if previous := batch.saved(sample_id, input_digest):
                predictions.append(previous)
                continue
            batch.stop_reason = batch.exhaustion()
            if batch.stop_reason is not None:
                break
            before = batch.requests_used
        claim, = supplied_claims(((claim_id, sample["text"]),))
        evidence = [
            {"evidence_id": _evidence_id(doc_id, n), "doc_id": str(doc_id), "kind": "body_text", "text": text,
             "locator": {"page_index": 0, "section": "abstract", "sentence_index": n}}
            for n, text in enumerate(documents[doc_id]) if text.strip()
        ]
        index = EvidenceIndex(evidence)
        model = None
        try:
            if batch is None:
                model = None if engine == "offline" else make_model(model_name, max_requests=4, max_seconds=180)
            else:
                model = make_model(model_name, max_requests=min(4, batch.max_requests - before),
                                   max_seconds=min(_BATCH_SAMPLE_SECONDS, batch.remaining_seconds), before_call=batch.reserve)
            source_ids = None if model is None else [row["evidence_id"] for row in evidence]
            finding, _ = review_claim(claim, index, model, cited_abstract_ids=source_ids)
        except ModelUnavailable as exc:
            if batch is not None and isinstance(exc, ModelPaused):
                batch.pause(sample_id, input_digest, batch.requests_used - before, getattr(model, "events", []))
                break
            raise ValueError(f"SciFact sample {sample['sample_id']}: {exc}") from exc
        finally:
            index.close()
        if batch is not None and finding.get("retryable"):
            batch.pause(sample_id, input_digest, batch.requests_used - before, getattr(model, "events", []), finding=finding)
            break
        prediction = _prediction(sample_id, claim_id, finding)
        if batch is not None:
            batch.record(sample_id, input_digest, prediction, batch.requests_used - before, getattr(model, "events", []))
        predictions.append(prediction)
    return predictions


def score_scifact(predictions: dict, gold: dict) -> dict:
    """Score labeled cited-abstract pairs using SciFact's abstract and sentence rules."""
    scored = score_predictions(predictions, gold)
    source_rows = {(row["sample_id"], row["claim_id"]): row for row in predictions["predictions"]}
    labels = {(row["sample_id"], row["claim_id"]): row for row in gold["labels"]}
    relevant_abstracts = retrieved_abstracts = relevant_sentences = retrieved_sentences = 0
    abstract_label = abstract_rationale = sentence_selection = sentence_label = 0
    for case in scored["cases"]:
        key = case["sample_id"], case["claim_id"]
        sample_id, claim_id = key
        prefix, separator, document_id = sample_id.rpartition(":")
        if not separator or prefix != f"scifact:{claim_id}" or not document_id.isascii() or not document_id.isdecimal():
            raise ValueError(f"{sample_id}/{claim_id} is not a SciFact claim-document ID")
        groups = [set(group) for group in labels[key]["accepted_evidence_groups"]]
        cited = source_rows[key]["evidence_ids"] if key in source_rows else []
        evidence_prefix = f"scifact:{document_id}:"
        for evidence_id in set(cited).union(*groups):
            index = evidence_id.removeprefix(evidence_prefix)
            if index == evidence_id or not index.isascii() or not index.isdecimal():
                raise ValueError(f"{sample_id}: {evidence_id} is outside the cited abstract")
        relevant_abstracts += bool(groups)
        relevant_sentences += sum(map(len, groups))
        if case["predicted_verdict"] not in _VERDICTS.values():
            continue
        retrieved_abstracts += 1
        retrieved_sentences += len(cited)
        correct = case["label_correct"]
        abstract_label += bool(groups and correct)
        abstract_rationale += int(correct and any(group <= set(cited[:3]) for group in groups))
        covered = set().union(*(group for group in groups if group <= set(cited)))
        sentence_selection += len(covered)
        sentence_label += len(covered) if correct else 0

    def metrics(correct: int, retrieved: int, relevant: int) -> dict:
        return {"precision": _ratio(correct, retrieved), "recall": _ratio(correct, relevant),
                "f1": _ratio(2 * correct, retrieved + relevant)}

    scored["scifact_cited_abstracts"] = {
        "abstract_label_only": metrics(abstract_label, retrieved_abstracts, relevant_abstracts),
        "abstract_rationalized": metrics(abstract_rationale, retrieved_abstracts, relevant_abstracts),
        "sentence_selection": metrics(sentence_selection, retrieved_sentences, relevant_sentences),
        "sentence_label": metrics(sentence_label, retrieved_sentences, relevant_sentences),
    }
    return scored
