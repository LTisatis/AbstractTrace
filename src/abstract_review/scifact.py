"""SciFact cited-abstract evaluation with original document and sentence IDs."""

import hashlib
import json
from pathlib import Path

from .audit import review_claim, supplied_claims
from .benchmark import _prediction
from .evidence import EvidenceIndex
from .models import ModelUnavailable, ResponsesModel
from .scoring import _ratio, score_predictions


_DATASET = "scifact-cited-abstracts"
_VERDICTS = {"SUPPORT": "supported", "CONTRADICT": "refuted"}


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


def run_scifact(manifest: object, engine: str, model_name: str, *, make_model=ResponsesModel) -> list[dict]:
    """Apply the shared claim retrieval and judge to input-only cited abstracts."""
    if engine not in {"offline", "openai"}:
        raise ValueError("SciFact benchmark engine must be offline or openai")
    documents, samples = _load_inputs(manifest)
    predictions = []
    for sample in samples:
        claim_id, doc_id = str(sample["claim_id"]), sample["doc_id"]
        claim, = supplied_claims(((claim_id, sample["text"]),))
        evidence = [
            {"evidence_id": _evidence_id(doc_id, n), "doc_id": str(doc_id), "kind": "body_text", "text": text,
             "locator": {"page_index": 0, "section": "abstract", "sentence_index": n}}
            for n, text in enumerate(documents[doc_id]) if text.strip()
        ]
        index = EvidenceIndex(evidence)
        try:
            model = None if engine == "offline" else make_model(model_name, max_requests=4, max_seconds=180)
            finding, _ = review_claim(claim, index, model)
        except ModelUnavailable as exc:
            raise ValueError(f"SciFact sample {sample['sample_id']}: {exc}") from exc
        finally:
            index.close()
        predictions.append(_prediction(sample["sample_id"], claim_id, finding))
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
