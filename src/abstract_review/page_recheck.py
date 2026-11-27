"""Reassemble original PDF pages for a second look at unresolved claims."""

import json
from collections import defaultdict
from contextlib import ExitStack
from typing import Literal

import pymupdf

from .evidence import EvidenceIndex, _terms
from .ingest import read_archived_pdf
from .models import ModelPaused, ModelUnavailable, Schema


class PageJudgement(Schema):
    verdict: Literal["supported", "refuted", "pending"]
    evidence_ids: list[str]
    reason: str


class PageConflictJudgement(PageJudgement):
    conflict: Literal["different_subclaims", "same_scope", "unresolved"]


def needs_conflict_review(finding: dict) -> bool:
    blockers = finding.get("decision_trace", {}).get("blocker_codes", ())
    return bool({"opposing_evidence", "conflicting_support"}.intersection(blockers))


def _page(row: dict) -> tuple[str, int]:
    return row["doc_id"], row["locator"]["page_index"]


def _rank_pages(claim: str, index: EvidenceIndex, records: dict, finding: dict) -> list[tuple[str, int]]:
    hits = index.search(claim, limit=48, expand=False)["evidence"]
    lexical = index._lexical_rank(_terms(claim), records.keys())[:36]
    weights = defaultdict(float)
    for ids, multiplier in (
        (finding.get("sent_candidate_ids", finding.get("candidate_ids", ())), 1.0),
        ([row["evidence_id"] for row in hits], 1.5),
        (lexical, 0.8),
    ):
        for rank, evidence_id in enumerate(ids, 1):
            if row := records.get(evidence_id):
                weights[_page(row)] += multiplier / (12 + rank)
    for page in {_page(records[eid]) for eid in finding["evidence_ids"] if eid in records}:
        weights[page] += 0.30
    return sorted(weights, key=lambda page: (-weights[page], page))


def _raw_excerpt(pdf, page: int, terms: set[str]) -> str:
    raw = pdf[page].get_text(sort=True)
    if len(raw) <= 8000:
        return raw
    lines = raw.splitlines()
    relevant = [i for i, line in enumerate(lines) if len(_terms(line) & terms) >= 2]
    included = {j for i in relevant for j in range(max(0, i - 4), min(i + 5, len(lines)))}
    return "\n".join(line for i, line in enumerate(lines) if i in included)[:8000]


def _page_packet(page: int, rows: list[dict], references: dict, raw: str, label: str) -> str:
    tables = defaultdict(list)
    lines = []
    for row in rows:
        loc = row["locator"]
        alias = f"e{len(references) + 1}"
        references[alias] = row["evidence_id"]
        text = " ".join(row["text"].split())
        position = tuple(loc.get(key) for key in ("table_index", "row_index", "column_index"))
        if row["kind"] == "table_cell" and all(type(value) is int for value in position):
            table, number, column = position
            tables[table, number].append((column, alias, text))
        else:
            lines.append(f"[{alias}] {text}")
    section = [f"PDF PAGE {page + 1}{label}"]
    for (table, number), cells in sorted(tables.items()):
        values = " | ".join(f"{col}:[{alias}] {text}" for col, alias, text in sorted(cells))
        section.append(f"T{table} R{number} {values}")
    section.extend(lines)
    if raw:
        section.append("PDF TEXT EXTRACT\n" + raw)
    return "\n".join(section)


def _packet(claim: str, index: EvidenceIndex, evidence, documents, data_dir, finding: dict):
    sources = {doc["doc_id"]: doc for doc in documents if doc.get("blob_path")}
    records = {row["evidence_id"]: row for row in evidence if row["doc_id"] in sources}
    by_page = defaultdict(list)
    for row in records.values():
        by_page[_page(row)].append(row)
    pages = []
    references = {}
    sections = []
    terms = _terms(claim)
    with ExitStack() as stack:
        pdfs = {}
        for doc_id, page in _rank_pages(claim, index, records, finding)[:4]:
            if doc_id not in pdfs:
                pdfs[doc_id] = None
                try:
                    if data := read_archived_pdf(data_dir, sources[doc_id]):
                        pdfs[doc_id] = stack.enter_context(pymupdf.open(stream=data, filetype="pdf"))
                except (OSError, ValueError, RuntimeError):
                    pass
            pdf = pdfs[doc_id]
            if pdf is None or not 0 <= page < len(pdf):
                continue
            rows = by_page[doc_id, page]
            raw = ""
            if sum(row["kind"] == "table_cell" for row in rows) < 15:
                try:
                    raw = _raw_excerpt(pdf, page, terms)
                except (ValueError, RuntimeError):
                    pass
            pending_refs = dict(references)
            label = f" ({sources[doc_id]['role']}, {doc_id})" if len(sources) > 1 else ""
            section = _page_packet(page, rows, pending_refs, raw, label)
            packet = "\n\n".join([*sections, section])
            if len(json.dumps(packet, ensure_ascii=True)) >= 73_000:
                continue
            references = pending_refs
            sections.append(section)
            pages.append({"doc_id": doc_id, "page_index": page})
    return "\n\n".join(sections), references, pages


def eligible(finding: dict, model, documents, data_dir) -> bool:
    if (model is None or data_dir is None or finding["review_status"] != "pending"
            or not any(doc.get("blob_path") for doc in documents)):
        return False
    if finding.get("page_recheck", {}).get("status") == "queued":
        return True
    blockers = finding.get("decision_trace", {}).get("blocker_codes", ())
    return (
        not finding.get("retryable")
        and finding.get("kind") not in {"background", "purpose"}
        and finding.get("stop_reason") not in {"invalid_judge_reference", "model_unavailable"}
        and "calculation_disagrees" not in blockers
    )


def recheck(finding: dict, index: EvidenceIndex, evidence, documents, data_dir, model,
            calculation: dict | None) -> dict:
    claim = finding["proposition"]
    packet, references, pages = _packet(claim, index, evidence, documents, data_dir, finding)
    if not pages:
        return {"status": "source_unavailable"}
    trace = {"pages": pages, "initial_stop_reason": finding["stop_reason"]}
    conflict = needs_conflict_review(finding)
    stage = "page_recheck_conflict" if conflict else "page_recheck"
    schema = PageConflictJudgement if conflict else PageJudgement
    try:
        answer = model.call(stage, f"Claim: {claim}\n\n{packet}", schema)
    except ModelPaused:
        raise
    except ModelUnavailable:
        return {**trace, "status": "unavailable"}
    if conflict:
        trace["conflict"] = answer.conflict
        if answer.conflict != "different_subclaims":
            return {**trace, "status": "pending", "reason": answer.reason}
    if answer.verdict == "pending":
        return {**trace, "status": "pending", "reason": answer.reason}
    if not answer.reason.strip() or not answer.evidence_ids or not set(answer.evidence_ids) <= references.keys():
        return {**trace, "status": "invalid_citations"}
    if calculation and (answer.verdict == "supported") != (calculation["verification"] == "matched"):
        return {**trace, "status": "calculation_disagrees"}
    ids = list(dict.fromkeys(references[key] for key in answer.evidence_ids))
    role = "support" if answer.verdict == "supported" else "refute"
    finding.update(review_status="completed", verdict=answer.verdict, reason=answer.reason, evidence_ids=ids,
                   evidence_groups=[{"role": role, "evidence_ids": ids}], missing_fields=[],
                   stop_reason="verdict")
    for key in ("coverage", "empirical_support_blocker", "refutation_blocker", "evidence_ranges"):
        finding.pop(key, None)
    return {**trace, "status": "adopted", "evidence_ids": ids, "sent_evidence_ids": list(references.values())}
