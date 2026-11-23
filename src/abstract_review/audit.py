"""Evidence retrieval, scoped calculations, and independent original verdicts."""

import hashlib
import re
from base64 import b64encode
from collections.abc import Callable, Iterable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field

from .calculations import comparison_for_claim
from .evidence import EvidenceIndex
from .models import Extraction, Judgement, ModelPaused, ModelUnavailable, Schema


class FieldCheck(Schema):
    requirement: str
    status: Literal["found", "searched_missing", "unchecked", "unreadable"]
    evidence_ids: list[str]


class FollowUpTarget(Schema):
    query: str
    requirement: str | None = None
    section: str | None = None
    table_label: str | None = None
    page_index: int | None = None
    doc_id: str | None = None


class ScopedJudgement(Judgement):
    coverage: list[FieldCheck] = Field(default_factory=list)
    follow_up_targets: list[FollowUpTarget] = Field(default_factory=list)


_SCOPE_KEYS = ("section", "table_label", "page_index", "doc_id")
_TABLE_QUERY = re.compile(r"(?<![A-Za-z0-9])(?:table|tab\.)\s*([A-Za-z]*\d+)|表\s*([A-Za-z]*\d+)", re.I)
_PAGE_QUERY = re.compile(r"(?:\b(?:page|p\.)\s*(\d+)|第\s*(\d+)\s*页)", re.I)
_SECTION_QUERY = re.compile(
    r"(?:methods?|materials and methods|experimental setup|results|discussion|limitations|"
    r"appendix(?:\s+[a-z0-9]+)?|方法|材料与方法|实验设置|结果|讨论|局限性|附录(?:\s*[a-z0-9]+)?)", re.I,
)


def _text_target(query: str) -> dict:
    target = {"query": query}
    if match := _TABLE_QUERY.search(query):
        target["table_label"] = match[1] or match[2]
    if match := _PAGE_QUERY.search(query):
        target["page_index"] = int(match[1] or match[2]) - 1
    if _SECTION_QUERY.fullmatch(query.strip()):
        target["section"] = query.strip()
    return target


def normalize_targets(queries: Iterable[str | dict], documents=()) -> list[dict]:
    targets = []
    known_docs = {document["doc_id"] for document in documents}
    for raw in queries:
        item = _text_target(raw) if isinstance(raw, str) else raw
        if not isinstance(item, dict) or not isinstance(item.get("query"), str):
            continue
        query = item["query"].strip()
        if not query or len(query) > 500:
            continue
        target = {"query": query}
        for key in ("requirement", "section", "table_label", "doc_id"):
            value = item.get(key)
            if isinstance(value, str) and value.strip() and len(value) <= 250:
                target[key] = value.strip()
        if target.get("doc_id") and known_docs and target["doc_id"] not in known_docs:
            continue
        page = item.get("page_index")
        if page is not None:
            if type(page) is not int or page < 0:
                continue
            target["page_index"] = page
        if target not in targets:
            targets.append(target)
        if len(targets) == 2:
            break
    return targets


def _page_failures(scope: dict, documents) -> list[dict]:
    failures = []
    allowed_docs = {scope["doc_id"]} if scope["doc_id"] else set(scope["document_ids"])
    for document in documents:
        doc_id = document["doc_id"]
        if allowed_docs and doc_id not in allowed_docs:
            continue
        for page in document.get("pages", []):
            if scope["page_index"] is not None and page["page_index"] != scope["page_index"]:
                continue
            if page.get("status") in {"unreadable", "no_text"} or page.get("table_status") == "unreadable":
                failures.append({"doc_id": doc_id, "page_index": page["page_index"],
                                 "status": page.get("table_status") or page["status"]})
    return failures


def targeted_evidence(index: EvidenceIndex, queries: Iterable[str | dict], *, documents=(), limit: int = 8) -> dict:
    """Check up to two named sources, and keep ordinary retrieval distinct from direct inspection."""
    documents = tuple(documents)
    sources = normalize_targets(queries, documents)
    evidence = {}
    scopes = []
    paths = []
    for source in sources:
        target = {key: source.get(key) for key in _SCOPE_KEYS}
        if any(target[key] is not None for key in ("section", "table_label", "page_index")):
            result = index.inspect(**target, limit=limit)
            scope = result["scope"]
            failures = _page_failures(scope, documents)
            scope["unreadable_locations"] = failures
            scope["complete"] = scope["complete"] and not failures
        else:
            result = index.search(source["query"], limit=limit, doc_id=source.get("doc_id"))
            scope = {**result["scope"], "mode": "ranked"}
        scopes.append({**scope, "query": source["query"], "requirement": source.get("requirement")})
        paths.extend(result.get("paths", []))
        evidence.update((record["evidence_id"], record) for record in result["evidence"])
    return {"evidence": list(evidence.values())[:limit], "paths": paths, "scopes": scopes}


def _field_key(value: str) -> str:
    return " ".join(value.casefold().split())


def _pending_targets(output, requirements: list[str], documents) -> list[dict]:
    proposals = [target.model_dump() for target in getattr(output, "follow_up_targets", [])]
    proposals.extend(output.follow_up_queries)
    # If the model calls the field missing, an explicitly named source in its original requirement
    # gives the controller a direct review path even without a proposed retrieval query.
    missing = {_field_key(field) for field in output.missing_fields}
    for field in requirements:
        target = _text_target(field)
        if _field_key(field) in missing and any(target.get(key) is not None for key in _SCOPE_KEYS):
            proposals.append({**target, "requirement": field})
    targets = normalize_targets(proposals, documents)
    if len(missing) == 1:
        requirement = next(iter(missing))
        for target in targets:
            target.setdefault("requirement", requirement)
    return targets


def _coverage(requirements: list[str], output, scopes: list[dict], sent: list[str], offline_ids=()) -> list[dict]:
    missing = {_field_key(field): field for field in output.missing_fields} if output is not None else {}
    fields = {_field_key(field): field for field in (*requirements, *missing.values()) if field.strip()}
    observations = getattr(output, "coverage", [])
    allowed = set(sent)
    covered = []
    for key, field in fields.items():
        direct = [scope for scope in scopes if scope.get("mode") == "direct"
                  and _field_key(scope.get("requirement") or "") == key]
        checks = [check for check in observations if _field_key(check.requirement) == key]
        found = [eid for check in checks if check.status == "found" and check.evidence_ids
                 and allowed.issuperset(check.evidence_ids) for eid in check.evidence_ids]
        found.extend(offline_ids)
        unreadable = [loc for scope in direct for loc in scope.get("unreadable_locations", [])]
        examined = list(dict.fromkeys(eid for scope in direct for eid in scope["examined_evidence_ids"]))
        complete = bool(direct) and all(scope["complete"] for scope in direct) and allowed.issuperset(examined)
        if found:
            status, ids = "found", list(dict.fromkeys(found))
        elif unreadable:
            status, ids = "unreadable", examined
        elif complete and (key in missing or any(check.status == "searched_missing" for check in checks)):
            status, ids = "searched_missing", examined
        else:
            status, ids = "unchecked", []
        covered.append({"requirement": field, "status": status, "evidence_ids": ids,
                        "checked_sources": [scope["locations"] for scope in direct],
                        "unreadable_locations": unreadable})
    return covered


def sentences(text: str) -> list[dict]:
    parts = []
    start = 0
    boundaries = re.finditer(r"[!?。！？](?=\s|[^a-zA-Z0-9]|$)|[。！？]|\.(?=\s|$)|\n+", text)
    for end in [*(match.end() for match in boundaries), len(text)]:
        segment = text[start:end]
        stripped = segment.strip()
        if stripped:
            offset = start + len(segment) - len(segment.lstrip())
            parts.append({"sentence_id": f"s{len(parts) + 1}", "text": stripped,
                          "start": offset, "end": offset + len(stripped)})
        start = end
    return parts


def extract_claims(text: str, model=None) -> list[dict]:
    parts = sentences(text)
    if model is None:
        extracted = [{"sentence_id": part["sentence_id"], "quote": part["text"], "proposition": part["text"],
                      "kind": "quantitative" if re.search(r"\d+(?:[.,]\d+)?\s*%|百分点|percentage points?", part["text"], re.I) else "result",
                      "requirements": ["observed result and scope"], "support_query": part["text"],
                      "counter_query": "limitation exception subgroup " + part["text"]} for part in parts]
    else:
        parsed = model.call("extract", {"sentences": parts}, Extraction)
        extracted = [claim.model_dump() for claim in parsed.claims]
    by_sentence = {part["sentence_id"]: part for part in parts}
    results = {}
    for claim in extracted:
        part = by_sentence.get(claim["sentence_id"])
        quote = claim["quote"]
        if part is None or not quote.strip() or quote not in part["text"]:
            raise ModelUnavailable("claim extraction returned an invalid abstract anchor")
        start = part["start"] + part["text"].index(quote)
        claim_id = hashlib.sha256(f"{start}:{quote}:{claim['proposition']}".encode()).hexdigest()[:20]
        results[claim_id] = {**claim, "claim_id": claim_id, "start": start, "end": start + len(quote),
                             "sentence_start": part["start"], "sentence_end": part["end"], "text": quote}
    return list(results.values())


def supplied_claims(source: tuple[tuple[str, str], ...]) -> list[dict]:
    results = []
    offset = 0
    for index, (claim_id, text) in enumerate(source):
        end = offset + len(text)
        results.append({"claim_id": claim_id, "sentence_id": f"s{index + 1}", "text": text,
                        "start": offset, "end": end, "quote": text, "proposition": text,
                        "sentence_start": offset, "sentence_end": end, "kind": "result",
                        "requirements": ["observed result and scope"], "support_query": text,
                        "counter_query": "limitation exception subgroup " + text})
        offset = end + 1
    return results


def blind_evidence(evidence: list[dict]) -> list[dict]:
    permitted = {"evidence_id", "doc_id", "kind", "text", "locator", "row_header", "column_header"}
    return [{key: item[key] for key in sorted(item.keys() & permitted)}
            for item in sorted(evidence, key=lambda item: item["evidence_id"])]


def validate_ids(ids: list[str], evidence: list[dict]) -> None:
    allowed = {item["evidence_id"] for item in evidence}
    if not allowed.issuperset(ids):
        raise ModelUnavailable("model cited evidence outside the supplied scope")


def _pending(claim: dict, reason: str, *, status: str = "pending") -> dict:
    return {**claim, "verdict": None, "review_status": status, "reason": reason,
            "evidence_ids": [], "evidence_groups": [], "missing_fields": claim["requirements"]}


def _offline(claim: dict, candidates: list[dict], comparison) -> dict:
    if claim["kind"] in {"background", "purpose"}:
        return _pending(claim, "This sentence is outside original-paper result verification.")
    if comparison is not None:
        calculation, expected = comparison
        matches = Decimal(calculation["result"]) == expected
        ids = calculation["evidence_ids"]
        return {**claim, "review_status": "completed", "verdict": "supported" if matches else "refuted",
                "reason": "The scoped table operands were independently recomputed.", "evidence_ids": ids,
                "evidence_groups": [{"role": "support" if matches else "refute", "evidence_ids": ids}],
                "missing_fields": []}
    quote = " ".join(claim["quote"].split()).strip(".。")
    matches = [item for item in candidates if item.get("kind") == "body_text"
               and quote and quote == " ".join(item["text"].split()).strip(".。")]
    if matches:
        ids = [matches[0]["evidence_id"]]
        return {**claim, "review_status": "completed", "verdict": "supported",
                "reason": "The supplied wording occurs verbatim in the paper body.", "evidence_ids": ids,
                "evidence_groups": [{"role": "support", "evidence_ids": ids}], "missing_fields": []}
    return _pending(claim, "The offline reviewer could not determine a verdict from the located evidence.")


def _figure_images(candidates: list[dict], documents: list[dict], data_dir: Path) -> list[dict]:
    import pymupdf

    paths = {item["doc_id"]: item["blob_path"] for item in documents}
    figures = []
    for evidence in (row for row in candidates if row["kind"] == "figure"):
        archive = paths.get(evidence["doc_id"])
        if archive is None:
            continue
        source = (data_dir / archive).resolve()
        if not source.is_relative_to(data_dir.resolve()):
            continue
        try:
            with pymupdf.open(source) as pdf:
                page = pdf[evidence["locator"]["page_index"]]
                page.set_rotation(0)
                rect = pymupdf.Rect(evidence["locator"]["bbox"])
                scale = min(2, 1400 / max(rect.width, rect.height))
                image = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=rect, alpha=False)
                data = b64encode(image.tobytes("png")).decode("ascii")
        except (OSError, RuntimeError, ValueError, IndexError):
            continue
        figures.append({"evidence_id": evidence["evidence_id"], "image_url": f"data:image/png;base64,{data}"})
        if len(figures) == 2:
            break
    return figures


def review_claim(
    claim: dict, index: EvidenceIndex, model=None, *, documents=(), data_dir=None,
    reserve_followup: Callable[[], bool] | None = None, additional_evidence: Sequence[dict] = (),
    saved_targets: Sequence[dict] = (), save_targets: Callable[[list[dict]], None] | None = None,
) -> tuple[dict, dict | None]:
    documents = tuple(documents)
    requirements = [field for field in claim["requirements"] if field.strip()]
    if not requirements:
        requirements = ["observed result and scope"]
        claim = {**claim, "requirements": requirements}
    examined_scopes = []
    paths = []
    by_id = {}
    breadth = []
    prioritized = []
    sent_candidate_ids = []
    output = None
    stop = None

    def remember(result: dict) -> None:
        paths.extend(result["paths"])
        breadth.append([item["evidence_id"] for item in result["evidence"]])
        by_id.update((item["evidence_id"], item) for item in result["evidence"])

    def gather(queries: Sequence[str]) -> None:
        for query in dict.fromkeys(queries):
            if not query.strip():
                continue
            result = index.search(query, limit=8)
            examined_scopes.append({**result["scope"], "mode": "ranked"})
            remember(result)

    def inspect(targets: Sequence[dict]) -> bool:
        known = set(by_id)
        results = targeted_evidence(index, targets, documents=documents)
        examined_scopes.extend(results["scopes"])
        remember(results)
        prioritized.extend(item["evidence_id"] for item in results["evidence"])
        return bool(set(by_id) - known) or any(scope.get("complete") for scope in results["scopes"])

    gather([claim["support_query"], claim["counter_query"]])
    if additional_evidence:
        supplements = index.get(item["evidence_id"] for item in additional_evidence)
        prioritized.extend(item["evidence_id"] for item in supplements)
        by_id.update((item["evidence_id"], item) for item in supplements)
    followup = bool(saved_targets or additional_evidence)
    if saved_targets:
        inspect(saved_targets)
    calculation = None
    while True:
        # Named inspections take priority; interleave remaining support and counter searches.
        interleaved = (eid for tier in range(8) for column in breadth for eid in column[tier:tier + 1])
        balanced_ids = list(dict.fromkeys((*prioritized, *interleaved)))
        candidates = [by_id[eid] for eid in balanced_ids[:8]]
        comparison = comparison_for_claim(claim["proposition"], candidates)
        calculation = comparison[0] if comparison else None
        if model is None:
            finding = _offline(claim, candidates, comparison)
            sent_candidate_ids = [item["evidence_id"] for item in candidates]
            break
        if claim["kind"] in {"background", "purpose"}:
            finding = _pending(claim, "This sentence is outside original-paper result verification.")
            break
        material = {"claim": claim["proposition"], "requirements": requirements,
                    "evidence": blind_evidence(candidates), "calculation": calculation,
                    "follow_up_already_performed": followup,
                    "direct_inspections": [scope for scope in examined_scopes if scope.get("mode") == "direct"],
                    "coverage_instructions": (
                        "For each required field, report its status; found requires supplied evidence IDs. "
                        "Ordinary search is not a complete inspection. For missing fields, request up to two concrete "
                        "follow_up_targets tied to the exact field name, with a section, table label or page when known."
                    )}
        if data_dir and any(item["kind"] == "figure" for item in candidates):
            material["figure_images"] = _figure_images(candidates, documents, data_dir)
        try:
            output = model.call("judge", material, ScopedJudgement)
            sent_candidate_ids = [item["evidence_id"] for item in material["evidence"]]
            ids = list(dict.fromkeys(eid for group in output.evidence_groups for eid in group.evidence_ids))
            try:
                validate_ids(ids, candidates)
            except ModelUnavailable:
                output = model.call("judge", {**material, "retry": "Review the supplied IDs carefully."}, ScopedJudgement)
                ids = list(dict.fromkeys(eid for group in output.evidence_groups for eid in group.evidence_ids))
                validate_ids(ids, candidates)
        except ModelUnavailable as exc:
            finding = _pending(claim, str(exc))
            finding["retryable"] = isinstance(exc, ModelPaused)
            break

        targets = _pending_targets(output, requirements, documents)
        if targets and not followup:
            if reserve_followup is None or reserve_followup():
                followup = True
                if save_targets is not None:
                    save_targets(targets)
                if inspect(targets):
                    continue
                stop = "no_progress"
            else:
                stop = "follow_up_budget"
        coverage = _coverage(requirements, output, examined_scopes, sent_candidate_ids)
        verdict = output.verdict
        role = {"supported": "support", "refuted": "refute"}.get(verdict)
        has_evidence = any(group.role == role and group.evidence_ids for group in output.evidence_groups)
        conflicting = {group.role for group in output.evidence_groups if group.evidence_ids}
        checked = {field["status"] for field in coverage}
        follow_up_requested = bool(output.follow_up_queries or output.follow_up_targets)
        decisive_fields_checked = checked <= {"found"} and not follow_up_requested
        can_call_insufficient = (
            "searched_missing" in checked and checked <= {"found", "searched_missing"}
            and not follow_up_requested
        )
        if ((role and not (has_evidence and decisive_fields_checked))
                or (verdict == "insufficient" and not can_call_insufficient)
                or (verdict == "manual_conflict" and not {"support", "refute"} <= conflicting)):
            verdict = "pending"
        status = verdict if verdict in {"pending", "manual_conflict"} else "completed"
        if (output.verdict == "insufficient" or (role and not decisive_fields_checked)) and verdict == "pending":
            reason = f"Required sources have not been fully checked. Proposed assessment: {output.reason}"
        else:
            reason = output.reason
        finding = {**claim, "review_status": status, "verdict": verdict if status == "completed" else None,
                   "reason": reason, "evidence_ids": ids,
                   "evidence_groups": [group.model_dump() for group in output.evidence_groups],
                   "missing_fields": output.missing_fields, "coverage": coverage}
        break
    candidate_ids = [item["evidence_id"] for item in candidates]
    finding["candidate_ids"] = candidate_ids
    finding["sent_candidate_ids"] = sent_candidate_ids
    finding["searched_scope"] = examined_scopes
    sent = set(sent_candidate_ids)
    finding["retrieval_paths"] = [path for path in paths if sent.issuperset(path["evidence_ids"])]
    finding["follow_up_performed"] = followup
    if "coverage" not in finding:
        offline_ids = finding["evidence_ids"] if model is None and finding["review_status"] == "completed" else ()
        finding["coverage"] = _coverage(requirements, None, examined_scopes, sent_candidate_ids, offline_ids)
    if finding["verdict"] == "insufficient":
        stop = "source_range_exhausted"
    elif finding["review_status"] == "completed":
        stop = "verdict"
    elif any(field["status"] == "unreadable" for field in finding["coverage"]):
        stop = "unreadable"
    finding["stop_reason"] = stop or "requires_review"
    return finding, calculation


def audit_version(
    report: dict, model, commit, *, embedder=None, data_dir=None,
    reserve_followup: Callable[[str], bool] | None = None,
) -> None:
    version = report["reviewed_abstract"]
    if "claims" not in report:
        report["claims"] = extract_claims(version["text"], model)
        commit()
    index = EvidenceIndex(report["evidence"], embedder=embedder)
    followups = report.setdefault("claim_followups", {})
    try:
        done = {finding["claim_id"] for finding in report["findings"] if not finding.get("retryable")}
        for claim in report["claims"]:
            claim_id = claim["claim_id"]
            if claim_id in done:
                continue

            def save_targets(targets: list[dict]) -> None:
                followups[claim_id] = targets
                commit()

            finding, calculation = review_claim(
                claim, index, model, documents=report.get("documents", ()), data_dir=data_dir,
                reserve_followup=(lambda: reserve_followup(claim_id)) if reserve_followup is not None else None,
                saved_targets=followups.get(claim_id, ()), save_targets=save_targets,
            )
            report["findings"] = [row for row in report["findings"] if row["claim_id"] != claim_id] + [finding]
            report["calculations"] = [row for row in report["calculations"] if row["claim_id"] != claim_id]
            if calculation is not None:
                report["calculations"].append({**calculation, "claim_id": claim_id})
            commit()
    finally:
        close = getattr(index, "close", None)
        if close is not None:
            close()
    report["unresolved_items"] = [{"claim_id": row["claim_id"], "reason": row["reason"]}
                                  for row in report["findings"]
                                  if row["review_status"] != "completed" or row["verdict"] == "insufficient"]
