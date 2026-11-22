"""Evidence retrieval, scoped calculations, and independent original verdicts."""

import hashlib
import re
from base64 import b64encode
from decimal import Decimal
from pathlib import Path

from .calculations import comparison_for_claim
from .evidence import EvidenceIndex
from .models import Extraction, Judgement, ModelUnavailable


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


def review_claim(claim: dict, index: EvidenceIndex, model=None, *, documents=(), data_dir=None) -> tuple[dict, dict | None]:
    examined_scopes = []
    paths = []
    by_id = {}
    breadth = []

    def gather(queries: list[str]) -> None:
        for query in dict.fromkeys(queries):
            if not query.strip():
                continue
            result = index.search(query, limit=8)
            examined_scopes.append(result["scope"])
            paths.extend(result["paths"])
            breadth.append([item["evidence_id"] for item in result["evidence"]])
            by_id.update((item["evidence_id"], item) for item in result["evidence"])

    gather([claim["support_query"], claim["counter_query"]])
    followup = False
    calculation = None
    while True:
        # Preserve diversity from both directions while restricting the review pack to eight items.
        columns = list(reversed(breadth)) if followup else breadth
        balanced_ids = list(dict.fromkeys(eid for tier in range(8) for column in columns for eid in column[tier:tier + 1]))
        candidates = [by_id[eid] for eid in balanced_ids[:8]]
        comparison = comparison_for_claim(claim["proposition"], candidates)
        calculation = comparison[0] if comparison else None
        if model is None:
            finding = _offline(claim, candidates, comparison)
            break
        if claim["kind"] in {"background", "purpose"}:
            finding = _pending(claim, "This sentence is outside original-paper result verification.")
            break
        material = {"claim": claim["proposition"], "requirements": claim["requirements"],
                    "evidence": blind_evidence(candidates), "calculation": calculation,
                    "follow_up_already_performed": followup}
        if data_dir and any(item["kind"] == "figure" for item in candidates):
            material["figure_images"] = _figure_images(candidates, documents, data_dir)
        try:
            output = model.call("judge", material, Judgement)
            ids = list(dict.fromkeys(eid for group in output.evidence_groups for eid in group.evidence_ids))
            try:
                validate_ids(ids, candidates)
            except ModelUnavailable:
                output = model.call("judge", {**material, "retry": "Review the supplied IDs carefully."}, Judgement)
                ids = list(dict.fromkeys(eid for group in output.evidence_groups for eid in group.evidence_ids))
                validate_ids(ids, candidates)
        except ModelUnavailable as exc:
            finding = _pending(claim, str(exc))
            finding["retryable"] = True
            break

        if not followup and output.follow_up_queries:
            prior = set(by_id)
            gather(output.follow_up_queries[:2])
            if set(by_id) != prior:
                followup = True
                continue
        verdict = output.verdict
        role = {"supported": "support", "refuted": "refute"}.get(verdict)
        has_evidence = any(group.role == role and group.evidence_ids for group in output.evidence_groups)
        conflicting = {group.role for group in output.evidence_groups if group.evidence_ids}
        if ((role and not has_evidence) or (verdict == "insufficient" and (
              output.follow_up_queries or not any(scope.get("locations") for scope in examined_scopes)
        )) or (verdict == "manual_conflict" and not {"support", "refute"} <= conflicting)):
            verdict = "pending"
        status = verdict if verdict in {"pending", "manual_conflict"} else "completed"
        finding = {**claim, "review_status": status, "verdict": verdict if status == "completed" else None,
                   "reason": output.reason, "evidence_ids": ids,
                   "evidence_groups": [group.model_dump() for group in output.evidence_groups],
                   "missing_fields": output.missing_fields}
        break
    candidate_ids = [item["evidence_id"] for item in candidates]
    finding["candidate_ids"] = candidate_ids
    finding["searched_scope"] = examined_scopes
    finding["retrieval_paths"] = [path for path in paths if path.get("evidence_ids", [None])[-1] in candidate_ids]
    finding["follow_up_performed"] = followup
    finding["stop_reason"] = "verdict" if finding["review_status"] == "completed" else "requires_review"
    return finding, calculation


def audit_version(report: dict, model, commit, *, embedder=None, data_dir=None) -> None:
    version = report["reviewed_abstract"]
    if "claims" not in report:
        report["claims"] = extract_claims(version["text"], model)
        commit()
    index = EvidenceIndex(report["evidence"], embedder=embedder)
    try:
        done = {finding["claim_id"] for finding in report["findings"] if finding["review_status"] != "pending"}
        for claim in report["claims"]:
            if claim["claim_id"] not in done:
                finding, calculation = review_claim(claim, index, model, documents=report["documents"], data_dir=data_dir)
                report["findings"] = [row for row in report["findings"] if row["claim_id"] != claim["claim_id"]] + [finding]
                report["calculations"] = [row for row in report["calculations"] if row["claim_id"] != claim["claim_id"]]
                if calculation is not None:
                    report["calculations"].append({**calculation, "claim_id": claim["claim_id"]})
                commit()
    finally:
        close = getattr(index, "close", None)
        if close is not None:
            close()
    report["unresolved_items"] = [{"claim_id": row["claim_id"], "reason": row["reason"]}
                                  for row in report["findings"] if row["review_status"] != "completed"]
