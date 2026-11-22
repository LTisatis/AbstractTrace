"""Evidence-backed replacements and independent checks of an abstract version."""

import hashlib
import json
import re
from decimal import Decimal
from difflib import SequenceMatcher
from typing import Literal

from .audit import blind_evidence, sentences, validate_ids
from .calculations import comparison_for_claim
from .evidence import EvidenceIndex
from .models import ModelUnavailable, Schema


class SemanticChange(Schema):
    kind: Literal["retained", "modified", "added", "removed"]
    field: str
    before: str | None
    after: str | None
    evidence_ids: list[str]


class RevisionDraft(Schema):
    replacement: str
    evidence_ids: list[str]
    semantic_diff: list[SemanticChange]


class SentenceCheck(Schema):
    status: Literal["passed", "returned", "manual_review"]
    reason: str
    evidence_ids: list[str]


class AbstractIssue(Schema):
    sentence_ids: list[str]
    reason: str
    evidence_ids: list[str]


class AbstractCheck(Schema):
    status: Literal["passed", "returned", "manual_review"]
    reason: str
    evidence_ids: list[str]
    issues: list[AbstractIssue]


_POINTS = re.compile(r"(?P<value>[+−-]?\d+(?:\.\d+)?)\s*(?:percentage points?\b|百分点)", re.I)
_RELATIVE = re.compile(
    r"(?:relative(?:ly)?\s+(?:increase\s+of\s+)?|相对(?:提高|增加)?)\s*(?P<value>\d+(?:\.\d+)?)\s*%",
    re.I,
)


def _digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def new_version(text: str, kind: str, parent_version_id: str, revision_ids: list[str]) -> dict:
    ids = list(revision_ids)
    return {"version_id": _digest([kind, parent_version_id, text, ids]), "kind": kind,
            "text": text, "revision_ids": ids, "parent_version_id": parent_version_id}


def apply_revisions(original_text: str, revisions: list[dict]) -> str:
    """Apply disjoint replacements against the exact stored original text."""
    parts = []
    offset = 0
    for item in sorted(revisions, key=lambda revision: revision["start"]):
        start, end = item["start"], item["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(original_text):
            raise ValueError("revision range is outside the original abstract")
        if start < offset:
            raise ValueError("revision ranges overlap")
        if original_text[start:end] != item["original_text"]:
            raise ValueError("revision range does not match the original abstract")
        parts.extend((original_text[offset:start], item["replacement"]))
        offset = end
    return "".join([*parts, original_text[offset:]])


def _scope(report: dict, findings: list[dict], additional_ids=()) -> list[dict]:
    ids = set(additional_ids)
    for item in findings:
        ids.update(item.get("candidate_ids", ()))
        ids.update(item.get("evidence_ids", ()))
    return [item for item in report["evidence"] if item["evidence_id"] in ids]


def _numeric_draft(source: str, evidence: list[dict]) -> RevisionDraft | None:
    comparison = comparison_for_claim(source, evidence)
    if comparison is None:
        return None
    calculation, stated = comparison
    expected = Decimal(calculation["result"])
    if expected == stated or expected == 0 or (expected < 0) != (stated < 0):
        return None
    pattern = _POINTS if calculation["operation"] == "percentage_points" else _RELATIVE
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        return None
    quantity = matches[0]
    value = quantity["value"]
    sign = value[0] if value[0] in "+-−" else ""
    corrected = sign + format(abs(expected), "f")
    replacement = source[:quantity.start("value")] + corrected + source[quantity.end("value"):]
    ids = calculation["evidence_ids"]
    return RevisionDraft(replacement=replacement, evidence_ids=ids, semantic_diff=[
        SemanticChange(kind="modified", field=calculation["unit"], before=value, after=corrected, evidence_ids=ids),
    ])


def _verify_sentence(source: str, replacement: str, evidence: list[dict], model) -> SentenceCheck:
    if model is None:
        correction = _numeric_draft(source, evidence)
        revised = comparison_for_claim(replacement, evidence)
        if correction is not None and replacement == correction.replacement and revised is not None:
            calculation, stated = revised
            if Decimal(calculation["result"]) == stated:
                return SentenceCheck(status="passed", reason="The scoped replacement value was recomputed.",
                                     evidence_ids=calculation["evidence_ids"])
        return SentenceCheck(status="manual_review", reason="No offline check verified this replacement.", evidence_ids=[])
    try:
        output = model.call("verify_sentence", {
            "original_sentence": source, "revised_sentence": replacement, "evidence": blind_evidence(evidence),
        }, SentenceCheck)
        validate_ids(output.evidence_ids, evidence)
        if output.status == "passed" and not output.evidence_ids:
            raise ModelUnavailable("sentence verification supplied no evidence")
        return output
    except ModelUnavailable as exc:
        return SentenceCheck(status="manual_review", reason=str(exc), evidence_ids=[])


def _validate_draft(draft: RevisionDraft, source: str, evidence: list[dict]) -> None:
    changes = [change for change in draft.semantic_diff if change.kind != "retained"]
    if not draft.replacement.strip() or draft.replacement.strip() == source or not changes or not draft.evidence_ids:
        raise ModelUnavailable("no evidence-backed sentence replacement was supplied")
    validate_ids(draft.evidence_ids, evidence)
    if any(not change.evidence_ids or change.before == change.after for change in changes):
        raise ModelUnavailable("a semantic change has no supporting evidence or no changed value")
    validate_ids([eid for change in draft.semantic_diff for eid in change.evidence_ids], evidence)


def _review_proposal(part: dict, findings: list[dict], evidence: list[dict], model) -> tuple[RevisionDraft | None, list[SentenceCheck]]:
    source = part["text"]
    attempts = []
    material = {"original_sentence": source, "evidence": blind_evidence(evidence), "findings": [
        {key: row[key] for key in ("text", "proposition", "verdict", "reason", "evidence_ids") if key in row}
        for row in findings
    ]}
    draft = None
    for attempt in range(2 if model is not None else 1):
        try:
            draft = _numeric_draft(source, evidence) if model is None else model.call("revise", material, RevisionDraft)
            if draft is None:
                break
            _validate_draft(draft, source, evidence)
        except ModelUnavailable as exc:
            return None, [*attempts, SentenceCheck(status="manual_review", reason=str(exc), evidence_ids=[])]
        check = _verify_sentence(source, draft.replacement, evidence, model)
        attempts.append(check)
        if check.status != "returned" or attempt:
            break
        material = {**material, "previous_replacement": draft.replacement,
                    "verification_feedback": check.model_dump()}
    return draft, attempts


def _sentence_findings(text: str, findings: list[dict]) -> list[tuple[dict, list[dict]]]:
    grouped = []
    for part in sentences(text):
        rows = [item for item in findings if (
            item.get("sentence_id") == part["sentence_id"]
            and item.get("start", part["start"]) >= part["start"]
            and item.get("end", part["end"]) <= part["end"]
        )]
        if any(item.get("review_status") == "completed" and item.get("verdict") in {"refuted", "insufficient"}
               and item.get("evidence_ids") for item in rows):
            grouped.append((part, rows))
    return grouped


def build_candidate(report: dict, model=None) -> dict | None:
    """Record proposed replacements, then merge only independently checked sentences."""
    original = report["original_abstract"]
    revisions = report.setdefault("revisions", [])
    verifications = report.setdefault("verifications", [])
    unresolved = report.setdefault("unresolved_items", [])
    passed = []
    for part, rows in _sentence_findings(original["text"], report["findings"]):
        evidence = _scope(report, rows)
        scoped = [row for row in rows if row.get("verdict") in {"refuted", "insufficient"}]
        if model is None and len(scoped) != 1:
            draft, attempts = None, []
        else:
            draft, attempts = _review_proposal(part, rows, evidence, model)
        if draft is None:
            reason = attempts[-1].reason if attempts else "A draft could not be grounded in local evidence."
            unresolved.append({"code": "revision_manual_review", "sentence_id": part["sentence_id"], "reason": reason})
            continue
        check = attempts[-1]
        revision_id = _digest([original["version_id"], part["start"], part["end"], draft.replacement])[:24]
        patch = {"revision_id": revision_id, "sentence_id": part["sentence_id"],
                 "start": part["start"], "end": part["end"], "original_text": part["text"],
                 "replacement": draft.replacement, "evidence_ids": draft.evidence_ids,
                 "semantic_diff": [change.model_dump() for change in draft.semantic_diff],
                 "claim_ids": [item["claim_id"] for item in scoped], "status": check.status}
        verification = {"scope": "sentence", "revision_id": revision_id, "original_version_id": original["version_id"],
                        "start": part["start"], "end": part["end"], "original_sentence": part["text"],
                        "revised_sentence": draft.replacement, "repair_count": len(attempts) - 1, **check.model_dump()}
        revisions.append(patch)
        verifications.append(verification)
        if check.status == "passed":
            passed.append(patch)
        else:
            unresolved.append({"code": "revision_manual_review", "revision_id": revision_id, "reason": check.reason})
    if not passed:
        return None
    return new_version(apply_revisions(original["text"], passed), "candidate", original["version_id"],
                       [item["revision_id"] for item in passed])


def _whole_targets(candidate_text: str, patches: list[dict], issues: list[dict]) -> dict[str, list[dict]]:
    positions = []
    shift = 0
    for patch in patches:
        start = patch["start"] + shift
        end = start + len(patch["replacement"])
        positions.append((start, end, patch["revision_id"]))
        shift = end - patch["end"]
    parts = {part["sentence_id"]: part for part in sentences(candidate_text)}
    targets = {}
    for issue in issues:
        locations = issue.get("locations", [])
        if not locations or any(not isinstance(location, dict) or parts.get(location.get("sentence_id")) != location
                                for location in locations):
            continue
        overlaps = [(start, end, revision_id, location) for start, end, revision_id in positions
                    for location in locations if start < location["end"] and location["start"] < end]
        ids = {revision_id for _, _, revision_id, _ in overlaps}
        if len(ids) == 1 and all(start <= location["start"] and location["end"] <= end
                                 for start, end, _, location in overlaps):
            targets.setdefault(next(iter(ids)), []).append(issue)
    return targets


def repair_candidate(report: dict, candidate: dict, whole_review: dict, model) -> dict | None:
    """Try one repair per unambiguous position, sharing the sentence repair limit."""
    original = report["original_abstract"]
    if (model is None or whole_review.get("status") != "returned"
            or whole_review.get("version_id") != candidate["version_id"]
            or candidate.get("kind") != "candidate" or candidate.get("parent_version_id") != original["version_id"]):
        return None
    by_id = {patch["revision_id"]: patch for patch in report["revisions"] if patch["status"] == "passed"}
    if not candidate["revision_ids"] or not by_id.keys() >= set(candidate["revision_ids"]):
        return None
    patches = sorted((by_id[rid] for rid in candidate["revision_ids"]), key=lambda patch: patch["start"])
    try:
        if apply_revisions(original["text"], patches) != candidate["text"]:
            return None
    except ValueError:
        return None

    targets = _whole_targets(candidate["text"], patches, whole_review.get("issues", []))
    history = report["verifications"]
    replacements = {}
    for patch in patches:
        patch_id = patch["revision_id"]
        if patch_id not in targets:
            continue
        checks = [record for record in history if record.get("original_version_id") == original["version_id"]
                  and (record.get("start"), record.get("end")) == (patch["start"], patch["end"])]
        if (any(record.get("repair_count", 0) >= 1 for record in checks)
                or not any(record.get("revision_id") == patch_id and record.get("status") == "passed" for record in checks)):
            continue

        issues = targets[patch_id]
        feedback_ids = [eid for issue in issues for eid in issue.get("evidence_ids", [])]
        finding_rows = [row for row in report["findings"] if row.get("sentence_id") == patch["sentence_id"]]
        evidence = _scope(report, finding_rows, [*patch["evidence_ids"], *feedback_ids])
        try:
            validate_ids([*patch["evidence_ids"], *feedback_ids], evidence)
        except ModelUnavailable:
            continue

        anchor = {"original_version_id": original["version_id"], "start": patch["start"], "end": patch["end"], "repair_count": 1}
        attempt = {**anchor, "scope": "whole_repair", "revision_id": patch_id,
                   "candidate_version_id": candidate["version_id"], "status": "attempted"}
        history.append(attempt)
        try:
            draft = model.call("revise", {"original_sentence": patch["original_text"],
                "previous_replacement": patch["replacement"], "verification_feedback": issues,
                "evidence": blind_evidence(evidence)}, RevisionDraft)
            _validate_draft(draft, patch["original_text"], evidence)
            if draft.replacement.strip() == patch["replacement"].strip():
                raise ModelUnavailable("whole-abstract repair did not change the replacement")
        except ModelUnavailable as exc:
            attempt.update(status="manual_review", reason=str(exc))
            continue

        check = _verify_sentence(patch["original_text"], draft.replacement, evidence, model)
        attempt.update(status=check.status, reason=check.reason)
        if check.status != "passed":
            continue
        revision_id = _digest([original["version_id"], patch["start"], patch["end"], draft.replacement])[:24]
        revised_patch = {**patch, "revision_id": revision_id, "replacement": draft.replacement,
                         "evidence_ids": draft.evidence_ids, "replaces_revision_id": patch_id,
                         "semantic_diff": [change.model_dump() for change in draft.semantic_diff]}
        verification = {**anchor, "scope": "sentence", "revision_id": revision_id,
                        "original_sentence": patch["original_text"], "revised_sentence": draft.replacement,
                        **check.model_dump()}
        replacements[patch_id] = (revised_patch, verification)

    if not replacements:
        return None
    merged = []
    for patch in patches:
        recovered = replacements.get(patch["revision_id"])
        if recovered is None:
            merged.append(patch)
            continue
        revised_patch, verification = recovered
        patch.update(status="superseded", superseded_by=revised_patch["revision_id"])
        merged.append(revised_patch)
        report["revisions"].append(revised_patch)
        history.append(verification)
    return new_version(apply_revisions(original["text"], merged), "candidate", original["version_id"],
                       [patch["revision_id"] for patch in merged])

def _location(parts: list[dict], start: int, end: int, text: str) -> dict:
    start_offset = parts[start]["start"] if start < len(parts) else len(text)
    end_offset = parts[end - 1]["end"] if end > start else start_offset
    return {"sentence_ids": [part["sentence_id"] for part in parts[start:end]], "start": start_offset,
            "end": end_offset, "text": text[start_offset:end_offset]}


def _check_changes(report: dict, original: dict, revised: dict, model) -> list[dict]:
    left, right = sentences(original["text"]), sentences(revised["text"])
    matching = SequenceMatcher(a=[part["text"] for part in left], b=[part["text"] for part in right], autojunk=False)
    known = {row["revision_id"]: row for row in report.get("verifications", []) if row.get("scope") == "sentence"
             and row.get("original_version_id") == original["version_id"] and row["status"] == "passed"}
    reusable = [known[rid] for rid in revised["revision_ids"] if rid in known]
    index = None
    checks = []
    for tag, i, j, k, m in matching.get_opcodes():
        if tag == "equal":
            continue
        spans = [(i, j, k, m)]
        if j - i == m - k:
            spans = [(a, a + 1, b, b + 1) for a, b in zip(range(i, j), range(k, m))]
        for a, b, c, d in spans:
            source = _location(left, a, b, original["text"])
            target = _location(right, c, d, revised["text"])
            reviewed = next((row for row in reusable if row["start"] == source["start"]
                             and row["end"] == source["end"] and row["revised_sentence"] == target["text"]), None)
            if reviewed is not None and revised["kind"] == "candidate":
                result = {key: reviewed[key] for key in ("status", "reason", "evidence_ids")}
            else:
                if index is None:
                    index = EvidenceIndex(report["evidence"])
                findings = [row for row in report["findings"] if row.get("sentence_id") in source["sentence_ids"]]
                related = index.search(source["text"] + " " + target["text"], limit=8)["evidence"]
                evidence = _scope(report, findings, (item["evidence_id"] for item in related))
                result = _verify_sentence(source["text"], target["text"], evidence, model).model_dump()
            checks.append({"version_id": revised["version_id"], "original": source, "revised": target, **result})
    return checks


def verify_version(report: dict, original: dict, revised: dict, model=None) -> dict:
    """Check this exact version, including cross-sentence constraints and all changed text."""
    checks = _check_changes(report, original, revised, model)
    summary = {"version_id": revised["version_id"], "sentence_checks": checks, "evidence_ids": [], "issues": []}
    if model is None:
        return {**summary, "status": "manual_review", "reason": "Offline checks do not establish whole-abstract consistency."}
    by_sentence = {part["sentence_id"]: part for part in sentences(revised["text"])}
    ids = [eid for item in checks for eid in item["evidence_ids"]]
    evidence = _scope(report, report["findings"], ids)
    try:
        result = model.call("verify_abstract", {
            "version_id": revised["version_id"], "original_abstract": original["text"],
            "revised_abstract": revised["text"], "revised_sentences": list(by_sentence.values()),
            "evidence": blind_evidence(evidence), "unresolved_items": report.get("unresolved_items", []),
        }, AbstractCheck)
        validate_ids(result.evidence_ids, evidence)
        issues = []
        for item in result.issues:
            validate_ids(item.evidence_ids, evidence)
            if not item.sentence_ids or not by_sentence.keys() >= set(item.sentence_ids):
                raise ModelUnavailable("whole-abstract verification did not supply valid sentence locations")
            issues.append({**item.model_dump(), "locations": [by_sentence[sid] for sid in item.sentence_ids]})
    except ModelUnavailable as exc:
        return {**summary, "status": "manual_review", "reason": str(exc)}
    status = result.status
    if issues and status == "passed":
        status = "returned"
    if status == "passed" and any(check["status"] != "passed" for check in checks):
        return {**summary, "evidence_ids": result.evidence_ids, "status": "manual_review",
                "reason": "One or more changed sentences require further verification."}
    return {**summary, "evidence_ids": result.evidence_ids, "issues": issues, "status": status, "reason": result.reason}
