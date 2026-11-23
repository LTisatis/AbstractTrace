"""Evidence-backed replacements and independent checks of an abstract version."""

import hashlib
import json
import re
from decimal import Decimal
from difflib import SequenceMatcher
from typing import Literal

from pydantic import Field

from .audit import blind_evidence, sentences, validate_ids
from .calculations import check_revision_numbers, comparison_for_claim
from .evidence import EvidenceIndex, with_method_scope
from .followups import FollowUps
from .models import ROLE_PROMPTS, ModelPaused, ModelUnavailable, Schema, VerificationQuery


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
    follow_up_queries: list[VerificationQuery] = Field(default_factory=list)


class SentenceVerification(SentenceCheck):
    calculations: list[dict] = Field(default_factory=list)


class AbstractIssue(Schema):
    sentence_ids: list[str]
    reason: str
    evidence_ids: list[str]
    follow_up_queries: list[VerificationQuery] = Field(default_factory=list)


class AbstractCheck(Schema):
    status: Literal["passed", "returned", "manual_review"]
    reason: str
    evidence_ids: list[str]
    issues: list[AbstractIssue]


_POINTS = re.compile(r"(?P<value>[+−-]?\d+(?:\.\d+)?)\s*(?:percentage points?\b|个?百分点|pp\b)", re.I)
_RELATIVE = re.compile(
    r"(?:relative(?:ly)?\s+(?:increase\s+of\s+)?|相对(?:提高|增加)?)\s*(?P<value>\d+(?:\.\d+)?)\s*%",
    re.I,
)


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _call_step(model, stage: str, material: dict, schema, followups: FollowUps | None, target):
    if followups is None:
        return model.call(stage, material, schema)
    steps = followups.report.setdefault("revision_model_steps", {})
    key = _digest([stage, target, ROLE_PROMPTS[stage], schema.model_json_schema(), material])
    if key not in steps:
        result = model.call(stage, material, schema)
        steps[key] = result.model_dump(mode="json")
        followups.checkpoint()
        return result
    return schema.model_validate(steps[key])


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


def _scope(report: dict, findings: list[dict], additional_ids=(), *, additional_paths=()) -> list[dict]:
    ids = set(additional_ids)
    paths = list(additional_paths)
    for item in findings:
        ids.update(item.get("candidate_ids", ()))
        ids.update(item.get("sent_candidate_ids", ()))
        ids.update(item.get("evidence_ids", ()))
        paths.extend(item.get("retrieval_paths", ()))
    scope = {item["evidence_id"]: item for item in report["evidence"] if item["evidence_id"] in ids}
    for check in report.get("external_checks", ()):
        if check.get("support_status") != "not_supported":
            continue
        reviewed = ids.intersection(check.get("evidence_ids", ()))
        for item in check.get("evidence", ()):
            if item["evidence_id"] in reviewed:
                scope.setdefault(item["evidence_id"], item)
    return with_method_scope(scope.values(), paths)


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


def _followup_keys(original: dict, findings: list[dict], start: int, end: int, *, insertion: str = "") -> list[str]:
    claim_ids = [row["claim_id"] for row in findings]
    return claim_ids or [f"position:{original['version_id']}:{start}:{end}:{insertion}"]


def _verify_numbers(check: SentenceCheck, source: str, replacement: str, evidence: list[dict]) -> SentenceVerification:
    calculations = check_revision_numbers(source, replacement, evidence)
    result = check.model_dump()
    mismatch = next((item for item in calculations if item["verification"] == "mismatch"), None)
    if mismatch:
        result.update(status="returned", reason=mismatch.get("reason") or (
            f"Recalculation gave {mismatch['result']} {mismatch['unit']}, but the revision states {mismatch['stated']}."
        ))
    elif check.status == "passed" and (unresolved := next((item for item in calculations if item["verification"] == "unresolved"), None)):
        result.update(status="manual_review", reason=unresolved["reason"])
    if mismatch or result["status"] == "passed":
        ids = [eid for item in calculations for eid in [*item["evidence_ids"], *item.get("scope_evidence_ids", ())]]
        result["evidence_ids"] = list(dict.fromkeys([*result["evidence_ids"], *ids]))
    return SentenceVerification(**result, calculations=calculations)


def _verify_sentence(source: str, replacement: str, evidence: list[dict], model, *,
                     followups: FollowUps | None = None, keys: list[str] | None = None, target=None) -> SentenceVerification:
    if model is None:
        correction = _numeric_draft(source, evidence)
        if correction is not None and replacement == correction.replacement:
            result = SentenceCheck(status="passed", reason="The scoped replacement value was recomputed.",
                                   evidence_ids=correction.evidence_ids)
        else:
            result = SentenceCheck(status="manual_review", reason="No offline check verified this replacement.", evidence_ids=[])
        return _verify_numbers(result, source, replacement, evidence)
    try:
        if followups is not None:
            evidence = [*evidence, *followups.material(keys, evidence)]
        material = {"original_sentence": source, "revised_sentence": replacement}

        def check(rows: list[dict]) -> SentenceCheck:
            output = _call_step(model, "verify_sentence", {**material, "evidence": blind_evidence(rows)}, SentenceCheck,
                                followups, target)
            validate_ids(output.evidence_ids, rows)
            return output

        output = check(evidence)
        if output.follow_up_queries:
            added = followups.collect(keys, output.follow_up_queries, evidence, "verify_sentence") if followups else []
            if added:
                evidence.extend(added)
                output = check(evidence)
            if output.follow_up_queries:
                output = output.model_copy(update={"status": "manual_review"})
        if output.status == "passed" and not output.evidence_ids:
            raise ModelUnavailable("sentence verification supplied no evidence")
    except ModelPaused:
        raise
    except ModelUnavailable as exc:
        output = SentenceCheck(status="manual_review", reason=str(exc), evidence_ids=[])
    return _verify_numbers(output, source, replacement, evidence)


def _validate_draft(draft: RevisionDraft, source: str, evidence: list[dict]) -> None:
    changes = [change for change in draft.semantic_diff if change.kind != "retained"]
    if not draft.replacement.strip() or draft.replacement.strip() == source or not changes or not draft.evidence_ids:
        raise ModelUnavailable("no evidence-backed sentence replacement was supplied")
    validate_ids(draft.evidence_ids, evidence)
    if any(not change.evidence_ids or change.before == change.after for change in changes):
        raise ModelUnavailable("a semantic change has no supporting evidence or no changed value")
    validate_ids([eid for change in draft.semantic_diff for eid in change.evidence_ids], evidence)


def _review_proposal(part: dict, findings: list[dict], evidence: list[dict], model, *,
                     followups: FollowUps, keys: list[str], target: str) -> tuple[RevisionDraft | None, list[SentenceCheck]]:
    source = part["text"]
    attempts = []
    material = {"original_sentence": source, "evidence": blind_evidence(evidence), "findings": [
        {key: row[key] for key in ("text", "proposition", "verdict", "reason", "evidence_ids", "extension") if key in row}
        for row in findings
    ]}
    draft = None
    for attempt in range(2 if model is not None else 1):
        if attempt:
            evidence = [*evidence, *followups.material(keys, evidence)]
        material["evidence"] = blind_evidence(evidence)
        try:
            draft = _numeric_draft(source, evidence) if model is None else _call_step(
                model, "revise", material, RevisionDraft, followups, [target, attempt]
            )
            if draft is None:
                break
            _validate_draft(draft, source, evidence)
        except ModelPaused:
            raise
        except ModelUnavailable as exc:
            return None, [*attempts, SentenceCheck(status="manual_review", reason=str(exc), evidence_ids=[])]
        check = _verify_sentence(source, draft.replacement, evidence, model, followups=followups, keys=keys,
                                 target=[target, attempt])
        attempts.append(check)
        if check.status != "returned" or attempt:
            break
        material = {**material, "previous_replacement": draft.replacement,
                    "verification_feedback": check.model_dump(exclude={"calculations"})}
    return draft, attempts


def _extensions(report: dict, original: dict) -> list[dict]:
    parts = {part["sentence_id"]: part for part in sentences(original["text"])}
    rows = []
    for finding in report.get("omissions", ()):
        anchor = parts.get(finding.get("anchor_sentence_id"))
        ids = finding.get("evidence_ids", ())
        if finding.get("status") == "possible_omission" and anchor and ids and len(_scope(report, [], ids)) == len(set(ids)):
            rows.append((anchor, finding, ids, "omission"))
    for finding in report.get("external_checks", ()):
        anchor = finding.get("attribution", {})
        ids = finding.get("evidence_ids", ())
        if (finding.get("kind") == "explicit_citation" and finding.get("support_status") == "not_supported"
                and parts.get(anchor.get("sentence_id")) == anchor and ids and len(_scope(report, [], ids)) == len(set(ids))):
            rows.append((anchor, finding, ids, "external_citation"))
    return [
        {**part, "claim_id": f"{kind}:{_digest([original['version_id'], part['sentence_id'], ids])[:24]}",
         "extension": kind, "verdict": "refuted" if kind == "external_citation" else None,
         "reason": finding["reason"], "evidence_ids": list(ids)}
        for part, finding, ids, kind in rows
    ]


def _sentence_findings(text: str, findings: list[dict], compounds: dict[str, dict]) -> list[tuple[dict, list[dict], list[dict]]]:
    grouped = []
    for part in sentences(text):
        rows = [item for item in findings if (
            item.get("sentence_id") == part["sentence_id"]
            and item.get("start", part["start"]) >= part["start"]
            and item.get("end", part["end"]) <= part["end"]
        )]
        issues = [item for item in rows if _has_revision_reason(item, compounds)]
        if any(item.get("evidence_ids") for item in issues):
            grouped.append((part, rows, issues))
    return grouped


def _has_revision_reason(finding: dict, compounds: dict[str, dict]) -> bool:
    if finding.get("extension"):
        return True
    revisable_verdicts = {"refuted", "insufficient"}
    if finding.get("review_status") != "completed" or finding.get("verdict") not in revisable_verdicts:
        return False
    if membership := finding.get("compound"):
        group = compounds.get(membership["compound_id"], {})
        return group.get("review_status") == "completed" and group.get("verdict") in revisable_verdicts
    return True


def build_candidate(report: dict, model=None, *, followups: FollowUps | None = None) -> dict | None:
    """Record proposed replacements, then merge only independently checked sentences."""
    original = report["original_abstract"]
    revisions = report.setdefault("revisions", [])
    verifications = report.setdefault("verifications", [])
    unresolved = report.setdefault("unresolved_items", [])
    followups = followups or FollowUps(report)
    passed = []
    extensions = _extensions(report, original) if model is not None else []
    steps = report.setdefault("revision_steps", {})
    groups = {group["compound_id"]: group for group in report.get("compound_findings", ())}
    for part, rows, scoped in _sentence_findings(original["text"], [*report["findings"], *extensions], groups):
        evidence = _scope(report, rows)
        step = _digest([original["version_id"], part, rows, [row["evidence_id"] for row in evidence]])
        if step in steps:
            prior = next((row for row in revisions if row["revision_id"] == steps[step]), None)
            if prior is not None and prior["status"] == "passed":
                keys = _followup_keys(original, [row for row in rows if not row.get("extension")], part["start"], part["end"])
                evidence.extend(followups.material(keys, evidence))
                prior_check = _verify_numbers(SentenceCheck(status="passed", reason="", evidence_ids=prior["evidence_ids"]),
                                              part["text"], prior["replacement"], evidence)
                if prior_check.status == "passed":
                    passed.append(prior)
            continue
        if model is None and len(scoped) != 1:
            draft, attempts = None, []
        else:
            keys = _followup_keys(original, [row for row in rows if not row.get("extension")], part["start"], part["end"])
            draft, attempts = _review_proposal(part, rows, evidence, model, followups=followups, keys=keys, target=step)
        if draft is None:
            reason = attempts[-1].reason if attempts else "A draft could not be grounded in local evidence."
            unresolved.append({"code": "revision_manual_review", "sentence_id": part["sentence_id"], "reason": reason})
            steps[step] = None
            followups.checkpoint()
            continue
        check = attempts[-1]
        revision_id = _digest([original["version_id"], part["start"], part["end"], draft.replacement])[:24]
        if any(row["revision_id"] == revision_id for row in revisions):
            revision_id = _digest([revision_id, step])[:24]
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
        steps[step] = revision_id
        if check.status == "passed":
            passed.append(patch)
        else:
            unresolved.append({"code": "revision_manual_review", "revision_id": revision_id, "reason": check.reason})
        followups.checkpoint()
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


def repair_candidate(report: dict, candidate: dict, whole_review: dict, model, *,
                     followups: FollowUps | None = None) -> dict | None:
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
    followups = followups or FollowUps(report)
    history = report["verifications"]
    replacements = {}
    for patch in patches:
        patch_id = patch["revision_id"]
        if patch_id not in targets:
            continue
        checks = [record for record in history if record.get("original_version_id") == original["version_id"]
                  and (record.get("start"), record.get("end")) == (patch["start"], patch["end"])]
        attempt = next((record for record in checks if record.get("scope") == "whole_repair"
                        and record.get("candidate_version_id") == candidate["version_id"]
                        and record.get("revision_id") == patch_id), None)
        if attempt is not None and attempt["status"] != "attempted":
            if "patch" in attempt:
                replacements[patch_id] = (attempt["patch"], attempt["verification"])
            continue
        if (any(record is not attempt and record.get("repair_count", 0) >= 1 for record in checks)
                or not any(record.get("revision_id") == patch_id and record.get("status") == "passed"
                           and record.get("scope") == "sentence" for record in checks)):
            continue

        issues = targets[patch_id]
        feedback_ids = [eid for issue in issues for eid in issue.get("evidence_ids", [])]
        finding_rows = [row for row in report["findings"] if row.get("sentence_id") == patch["sentence_id"]]
        keys = _followup_keys(original, finding_rows, patch["start"], patch["end"])
        evidence = _scope(report, finding_rows, [*patch["evidence_ids"], *feedback_ids])
        evidence = [*evidence, *followups.material(keys, evidence)]
        try:
            validate_ids([*patch["evidence_ids"], *feedback_ids], evidence)
        except ModelUnavailable:
            continue

        anchor = {"original_version_id": original["version_id"], "start": patch["start"], "end": patch["end"], "repair_count": 1}
        if attempt is None:
            attempt = {**anchor, "scope": "whole_repair", "revision_id": patch_id,
                       "candidate_version_id": candidate["version_id"], "status": "attempted"}
            history.append(attempt)
            followups.checkpoint()
        target = [candidate["version_id"], patch_id]
        try:
            draft = _call_step(model, "revise", {"original_sentence": patch["original_text"],
                "previous_replacement": patch["replacement"], "verification_feedback": issues,
                "evidence": blind_evidence(evidence)}, RevisionDraft, followups, target)
            _validate_draft(draft, patch["original_text"], evidence)
            if draft.replacement.strip() == patch["replacement"].strip():
                raise ModelUnavailable("whole-abstract repair did not change the replacement")
        except ModelPaused:
            raise
        except ModelUnavailable as exc:
            attempt.update(status="manual_review", reason=str(exc))
            followups.checkpoint()
            continue

        check = _verify_sentence(patch["original_text"], draft.replacement, evidence, model,
                                 followups=followups, keys=keys, target=target)
        attempt.update(status=check.status, reason=check.reason, calculations=check.calculations)
        if check.status != "passed":
            followups.checkpoint()
            continue
        revision_id = _digest([original["version_id"], patch["start"], patch["end"], draft.replacement])[:24]
        revised_patch = {**patch, "revision_id": revision_id, "replacement": draft.replacement,
                         "evidence_ids": draft.evidence_ids, "replaces_revision_id": patch_id,
                         "semantic_diff": [change.model_dump() for change in draft.semantic_diff]}
        verification = {**anchor, "scope": "sentence", "revision_id": revision_id,
                        "original_sentence": patch["original_text"], "revised_sentence": draft.replacement,
                        **check.model_dump()}
        attempt.update(patch=revised_patch, verification=verification)
        followups.checkpoint()
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


def _aligned_spans(original: dict, revised: dict, *, include_equal: bool = False):
    left, right = sentences(original["text"]), sentences(revised["text"])
    matching = SequenceMatcher(a=[part["text"] for part in left], b=[part["text"] for part in right], autojunk=False)
    for tag, i, j, k, m in matching.get_opcodes():
        if tag == "equal" and not include_equal:
            continue
        spans = [(i, j, k, m)]
        if j - i == m - k:
            spans = [(a, a + 1, b, b + 1) for a, b in zip(range(i, j), range(k, m))]
        for a, b, c, d in spans:
            yield _location(left, a, b, original["text"]), _location(right, c, d, revised["text"])


def _location_keys(report: dict, original: dict, revised: dict, source: dict, target: dict) -> list[str]:
    findings = [row for row in report["findings"] if row.get("sentence_id") in source["sentence_ids"]]
    insertion = f"{revised['version_id']}:{target['start']}" if source["start"] == source["end"] else ""
    return _followup_keys(original, findings, source["start"], source["end"], insertion=insertion)


def _check_changes(report: dict, original: dict, revised: dict, model, followups: FollowUps) -> list[dict]:
    known = {row["revision_id"]: row for row in report.get("verifications", []) if row.get("scope") == "sentence"
             and row.get("original_version_id") == original["version_id"] and row["status"] == "passed"}
    reusable = [known[rid] for rid in revised["revision_ids"] if rid in known]
    index = None
    checks = []
    try:
        for source, target in _aligned_spans(original, revised):
            reviewed = next((row for row in reusable if row["start"] == source["start"]
                             and row["end"] == source["end"] and row["revised_sentence"] == target["text"]), None)
            findings = [row for row in report["findings"] if row.get("sentence_id") in source["sentence_ids"]]
            keys = _location_keys(report, original, revised, source, target)
            if reviewed is not None and revised["kind"] == "candidate":
                evidence = _scope(report, findings, reviewed["evidence_ids"])
                evidence.extend(followups.material(keys, evidence))
                check = SentenceCheck(**{key: reviewed[key] for key in ("status", "reason", "evidence_ids")})
                result = _verify_numbers(check, source["text"], target["text"], evidence).model_dump()
            else:
                if index is None:
                    index = EvidenceIndex(report["evidence"])
                related = index.search(source["text"] + " " + target["text"], limit=8)
                related_ids = [item["evidence_id"] for item in related["evidence"]]
                context = index.table_context(related_ids)
                evidence = _scope(report, findings, [*related_ids, *(item["evidence_id"] for item in context["evidence"])],
                                  additional_paths=related["paths"])
                finding_revision = len(report.get("finding_history", []))
                result = _verify_sentence(source["text"], target["text"], evidence, model,
                                          followups=followups, keys=keys,
                                          target=[revised["version_id"], source, target, finding_revision]).model_dump()
            checks.append({"version_id": revised["version_id"], "original": source, "revised": target, **result})
    finally:
        if index is not None:
            index.close()
    return checks


def verify_version(report: dict, original: dict, revised: dict, model=None, *,
                   followups: FollowUps | None = None) -> dict:
    """Check this exact version, including cross-sentence constraints and all changed text."""
    followups = followups or FollowUps(report)
    checks = _check_changes(report, original, revised, model, followups)
    summary = {"version_id": revised["version_id"], "sentence_checks": checks, "evidence_ids": [], "issues": []}
    if model is None:
        return {**summary, "status": "manual_review", "reason": "Offline checks do not establish whole-abstract consistency."}
    by_sentence = {part["sentence_id"]: part for part in sentences(revised["text"])}
    keys_by_sentence = {
        sid: _location_keys(report, original, revised, source, target)
        for source, target in _aligned_spans(original, revised, include_equal=True) for sid in target["sentence_ids"]
    }
    ids = [eid for item in checks for eid in item["evidence_ids"]]
    evidence = _scope(report, report["findings"], ids)
    all_keys = list(dict.fromkeys(key for keys in keys_by_sentence.values() for key in keys))
    evidence.extend(followups.material(all_keys, evidence))

    def check(rows: list[dict]) -> AbstractCheck:
        result = _call_step(model, "verify_abstract", {
            "version_id": revised["version_id"], "original_abstract": original["text"],
            "revised_abstract": revised["text"], "revised_sentences": list(by_sentence.values()),
            "evidence": blind_evidence(rows), "unresolved_items": report.get("unresolved_items", []),
        }, AbstractCheck, followups, [revised["version_id"], len(report.get("finding_history", []))])
        validate_ids(result.evidence_ids, rows)
        for item in result.issues:
            validate_ids(item.evidence_ids, rows)
            if not item.sentence_ids or not by_sentence.keys() >= set(item.sentence_ids):
                raise ModelUnavailable("whole-abstract verification did not supply valid sentence locations")
        return result

    try:
        result = check(evidence)
        added = []
        for item in result.issues:
            keys = list(dict.fromkeys(key for sid in item.sentence_ids for key in keys_by_sentence.get(sid, ())))
            added.extend(followups.collect(keys, item.follow_up_queries, [*evidence, *added], "verify_abstract"))
        if added:
            result = check([*evidence, *added])
        issues = [{**item.model_dump(), "locations": [by_sentence[sid] for sid in item.sentence_ids]}
                  for item in result.issues]
    except ModelPaused:
        raise
    except ModelUnavailable as exc:
        return {**summary, "status": "manual_review", "reason": str(exc)}
    status = result.status
    if any(item.follow_up_queries for item in result.issues):
        status = "manual_review"
    if issues and status == "passed":
        status = "returned"
    if status == "passed" and any(check["status"] != "passed" for check in checks):
        return {**summary, "evidence_ids": result.evidence_ids, "status": "manual_review",
                "reason": "One or more changed sentences require further verification."}
    return {**summary, "evidence_ids": result.evidence_ids, "issues": issues, "status": status, "reason": result.reason}
