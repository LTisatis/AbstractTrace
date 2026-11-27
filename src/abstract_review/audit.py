"""Evidence retrieval, scoped calculations, and independent original verdicts."""

import hashlib
import json
import re
import unicodedata
from base64 import b64encode
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from decimal import Decimal
from itertools import zip_longest
from math import isfinite
from pathlib import Path
from typing import Literal

from pydantic import Field

from .calculations import comparison_for_claim
from .compound import (
    adjudicable_shared_exception, aggregate_compound_findings, attach_compound, offline_compound_parts,
    reported_compound_parts, requires_whole_sentence, shared_procedure_parts,
)
from .evidence import EvidenceIndex, _document_key, _inspection_section, _printed_table_label, _terms, with_method_scope
from .models import EvidenceGroup, Extraction, InvalidJudgeReference, Judgement, ModelPaused, ModelUnavailable, Schema
from .source_queries import focused_source_context


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
    source_kind: Literal["methods", "results", "proof"] | None = None


class DirectContradiction(Schema):
    evidence_ids: list[str]
    scope_evidence_ids: list[str] = Field(default_factory=list)
    subject_and_conditions: Literal["same", "different", "uncertain"]
    relation: Literal["opposite_result", "in_scope_counterexample", "uncertain"]
    scope_reason: str
    supplementary_fields: list[str]


class MatchedComparison(Schema):
    focal_condition: str
    comparator_condition: str
    shared_outcome: str
    focal_evidence_ids: list[str]
    comparator_evidence_ids: list[str]
    comparable: Literal["same", "different", "uncertain"]


class EmpiricalSupport(Schema):
    result_source: Literal["direct_result", "summary", "uncertain"]
    result_evidence_ids: list[str]
    scope_evidence_ids: list[str]
    same_study_and_scope: Literal["same", "different", "uncertain"]
    comparison: MatchedComparison | None = None


class CitationRange(Schema):
    role: Literal["support", "refute", "limit"]
    start_evidence_id: str
    end_evidence_id: str


class ScopedJudgement(Judgement):
    coverage: list[FieldCheck] = Field(default_factory=list)
    follow_up_targets: list[FollowUpTarget] = Field(default_factory=list)
    direct_contradiction: DirectContradiction | None = None
    empirical_support: EmpiricalSupport | None = None
    evidence_ranges: list[CitationRange] = Field(default_factory=list)


class CitedAbstractJudgement(ScopedJudgement):
    """A cited abstract exposes individual sentences, not PDF citation ranges."""

    evidence_ranges: None = None


_SCOPE_KEYS = ("section", "table_label", "page_index", "doc_id")
_SOURCE_SECTIONS = {"methods": "Methods", "results": "Results"}
_CITED_ABSTRACT_MAX_SENTENCES = 16
_CITED_ABSTRACT_MAX_CHARS = 5500
_MAX_ORIGINAL_FOLLOWUPS = 2
_CITED_ABSTRACT_INSTRUCTIONS = (
    "Decide from the cited abstract alone. Combine sentences from the same study and recognize paraphrases that "
    "preserve the population, outcome, direction, and strength of the claim. Supported requires evidence for the "
    "claim as stated, including a reported absence when the claim reports that same absence. Refuted requires an "
    "incompatible finding or in-scope counterexample. A different outcome or population, missing data, and a "
    "non-significant estimate without demonstrated equivalence do not by themselves establish contradiction. "
    "For support or refutation, mark the exact supplied requirements found and cite their sentence IDs together "
    "in one group. For refutation, provide direct_contradiction with its same-scope anchors in the refute group. "
    "If the complete abstract does not settle the claim, return insufficient and mark unresolved requirements "
    "searched_missing. If the supplied abstract is incomplete and the evidence does not settle it, return pending. "
    "Leave follow-up fields empty. For consecutive sentences, list each ID explicitly. Set evidence_ranges to null."
)
_EMPIRICAL_RESULT = re.compile(
    r"\b(?:evaluation results?|experimental results?|experiment(?:s)? show|measured (?:effect|performance)|"
    r"observed (?:effect|outcome|failure)|performance (?:gains?|improvement|drop)|outperform\w*|"
    r"(?:accuracy|recall|error rate|latency)\s+(?:increas\w*|decreas\w*|improv\w*|fell|rose))\b"
    r"|评估结果|实验结果|实测|性能增益|优于基线|准确率(?:提高|下降)|误差率(?:提高|下降)", re.I,
)
_FAILURE_SUBJECT = re.compile(r"\b(?:limitations?|failures?|errors?)\b|局限|错误|失败", re.I)
_COMPARATIVE_EMPHASIS = re.compile(
    r"\b(?:especially|particularly|most pronounced)\s+(?:(?:so|pronounced|frequent|common|severe|strong|large|"
    r"high|low|marked|clear)\s+)?(?:in|at|for|under|on|among|with|when)\b"
    r"|(?:尤其|特别)(?:是|在|当|对于)", re.I,
)
_ABSENCE_CONDITION = re.compile(r"\b(?:without|never|neither|no longer)\b|未经过|无需", re.I)
_FAILURE_HEADING = re.compile(r"\b(?:errors?|failures?)\b|错误|失败", re.I)
_FAILURE_QUERY = "evaluation failure cases and error categories"
_MAIN_RESULT_QUERY = re.compile(r"\b(?:main|primary|overall)\b|主要|总体", re.I)
_MAIN_RESULT_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?\s+)?(?P<title>(?:(?:main|experimental) )?results(?: and discussion)?"
    r"|(?:主要|实验)?结果(?:与讨论)?)[：:]?$", re.I,
)
_RESULT_HEADING = re.compile(r"\b(?:results?|evaluation|experiments?|ablation|analysis)\b|结果|评估|实验|分析", re.I)
_SCOPE_HEADING = re.compile(r"\b(?:methods?|setup|datasets?|participants?|tasks?)\b|方法|设置|数据集|任务", re.I)
_PROOF_NAVIGATION = re.compile(
    r"\b(?:theorem|proof|lemma|counterexample)\b|\bcannot\s+(?:capture|compute|express|model|recognize|represent)\b"
    r"|定理|证明|引理|反例", re.I,
)
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


def normalize_targets(queries: Iterable[str | dict], documents=(), *, initial_queries: Iterable[str] = ()) -> list[dict]:
    targets = []
    known_docs = {document["doc_id"] for document in documents}
    repeated = {_field_key(query) for query in initial_queries}
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
        kind = item.get("source_kind")
        if isinstance(kind, str) and kind in {"methods", "results", "proof"}:
            target["source_kind"] = kind
        if target.get("doc_id") and known_docs and target["doc_id"] not in known_docs:
            continue
        page = item.get("page_index")
        if page is not None:
            if type(page) is not int or page < 0:
                continue
            target["page_index"] = page
        if (_field_key(query) in repeated and not target.keys() & {"section", "table_label", "page_index", "source_kind"}
                and ("doc_id" not in target or known_docs == {target["doc_id"]})):
            continue
        if target not in targets:
            targets.append(target)
        if len(targets) == 2:
            break
    return targets


def _page_failures(scope: dict, documents) -> list[dict]:
    failures = []
    for document in documents:
        doc_id = document["doc_id"]
        if scope["doc_id"] is not None and doc_id != scope["doc_id"]:
            continue
        for page in document.get("pages", []):
            if scope["page_index"] is not None and page["page_index"] != scope["page_index"]:
                continue
            if page.get("status") in {"unreadable", "no_text"} or page.get("table_status") == "unreadable":
                failures.append({"doc_id": doc_id, "page_index": page["page_index"],
                                 "status": page.get("table_status") or page["status"]})
    return failures


def _printed_headings(page: dict) -> tuple[str, ...]:
    titles = page.get("source_headings")
    if not isinstance(titles, list) or page.get("status") in {"unreadable", "no_text"}:
        return ()
    clean = dict.fromkeys(" ".join(title.split()) for title in titles if isinstance(title, str))
    return tuple(title for title in clean if 0 < len(title) <= 100)


def _source_hints(
    documents: Iterable[dict], queries: Iterable[str], proof_labels: Iterable[tuple[str, int, str]] = (),
) -> list[dict]:
    """Select printed navigation labels by query relevance, keeping physical page numbers."""
    terms = _terms(" ".join(queries))
    proof_by_page = {}
    for doc_id, page_index, label in proof_labels:
        proof_by_page.setdefault((doc_id, page_index), []).append(label)
    candidates = []
    seen = set()
    for doc_order, document in enumerate(documents):
        if not isinstance(document, dict):
            continue
        doc_id, pages = document.get("doc_id"), document.get("pages")
        if (not isinstance(doc_id, str) or not 0 < len(doc_id) <= 250 or doc_id != doc_id.strip()
                or not isinstance(pages, list)):
            continue
        for page in pages:
            if not isinstance(page, dict) or page.get("status") in {"unreadable", "no_text"}:
                continue
            page_index = page.get("page_index")
            if type(page_index) is not int or not 0 <= page_index < 2**32:
                continue
            key = doc_id, page_index
            if key in seen:
                continue
            headings = _printed_headings(page) or proof_by_page.get(key, ())
            scored = {title: len(terms & _terms(title)) for title in headings}
            if not scored:
                continue
            chosen = set(sorted(scored, key=scored.__getitem__, reverse=True)[:2])
            hint = {"doc_id": doc_id, "page_index": page_index, "headings": [title for title in scored if title in chosen]}
            candidates.append((-max(scored.values()), doc_order, page_index, hint))
            seen.add(key)
    selected = sorted(candidates, key=lambda item: item[:3])[:6]
    return [hint for _, _, _, hint in sorted(selected, key=lambda item: item[1:3])]


def _is_printed_page_heading(target: dict, documents: Iterable[dict]) -> bool:
    """A displayed heading and its page can be sound even when indexed section labels lag."""
    if target["page_index"] is None or not target["doc_id"] or not target["section"] or target["table_label"]:
        return False
    return any(_field_key(heading) == _field_key(target["section"])
               for document in documents if document["doc_id"] == target["doc_id"]
               for page in document.get("pages", ()) if page["page_index"] == target["page_index"]
               for heading in _printed_headings(page))


def _interleave_units(index: EvidenceIndex, lanes: Iterable[Iterable[str]]) -> Iterable[tuple[str, ...]]:
    groups = [index.evidence_units(lane) for lane in lanes]
    return (unit for tier in zip_longest(*groups) for unit in tier if unit is not None)


def targeted_evidence(
    index: EvidenceIndex, queries: Iterable[str | dict], *, documents=(), limit: int = 8,
    prior_ids: Iterable[str] = (),
) -> dict:
    """Check up to two named sources, and keep ordinary retrieval distinct from direct inspection."""
    documents = tuple(documents)
    sources = normalize_targets(queries, documents)
    primary_documents = [doc["doc_id"] for doc in documents if doc.get("role") == "main"]
    prior_ids = tuple(prior_ids)
    primary, wider = [], []
    scopes = []
    paths = []
    for source in sources:
        target = {key: source.get(key) for key in _SCOPE_KEYS}
        if source.get("source_kind") and target["doc_id"] is None and len(primary_documents) == 1:
            target["doc_id"] = primary_documents[0]
        explicit_location = any(target[key] is not None for key in ("section", "table_label", "page_index"))
        if not explicit_location:
            target["section"] = _SOURCE_SECTIONS.get(source.get("source_kind"))
        query = source["query"]
        proof_query = source.get("source_kind") == "proof"
        if any(target[key] is not None for key in ("section", "table_label", "page_index")):
            result = index.inspect(**target, limit=limit, query=query, prior_ids=prior_ids, passage_query=proof_query)
            scope = result["scope"]
            repaired_heading = not scope["matching_count"] and _is_printed_page_heading(target, documents)
            if repaired_heading:
                result = index.inspect(**{**target, "section": None}, limit=limit, query=query,
                                       prior_ids=prior_ids, passage_query=proof_query)
                scope = result["scope"]
                scope["printed_heading"] = target["section"]
            failures = _page_failures(scope, documents)
            scope["unreadable_locations"] = failures
            scope["complete"] = explicit_location and scope["complete"] and not failures and not repaired_heading
            section_hint = proof_query and target["section"] is not None and all(
                target[key] is None for key in ("page_index", "table_label")
            )
            can_widen = not scope["matching_count"] or not (repaired_heading or target["table_label"])
            if (section_hint or not scope["complete"]) and can_widen:
                broad = index.search(query, limit=limit, doc_id=target["doc_id"],
                                     followup=True, prior_ids=prior_ids, passage_query=proof_query)
                inspected = set(scope["examined_evidence_ids"])
                broad_by_id = {row["evidence_id"]: row for row in broad["evidence"]}
                extra_units = [unit for unit in index.evidence_units(broad_by_id) if not inspected.issuperset(unit)]
                if extra_units:
                    extra = [broad_by_id[eid] for unit in extra_units for eid in unit]
                    scopes.append({**broad["scope"], "mode": "ranked", "requirement": source.get("requirement")})
                    paths.extend(broad["paths"])
                    if section_hint and len(extra_units[0]) > 1:
                        wider.append(result["evidence"])
                        result["evidence"] = extra
                    elif result["evidence"]:
                        wider.append(extra)
                    else:
                        result["evidence"] = extra
        else:
            result = index.search(query, limit=limit, doc_id=target["doc_id"],
                                  followup=True, prior_ids=prior_ids, passage_query=proof_query)
            scope = {**result["scope"], "mode": "ranked"}
        if source.get("source_kind"):
            scope["source_kind"] = source["source_kind"]
        scopes.append({**scope, "query": source["query"], "requirement": source.get("requirement")})
        paths.extend(result.get("paths", []))
        primary.append(result["evidence"])

    ordered = list(_interleave_units(index, ([row["evidence_id"] for row in lane] for lane in primary)))
    expanded = _interleave_units(index, ([row["evidence_id"] for row in lane] for lane in wider))
    preferred_ids = {eid for unit in ordered for eid in unit}
    alternatives = [unit for unit in expanded if not preferred_ids.issuperset(unit)]
    primary_space = limit - len(ordered[0]) if ordered else limit
    reserve = min(limit // 2, len(alternatives[0]), primary_space) if alternatives else 0
    selected = {}

    def pack(units, cap):
        for unit in units:
            if len(selected.keys() | set(unit)) <= cap:
                selected.update(dict.fromkeys(unit))

    pack(ordered, limit - reserve)
    pack(alternatives, min(limit, len(selected) + reserve))
    pack(ordered, limit)
    pack(alternatives, limit)
    for scope in scopes:
        if scope["mode"] == "direct" and scope["complete"]:
            scope["complete"] = set(scope["examined_evidence_ids"]).issubset(selected)
    by_id = {row["evidence_id"]: row for lane in (*primary, *wider) for row in lane}
    return {"evidence": [by_id[eid] for eid in selected], "paths": paths, "scopes": scopes}


def _field_key(value: str) -> str:
    return " ".join(value.casefold().split())


def _target_location(target: dict, documents: Sequence[dict]) -> tuple:
    doc_id = target.get("doc_id") or (documents[0]["doc_id"] if len(documents) == 1 else None)
    table = target.get("table_label", "")
    return (doc_id, target.get("page_index"), _inspection_section(target.get("section", "")),
            _printed_table_label(table) or _field_key(table))


def _pending_targets(output, requirements: list[str], documents, *, initial_queries: Iterable[str] = ()) -> list[dict]:
    proposals = [target.model_dump() for target in getattr(output, "follow_up_targets", [])]
    typed = normalize_targets(proposals, documents, initial_queries=initial_queries)
    represented = {_field_key(target["query"]) for target in typed}
    proposals = [*typed, *(query for query in output.follow_up_queries if _field_key(query) not in represented)]
    # If the model calls the field missing, an explicitly named source in its original requirement
    # gives the controller a direct review path even without a proposed retrieval query.
    missing = {_field_key(field) for field in output.missing_fields}
    for field in requirements:
        target = _text_target(field)
        if _field_key(field) in missing and any(target.get(key) is not None for key in _SCOPE_KEYS):
            proposals.append({**target, "requirement": field})
    targets = normalize_targets(proposals, documents, initial_queries=initial_queries)
    if len(missing) == 1:
        requirement = next(iter(missing))
        for target in targets:
            target.setdefault("requirement", requirement)
    return targets


def _additional_inspection(index, targets, previous, coverage, requirements, documents, sent, limit):
    unresolved = {_field_key(field["requirement"]) for field in coverage if field["status"] != "found"}
    unresolved.intersection_update(_field_key(field) for field in requirements)
    visited = {_target_location(target, documents) for target in previous}
    targets = [target for target in targets
               if target.keys() & {"page_index", "section", "table_label"}
               and _field_key(target.get("requirement", "")) in unresolved
               and _target_location(target, documents) not in visited]
    if not targets:
        return [], None
    results = targeted_evidence(index, targets, documents=documents, limit=limit, prior_ids=sorted(sent))
    fresh = {row["evidence_id"] for row in results["evidence"]} - set(sent)
    if any(fresh.intersection(scope["examined_evidence_ids"]) for scope in results["scopes"] if scope["mode"] == "direct"):
        return targets, results
    return [], None


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


def _requires_empirical_support(claim: dict) -> bool:
    if claim["kind"] in {"quantitative", "causal"}:
        return True
    if claim["kind"] not in {"result", "scope"}:
        return False
    typed = claim.get("empirical_result")
    return typed if type(typed) is bool else bool(_EMPIRICAL_RESULT.search(claim["proposition"]))


def _empirical_support_blocker(
    output: ScopedJudgement, sent: list[str], scoped: dict[str, dict], documents: Iterable[dict],
    *, comparison_required: bool = False,
) -> str | None:
    proof = output.empirical_support
    if proof is None:
        return "missing_empirical_support"
    if proof.result_source == "summary":
        return "summary_only_result"
    result = set(proof.result_evidence_ids)
    scope = set(proof.scope_evidence_ids)
    if proof.result_source != "direct_result" or not result:
        return "missing_observed_result"
    if proof.same_study_and_scope != "same" or not scope:
        return "unconfirmed_tested_scope"
    anchors = result | scope
    if not set(sent).issuperset(anchors):
        return "ungrounded_empirical_support"
    used = {_document_key(scoped[eid]) for eid in anchors}
    if len(used) > 1:
        doc_ids = {doc_id for doc_id, _ in used}
        primary = {doc["doc_id"] for doc in documents if doc.get("role") == "main"}
        supplements = {doc["doc_id"] for doc in documents if doc.get("role") == "supplement"}
        if (len(used) != len(doc_ids) or len(primary) != 1 or not primary <= doc_ids
                or not doc_ids <= primary | supplements):
            return "unconfirmed_tested_scope"
    if not any(group.role == "support" and set(group.evidence_ids).issuperset(anchors)
               for group in output.evidence_groups):
        return "unlinked_result_and_scope"
    if comparison_required:
        paired = proof.comparison
        if paired is None:
            return "unconfirmed_matched_comparison"
        focal, comparator = _field_key(paired.focal_condition), _field_key(paired.comparator_condition)
        if (paired.comparable != "same" or not focal or not comparator or focal == comparator
                or not paired.shared_outcome.strip() or not paired.focal_evidence_ids or not paired.comparator_evidence_ids
                or not result.issuperset((*paired.focal_evidence_ids, *paired.comparator_evidence_ids))):
            return "unconfirmed_matched_comparison"
    return None


def _empirical_followup(claim: dict, blocker: str, hints: list[dict]) -> dict | None:
    """Use real printed headings for one bounded check if the judge stops early."""
    if blocker == "unconfirmed_matched_comparison":
        return {"query": f"Matched outcome comparison between tested conditions {claim['support_query']}",
                "requirement": "Comparable observed outcomes under both conditions", "source_kind": "results"}
    if blocker == "unconfirmed_tested_scope":
        choices = ((_SCOPE_HEADING, "tested models tasks comparator and study conditions", "methods"),)
    elif blocker in {"missing_empirical_support", "summary_only_result", "missing_observed_result"}:
        choices = ((_RESULT_HEADING, "observed outcomes and measured comparisons", "results"),)
        if _FAILURE_SUBJECT.search(claim["proposition"]):
            choices = ((_FAILURE_HEADING, _FAILURE_QUERY, "results"), *choices)
    else:
        return None
    for pattern, focus, kind in choices:
        for hint in hints:
            matching = next((title for title in hint["headings"] if pattern.search(title)), None)
            if matching:
                field = "Tested study scope" if kind == "methods" else "Observed empirical result"
                return {"query": f"{matching} {focus}", "requirement": field, "source_kind": kind,
                        "doc_id": hint["doc_id"], "page_index": hint["page_index"]}
    return None


def _empirical_context_query(claim: dict, scope: dict, documents: Iterable[dict]) -> str | None:
    """Use a printed heading as a second retrieval cue within the requested results page."""
    if scope.get("source_kind") != "results" or scope.get("section") or scope.get("page_index") is None:
        return None
    for document in documents:
        if document["doc_id"] != scope["doc_id"]:
            continue
        for page in document.get("pages", ()):
            if page["page_index"] != scope["page_index"]:
                continue
            headings = [title for title in page.get("source_headings", ())
                        if isinstance(title, str) and 0 < len(title) <= 100]
            if _FAILURE_SUBJECT.search(claim["proposition"]):
                failure_heading = next((title for title in headings if _FAILURE_HEADING.search(title)), None)
                if failure_heading:
                    return f"{failure_heading} {_FAILURE_QUERY}"
            if _MAIN_RESULT_QUERY.search(scope["query"]):
                for title in headings:
                    if match := _MAIN_RESULT_HEADING.fullmatch(title):
                        return match["title"]
    return None


def _contradiction_error(output: ScopedJudgement, sent: list[str]) -> str | None:
    proof = output.direct_contradiction
    if proof is None:
        return "missing_direct_contradiction"
    if proof.subject_and_conditions != "same" or proof.relation == "uncertain" or not proof.scope_reason.strip():
        return "unconfirmed_contradiction_scope"
    ids = set(proof.evidence_ids) | set(proof.scope_evidence_ids)
    if (not proof.evidence_ids or not set(sent).issuperset(ids)
            or not any(group.role == "refute" and set(group.evidence_ids).issuperset(ids)
                       for group in output.evidence_groups)):
        return "ungrounded_contradiction"
    return None


_OPTIONAL_CALIBRATION_DETAIL = re.compile(r"(?:instrument|equipment|sensor|noise) calibration(?: (?:records|details))?\Z", re.I)
_CALIBRATION_SCOPE = re.compile(r"\bcalibrat\w*\b|校准|标定", re.I)


def _optional_refutation_requirements(claim: dict) -> list[str]:
    """Declare optional measurement details before the judge proposes a verdict."""
    if _CALIBRATION_SCOPE.search(claim["proposition"]):
        return []
    return [field for field in claim["requirements"] if _OPTIONAL_CALIBRATION_DETAIL.fullmatch(_field_key(field))]


def _covers_requirements(coverage: list[dict], required: set[str], evidence_ids: Iterable[str]) -> bool:
    anchors = set(evidence_ids)
    linked = {_field_key(field["requirement"]) for field in coverage
              if field["status"] == "found" and anchors.intersection(field["evidence_ids"])}
    return required <= linked


def _direct_refutation_blocker(claim: dict, output: ScopedJudgement, coverage: list[dict]) -> str | None:
    proof = output.direct_contradiction
    assert proof is not None
    if claim.get("logic_review_required"):
        return "unresolved_sentence_logic"
    if any(group.role == "support" and group.evidence_ids for group in output.evidence_groups):
        return "conflicting_support"
    if any(field["status"] == "unreadable" for field in coverage):
        return "unreadable_source"
    supplementary = {_field_key(field) for field in proof.supplementary_fields if field.strip()}
    allowed_supplementary = {_field_key(field) for field in _optional_refutation_requirements(claim)}
    if not supplementary <= allowed_supplementary:
        return "unapproved_supplementary_requirement"
    required = {_field_key(field["requirement"]) for field in coverage}
    unresolved = {_field_key(field["requirement"]) for field in coverage if field["status"] != "found"}
    core = required - supplementary
    if not unresolved <= supplementary:
        return "unchecked_decisive_requirement"
    scope_ids = set(proof.scope_evidence_ids or proof.evidence_ids)
    if not any(_field_key(field["requirement"]) in core and field["status"] == "found"
               and scope_ids.intersection(field["evidence_ids"]) for field in coverage):
        return "missing_scope_coverage"
    proof_ids = scope_ids | set(proof.evidence_ids)
    if not _covers_requirements(coverage, core, proof_ids):
        return "missing_scope_coverage"
    if output.follow_up_queries or any(not target.requirement or _field_key(target.requirement) not in supplementary
                                       for target in output.follow_up_targets):
        return "outstanding_decisive_followup"
    return None


_SENTENCE_BOUNDARY = re.compile(r"[!?。！？](?=\s|[^a-zA-Z0-9]|$)|[。！？]|\.(?=\s|$)|\n+")
# Some PDF layouts emit a footnote callout followed by the same standalone marker.
_TRAILING_FOOTNOTE = re.compile(
    r"(?<=[^\W\d_]{2}\.)(?:(?P<marker>[1-9]\d?)[ \t]+(?P=marker)|[⁰¹²³⁴⁵⁶⁷⁸⁹]{1,2})$"
)
_ABBREVIATION_CONTINUATIONS = (
    (re.compile(r"\bet[ \t]+al\.$", re.I), re.compile(r"[ \t]+(?:\(?[12]\d{3}[a-zA-Z]?(?=[.,;)\]])|[a-z])")),
    (re.compile(r"\b(?:e\.g|i\.e)\.$", re.I), re.compile(r"[ \t]+(?=\w)")),
    (re.compile(r"\b(?:figs?|eqs?|secs?|refs?|nos?|pp)\.$", re.I), re.compile(r"[ \t]+\(?\d")),
)
_DETACHED_FOOTNOTE = re.compile(r"[1-9][0-9]?")
_FIDELITY_NUMBER = re.compile(
    r"(?<![a-z0-9_.])(?P<value>[+\-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+))"
    r"(?:\s*(?P<unit>percentage\s+points?\b|p\.?p\b\.?|个?百分点|percent\b|%|"
    r"milliseconds?\b|ms\b|seconds?\b|secs?\b|(?:minutes?|hours?|days?|weeks?|years?)\b|毫秒|秒|"
    r"(?:[km]\s*)?(?:tokens?|samples?|words?)\b|fold\b|×))?(?![a-z0-9_]|\.\d)"
)
_FIDELITY_UNITS = {
    "%": "%", "percent": "%", "pp": "pp", "p.p": "pp", "p.p.": "pp", "百分点": "pp", "个百分点": "pp",
    "ms": "ms", "millisecond": "ms", "milliseconds": "ms", "毫秒": "ms",
    "sec": "s", "secs": "s", "second": "s", "seconds": "s", "秒": "s", "fold": "fold", "×": "fold",
}
_FIDELITY_MARKERS = {
    "negation": r"\b(?:not(?!\s+only\b)|no(?!\s+(?:more|less|fewer)\b|\.)|never|neither|nor|cannot|without)\b|n['’]t\b|未能|没有|不能|并非|未(?:提高|降低|提升|增加|下降)",
    "exclusive": r"\b(?:only|solely|exclusively)\b(?<!not only)|仅(?:有|限|在|对|用于)?",
    "exception": r"\b(?:unless|except)\b|除非|除了",
    "conditional": r"\b(?:if|provided that)\b|如果|前提是",
    "minimum": r"\b(?:at least|no (?:less|fewer) than)\b|至少|不低于",
    "maximum": r"\b(?:at most|up to|no more than)\b|至多|不超过",
    "universal": r"\b(?:all|each|every)\b|所有|每(?:项|个|位)",
    "human": r"\b(?:humans?|patients?)\b|人体|人类|患者",
    "adult": r"\badults?\b|成人",
    "child": r"\b(?:children|child|pediatric)\b|儿童",
    "mouse": r"\b(?:mice|mouse|murine)\b|小鼠",
    "rat": r"\brats?\b|大鼠",
    "test_split": r"\btest\s+(?:split|set|cohort)\b|测试集",
    "validation_split": r"\b(?:validation|development)\s+(?:split|set|cohort)\b|验证集|开发集",
    "training_split": r"\btraining\s+(?:split|set|cohort)\b|训练集",
}
_FIDELITY_MARKER_PATTERNS = {name: re.compile(pattern) for name, pattern in _FIDELITY_MARKERS.items()}
_FIDELITY_WORD_UNITS = re.compile(r"(?:(?:[km]\s*)?(?:token|sample|word)|minute|hour|day|week|year)s?")
_FIDELITY_DIRECTIONS = (
    re.compile(r"\b(?:increas\w*|ris(?:e|es|ing)|rose|gain\w*|higher)\b|增加|上升|提高"),
    re.compile(r"\b(?:decreas\w*|reduc\w*|declin\w*|fell|lower|drop\w*)\b|减少|下降|降低"),
)


def sentences(text: str) -> list[dict]:
    parts = []
    start = 0
    ordinal = 0
    internal_boundaries = 0
    boundaries = _SENTENCE_BOUNDARY.finditer(text)
    for end in [*(match.end() for match in boundaries), len(text)]:
        if text[end - 1:end] == "." and any(
            suffix.search(text, max(start, end - 24), end) and continuation.match(text, end)
            for suffix, continuation in _ABBREVIATION_CONTINUATIONS
        ):
            internal_boundaries += 1
            continue
        segment = text[start:end]
        stripped = segment.strip()
        if stripped:
            ordinal += 1
            offset = start + len(segment) - len(segment.lstrip())
            parts.append({"sentence_id": f"s{ordinal}", "text": stripped,
                          "start": offset, "end": offset + len(stripped)})
        # Keep IDs of subsequent unchanged spans stable in saved reports and gold labels.
        ordinal += internal_boundaries
        internal_boundaries = 0
        start = end
    return parts


def _anchor_claim(part: dict, claim: dict, offset: int | None = None, *, prefer_token: bool = False) -> dict:
    quote = claim["quote"]
    if offset is None:
        offset = part["text"].find(quote)
        if prefer_token and re.fullmatch(r"[A-Za-z0-9_]+", quote):
            token = re.search(rf"(?<![A-Za-z0-9_]){re.escape(quote)}(?![A-Za-z0-9_])", part["text"])
            offset = token.start() if token else -1
    if not quote.strip() or offset < 0 or part["text"][offset:offset + len(quote)] != quote:
        raise ModelUnavailable("claim extraction returned an invalid abstract anchor")
    start = part["start"] + offset
    claim_id = hashlib.sha256(f"{start}:{quote}:{claim['proposition']}".encode()).hexdigest()[:20]
    return {**claim, "claim_id": claim_id, "start": start, "end": start + len(quote),
            "sentence_start": part["start"], "sentence_end": part["end"], "text": quote}


def _plain_claim(part: dict, offset: int = 0, end: int | None = None, proposition: str | None = None) -> dict:
    quote = part["text"][offset:end]
    text = proposition or quote
    whole = requires_whole_sentence(text)
    kind = "quantitative" if re.search(r"\d+(?:[.,]\d+)?\s*%|百分点|percentage points?", text, re.I) else "result"
    claim = {"sentence_id": part["sentence_id"], "quote": quote, "proposition": text,
             "kind": "scope" if whole else kind, "requirements": ["observed result and scope"],
             "support_query": text, "counter_query": "limitation exception subgroup " + text}
    if whole:
        claim["logic_review_required"] = True
    return _anchor_claim(part, claim, offset)


def _compound_claims(part: dict, split: tuple[list[tuple[int, int, str]], str]) -> list[dict]:
    operands, operator = split
    atoms = [_plain_claim(part, start, end, proposition) for start, end, proposition in operands]
    attach_compound(part, atoms, operator)
    if not all("compound" in atom for atom in atoms):
        return []
    if condition := adjudicable_shared_exception(part["text"]):
        for atom in atoms:
            atom.pop("logic_review_required", None)
            atom["requirements"].append(f"scope in which the exception does not hold ({condition})")
    return atoms


def _claim_sentence(part: dict) -> dict:
    """Keep the raw sentence span available to references and revision callers."""
    text = part["text"]
    if footnote := _TRAILING_FOOTNOTE.search(text):
        end = footnote.start()
        if not any(suffix.search(text, max(0, end - 24), end) for suffix, _ in _ABBREVIATION_CONTINUATIONS):
            return {**part, "text": text[:end], "end": part["start"] + end}
    return part


def _claim_sentences(text: str) -> list[dict]:
    parts = sentences(text)
    if len(parts) > 1:
        previous, last = parts[-2:]
        gap = text[previous["end"]:last["start"]]
        if (_DETACHED_FOOTNOTE.fullmatch(last["text"]) and re.search(r"[^\W\d_]{2}[.!?。！？]$", previous["text"])
                and gap and not gap.strip()):
            parts.pop()
    return [_claim_sentence(part) for part in parts]


def recover_compound_claim(claim: dict, source_text: str) -> list[dict]:
    """Recover reviewable operands of a cached claim only at its original source anchors."""
    if not claim.get("logic_review_required"):
        return []
    part = next((row for row in _claim_sentences(source_text) if row["sentence_id"] == claim.get("sentence_id")), None)
    if (part is None or part["start"] != claim.get("sentence_start") or part["end"] != claim.get("sentence_end")):
        return []
    split = shared_procedure_parts(part["text"]) or reported_compound_parts(part["text"])
    if split is None:
        return []
    if group := claim.get("compound"):
        fields = ("claim_id", "start", "end", "quote", "proposition")
        for atom in _compound_claims(part, split):
            if (not atom.get("logic_review_required") and group == atom["compound"]
                    and all(atom[field] == claim.get(field) for field in fields)):
                updated = {key: value for key, value in claim.items() if key != "logic_review_required"}
                updated["requirements"] = list(dict.fromkeys([*claim["requirements"], *atom["requirements"]]))
                return [updated]
        return []
    if (part["start"] != claim.get("start") or part["end"] != claim.get("end")
            or part["text"] != claim.get("quote") or part["text"] != claim.get("proposition")):
        return []
    return _compound_claims(part, split)


def _fidelity_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().replace("−", "-")


def _fidelity_quantities(text: str) -> set[tuple[Decimal, str]]:
    quantities = set()
    for match in _FIDELITY_NUMBER.finditer(text):
        unit = " ".join((match["unit"] or "").split())
        if unit.startswith("percentage point"):
            unit = "pp"
        if unit == "pp.":
            unit = "pp"
        if _FIDELITY_WORD_UNITS.fullmatch(unit):
            unit = unit.removesuffix("s").replace(" ", "")
        quantities.add((Decimal(match["value"].replace(",", "")), _FIDELITY_UNITS.get(unit, unit)))
    return quantities


def _fidelity_markers(text: str) -> set[str]:
    return {name for name, pattern in _FIDELITY_MARKER_PATTERNS.items() if pattern.search(text)}


def _claim_fidelity_blockers(part: dict, atom: dict, *, only_claim: bool) -> list[str]:
    source, quote, proposed = map(_fidelity_text, (part["text"], atom["quote"], atom["proposition"]))
    protected = source if only_claim else quote
    source_quantities = _fidelity_quantities(source)
    proposed_quantities = _fidelity_quantities(proposed)
    blockers = []
    if not _fidelity_quantities(protected) <= proposed_quantities <= source_quantities:
        blockers.append("quantity_changed")
    if not _fidelity_markers(protected) <= _fidelity_markers(proposed) <= _fidelity_markers(source):
        blockers.append("qualification_changed")
    directions = [tuple(bool(pattern.search(text)) for pattern in _FIDELITY_DIRECTIONS) for text in (protected, proposed)]
    if all(sum(direction) == 1 for direction in directions) and directions[0] != directions[1]:
        blockers.append("direction_changed")
    return blockers


def _restore_source_queries(atom: dict, part: dict) -> None:
    available = _fidelity_quantities(_fidelity_text(part["text"]))
    claim_units = {unit for _, unit in _fidelity_quantities(_fidelity_text(atom["proposition"]))}
    # Values in a named table or page are retrieval hypotheses. Unscoped queries should retain the source quantity.
    queries = (atom["support_query"], atom["counter_query"])
    queried = set().union(*(_fidelity_quantities(_fidelity_text(query)) for query in queries
                            if not (_TABLE_QUERY.search(query) or _PAGE_QUERY.search(query))))
    if any(unit in claim_units for _, unit in queried - available):
        atom["support_query"] = atom["proposition"]
        atom["counter_query"] = "limitation exception subgroup " + atom["proposition"]
        atom["extraction_guard"] = {"status": "query_rebuilt", "blocker_codes": ["ungrounded_query_quantity"]}


def extract_claims(text: str, model=None) -> list[dict]:
    parts = _claim_sentences(text)
    if not parts:
        return []
    if model is None:
        results = []
        for part in parts:
            split = offline_compound_parts(part["text"])
            atoms = _compound_claims(part, split) if split else []
            results.extend(atoms or [_plain_claim(part)])
        return results

    parsed = model.call("extract", {"sentences": parts}, Extraction)
    by_sentence = {part["sentence_id"]: part for part in parts}
    grounded_splits = {part["sentence_id"]: split for part in parts
                       if (split := shared_procedure_parts(part["text"]) or reported_compound_parts(part["text"]))}
    groups = getattr(parsed, "compounds", [])
    compound_indexes = {index for group in groups for index in group.claim_indexes}
    anchored = []
    whole_sentences = {part["sentence_id"] for part in parts if requires_whole_sentence(part["text"])}
    for index, row in enumerate(parsed.claims):
        claim = row.model_dump()
        part = by_sentence.get(claim["sentence_id"])
        if part is None:
            raise ModelUnavailable("claim extraction returned an invalid abstract anchor")
        try:
            anchored.append(_anchor_claim(part, claim, prefer_token=index in compound_indexes))
        except ModelUnavailable:
            if index not in compound_indexes:
                raise
            whole_sentences.add(part["sentence_id"])
            anchored.append(_plain_claim(part))
    for group in groups:
        sentence_id, operands = group.sentence_id, group.claim_indexes
        part = by_sentence.get(sentence_id)
        referenced = {anchored[i]["sentence_id"] for i in operands if 0 <= i < len(anchored)}
        if part is None:
            whole_sentences.update(referenced)
            continue
        if sentence_id in whole_sentences:
            continue
        scoped = {i for i, atom in enumerate(anchored) if (
            atom["sentence_id"] == sentence_id and atom["kind"] not in {"background", "purpose"}
        )}
        if len(operands) != len(scoped) or set(operands) != scoped:
            whole_sentences.update({sentence_id, *referenced})
            continue
        atoms = [anchored[i] for i in operands]
        present = {atom["compound"]["operator"] for atom in atoms if "compound" in atom}
        if not present:
            attach_compound(part, atoms, group.operator)
        if not all(atom.get("compound", {}).get("operator") == group.operator for atom in atoms):
            whole_sentences.add(sentence_id)

    counts = {}
    for atom in anchored:
        sentence_id = atom["sentence_id"]
        counts[sentence_id] = counts.get(sentence_id, 0) + 1

    fidelity_failures = {}
    for atom in anchored:
        sentence_id = atom["sentence_id"]
        if atom["kind"] not in {"background", "purpose"}:
            blockers = _claim_fidelity_blockers(by_sentence[sentence_id], atom, only_claim=counts[sentence_id] == 1)
            if blockers:
                fidelity_failures.setdefault(sentence_id, {}).update(dict.fromkeys(blockers))

    result = {}
    for atom in anchored:
        sentence_id = atom["sentence_id"]
        part = by_sentence[sentence_id]
        whole = sentence_id in whole_sentences
        blockers = fidelity_failures.get(sentence_id)
        unsplit = (counts[sentence_id] == 1 and atom["kind"] not in {"background", "purpose"}
                   and atom["quote"].rstrip(".?!") == part["text"].rstrip(".?!"))
        split = grounded_splits.get(sentence_id)
        if whole or unsplit or blockers or (split and "compound" not in atom):
            if blockers and not whole and not split:
                split = offline_compound_parts(part["text"])
            recovered = _compound_claims(part, split) if split else []
            if recovered:
                if blockers:
                    for member in recovered:
                        member["extraction_guard"] = {"status": "source_fallback", "blocker_codes": list(blockers)}
                result.update((member["claim_id"], member) for member in recovered)
                continue
        if whole or blockers:
            atom = _plain_claim(part)
            if whole:
                atom["logic_review_required"] = True
            if blockers:
                atom["extraction_guard"] = {"status": "source_fallback", "blocker_codes": list(blockers)}
        else:
            _restore_source_queries(atom, part)
        result[atom["claim_id"]] = atom
    missing = (whole_sentences | grounded_splits.keys()) - counts.keys()
    for part in parts:
        sentence_id = part["sentence_id"]
        if sentence_id not in missing:
            continue
        split = grounded_splits.get(sentence_id)
        if recovered := (_compound_claims(part, split) if split else []):
            result.update((member["claim_id"], member) for member in recovered)
        elif sentence_id in whole_sentences:
            atom = _plain_claim(part)
            atom["logic_review_required"] = True
            result[atom["claim_id"]] = atom
    return list(result.values())


def refresh_compound_findings(report: dict) -> None:
    compounds = aggregate_compound_findings(report.get("claims", []), report.get("findings", []))
    if compounds:
        version = report.get("original_abstract", {}).get("version_id")
        report["compound_findings"] = [{**group, "original_version_id": version} for group in compounds]
    else:
        report.pop("compound_findings", None)


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


def blind_evidence(evidence: list[dict], *, preserve_order: bool = False) -> list[dict]:
    permitted = {"evidence_id", "doc_id", "kind", "text", "locator", "row_header", "column_header",
                 "column_header_path", "cell_role", "note_marker", "note_scope"}
    ordered = evidence if preserve_order else sorted(evidence, key=lambda item: item["evidence_id"])
    return [{key: item[key] for key in sorted(item.keys() & permitted)} for item in ordered]


def validate_ids(ids: list[str], evidence: Iterable[dict]) -> None:
    allowed = {item["evidence_id"] for item in evidence}
    if not allowed.issuperset(ids):
        raise InvalidJudgeReference("model cited evidence outside the supplied scope")


def _checked_judgement(
    output: ScopedJudgement, material: dict, sent_ids: list[str], scoped: dict[str, dict],
) -> tuple[ScopedJudgement, list[str], list[dict]]:
    """Resolve explicit inclusive citations within a single displayed body span."""
    allowed = {eid: scoped[eid] for eid in sent_ids}
    resolved = []
    if output.evidence_ranges:
        spans = [tuple(group["evidence_ids"]) for group in material.get("body_passages", ())]
        spans.extend(tuple(line[0] for line in group["lines"]) for group in material.get("body_context", ()))
        windows = []
        for span in spans:
            unique = set(span)
            if len(unique) != len(span) or not unique <= allowed.keys():
                continue
            records = [allowed[eid] for eid in span]
            if (all(record.get("kind") == "body_text" for record in records)
                    and len({EvidenceIndex._line_location(record) for record in records}) == 1):
                windows.append(span)
        groups = list(output.evidence_groups)
        for request in output.evidence_ranges:
            first, last = request.start_evidence_id, request.end_evidence_id
            matches = set()
            for span in windows:
                if first in span and last in span:
                    start, end = span.index(first), span.index(last)
                    if start <= end:
                        matches.add(span[start:end + 1])
            if len(matches) != 1:
                raise InvalidJudgeReference("model selected an invalid or ambiguous citation range in the supplied body spans")
            ids = list(matches.pop())
            group = EvidenceGroup(role=request.role, evidence_ids=ids)
            if group not in groups:
                groups.append(group)
            trace = {**request.model_dump(), "evidence_ids": ids}
            if trace not in resolved:
                resolved.append(trace)
        output = output.model_copy(update={"evidence_groups": groups})
    ids = list(dict.fromkeys(eid for group in output.evidence_groups for eid in group.evidence_ids))
    validate_ids(ids, allowed.values())
    return output, ids, resolved


def _judge_attempt(model, material: dict, sent: list[str], scoped: dict[str, dict], attempts: list[dict], *, repair=False):
    payload = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    attempt = {"material_sha256": hashlib.sha256(payload).hexdigest(), "repair": repair,
               "proposed_verdict": None, "status": "model_unavailable"}
    attempts.append(attempt)
    try:
        schema = CitedAbstractJudgement if "evidence_boundary" in material else ScopedJudgement
        output = model.call("judge", material, schema)
    except InvalidJudgeReference:
        attempt["status"] = "model_reference_error"
        raise
    except ModelPaused:
        attempt["status"] = "model_paused"
        raise
    attempt["proposed_verdict"] = output.verdict
    attempt["status"] = "citation_error"
    result = _checked_judgement(output, material, sent, scoped)
    attempt["status"] = "checked"
    return result


def _sent_paths(paths: list[dict], sent_ids: list[str]) -> list[dict]:
    sent = set(sent_ids)
    unique = {}
    for path in paths:
        if sent.issuperset(path["evidence_ids"]):
            key = (tuple(path["evidence_ids"]), tuple(
                (edge["relation"], tuple(anchor["evidence_id"] for anchor in edge["anchors"]))
                for edge in path["edges"]
            ))
            unique.setdefault(key, path)
    return list(unique.values())


def _pending(claim: dict, reason: str, *, status: str = "pending") -> dict:
    return {**claim, "verdict": None, "review_status": status, "reason": reason,
            "evidence_ids": [], "evidence_groups": [], "missing_fields": claim["requirements"]}


def _offline(claim: dict, candidates: list[dict], calculation: dict | None) -> dict:
    if claim.get("logic_review_required"):
        return _pending(claim, "This logical relationship needs a scope-aware review.")
    if claim["kind"] in {"background", "purpose"}:
        return _pending(claim, "This sentence is outside original-paper result verification.")
    if calculation is not None:
        matches = calculation["verification"] == "matched"
        ids = calculation["evidence_ids"]
        return {**claim, "review_status": "completed", "verdict": "supported" if matches else "refuted",
                "reason": "The scoped table operands were independently recomputed.", "evidence_ids": ids,
                "evidence_groups": [{"role": "support" if matches else "refute", "evidence_ids": ids}],
                "missing_fields": []}
    quote = " ".join(claim["quote"].split()).strip(".。")
    matches = [item for item in candidates if item.get("kind") == "body_text"
               and quote and quote == " ".join(item["text"].split()).strip(".。")]
    if matches:
        finding = _pending(claim, "Matching wording was located. Its source role and context still need review.")
        finding["verbatim_candidate_ids"] = [item["evidence_id"] for item in matches]
        return finding
    return _pending(claim, "The offline reviewer could not determine a verdict from the located evidence.")


def _figure_images(candidates: list[dict], documents: Iterable[dict], data_dir: Path) -> list[dict]:
    import pymupdf

    from .ingest import MAX_PDF_BYTES

    by_id = {doc["doc_id"]: doc for doc in documents}
    root = data_dir.resolve()
    figures = []
    for evidence in candidates:
        document = by_id.get(evidence.get("doc_id"), {})
        checksum = document.get("sha256")
        role = document.get("role")
        locator = evidence.get("locator", {})
        index, coords = locator.get("page_index"), locator.get("bbox")
        if (not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum)
                or role in {"main", "supplement", "external"} and document["doc_id"] != f"{role}-{checksum}"
                or type(index) is not int or index < 0
                or not any(page.get("page_index") == index for page in document.get("pages", []))
                or not isinstance(coords, (list, tuple)) or len(coords) != 4
                or not all(type(value) in (int, float) and isfinite(value) for value in coords)):
            continue
        relative = Path("blobs", f"{checksum}.pdf")
        if document.get("blob_path") != relative.as_posix():
            continue
        try:
            source = (root / relative).resolve()
            if source.parent != root / "blobs":
                continue
            with source.open("rb") as stream:
                payload = stream.read(MAX_PDF_BYTES + 1)
            if len(payload) > MAX_PDF_BYTES or hashlib.sha256(payload).hexdigest() != checksum:
                continue
            with pymupdf.open(stream=payload, filetype="pdf") as pdf:
                page = pdf[index]
                page.set_rotation(0)
                rect = pymupdf.Rect(coords)
                if rect.is_empty or not page.rect.contains(rect):
                    continue
                scale = min(2, 1400 / max(rect.width, rect.height))
                image = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=rect, alpha=False, annots=False)
                data = b64encode(image.tobytes("png")).decode("ascii")
        except (OSError, RuntimeError, ValueError, IndexError, pymupdf.mupdf.FzErrorBase):
            continue
        figures.append({"evidence_id": evidence["evidence_id"], "image_url": f"data:image/png;base64,{data}"})
        if len(figures) == 2:
            break
    return figures


def figure_material(evidence: Iterable[dict], documents: Iterable[dict], data_dir: Path | None) -> dict:
    """Describe visual availability without promoting a pixel reading to an exact value."""
    candidates = list({row["evidence_id"]: row for row in evidence if row.get("kind") == "figure"}.values())
    if not candidates:
        return {}
    images = _figure_images(candidates, documents, data_dir) if data_dir is not None else []
    rendered = {row["evidence_id"] for row in images}
    sent = 0
    observations = []
    for row in candidates:
        eid = row["evidence_id"]
        if eid in rendered:
            status, precision = "available", "approximate"
            sent += 1
        else:
            status, precision = "not_sent" if sent == 2 else "missing", None
        observations.append({"evidence_id": eid, "status": status, "numeric_precision": precision})
    material = {"figure_observations": observations}
    if images:
        material["figure_images"] = images
    return material


def review_claim(
    claim: dict, index: EvidenceIndex, model=None, *, documents=(), data_dir=None,
    reserve_followup: Callable[[], bool] | None = None, additional_evidence: Sequence[dict] = (),
    saved_targets: Sequence[dict] = (), save_targets: Callable[[list[dict]], None] | None = None,
    cited_abstract_ids: Sequence[str] | None = None,
) -> tuple[dict, dict | None]:
    documents = tuple(documents)
    requirements = [field for field in claim["requirements"] if field.strip()]
    if not requirements:
        requirements = ["observed result and scope"]
        claim = {**claim, "requirements": requirements}
    cited_abstract = cited_abstract_ids is not None
    abstract_ids = set(cited_abstract_ids or ())
    abstract_rows = index.get(cited_abstract_ids or ())
    malformed_abstract = any(row.get("kind") != "body_text" or row["locator"].get("section") != "abstract"
                             for row in abstract_rows)
    if cited_abstract and (documents or additional_evidence or saved_targets or malformed_abstract
                           or len(abstract_ids) != len(abstract_rows)
                           or len({_document_key(row) for row in abstract_rows}) > 1):
        raise ValueError("cited abstract mode requires sentences from a single abstract")
    full_abstract = bool(abstract_rows) and (len(abstract_rows) <= _CITED_ABSTRACT_MAX_SENTENCES
                                            and sum(len(row["text"]) for row in abstract_rows) <= _CITED_ABSTRACT_MAX_CHARS)
    primary_limit = len(abstract_rows) if full_abstract else 8
    examined_scopes = []
    paths = []
    by_id = {}
    breadth = []
    prioritized = []
    follow_up_evidence_ids = []
    sent_candidate_ids = []
    sent_body_context = []
    sent_passages = []
    sent_table_context = []
    sent_table_region = None
    sent_linked_paths = []
    sent_priority = []
    inspected_body_windows = []
    continuation_windows = []
    new_continuation_ids = set()
    decision_attempts = []
    decision_blockers = []
    output = None
    stop = None

    def remember(result: dict) -> None:
        paths.extend(result["paths"])
        breadth.append([item["evidence_id"] for item in result["evidence"]])
        by_id.update((item["evidence_id"], item) for item in result["evidence"])

    def gather(queries: Sequence[str], *, prefer_passage: bool = False) -> None:
        scope = {}
        if cited_abstract:
            doc_id, version = _document_key(abstract_rows[0])
            scope = {"section": "abstract", "doc_id": doc_id, "doc_version_id": version, "expand": False}
        for position, query in enumerate(dict.fromkeys(queries)):
            if not query.strip():
                continue
            result = index.search(query, limit=8, passage_query=prefer_passage and position == 0, **scope)
            examined_scopes.append({**result["scope"], "mode": "ranked"})
            remember(result)

    def inspect(plan: dict, *, second: bool, results: dict | None = None) -> bool:
        previous, known = plan["primary_ids"], set(plan["sent_ids"])
        if results is None:
            limit = primary_limit - len(plan["retained_ids"]) if second else primary_limit
            prior = known if second else known - set(plan["context_ids"])
            results = targeted_evidence(index, plan["targets"], documents=documents, limit=limit,
                                        prior_ids=sorted(prior))
        examined_scopes.extend(results["scopes"])
        remember(results)
        selected = [item["evidence_id"] for item in results["evidence"]]
        retained_rows = index.get(plan["retained_ids"])
        retained = [item["evidence_id"] for item in retained_rows] or previous[:1]
        by_id.update((item["evidence_id"], item) for item in retained_rows)
        exhaustive = {eid for scope in results["scopes"] if scope.get("complete")
                      for eid in scope["examined_evidence_ids"]}
        grouped = {eid for unit in index.evidence_units(selected) if len(unit) > 1 for eid in unit}
        for eid in retained:
            if eid in selected:
                continue
            if len(selected) < primary_limit:
                if retained_rows:
                    selected.append(eid)
                continue
            for offset in reversed(range(len(selected))):
                previous_id = selected[offset]
                if previous_id not in exhaustive and previous_id not in grouped and previous_id not in retained:
                    selected[offset] = eid
                    break
        previous_windows = len(inspected_body_windows)
        for scope in results["scopes"]:
            if scope["mode"] == "direct" and scope["matching_count"] and scope["table_label"] is None:
                if scope["page_index"] is not None:
                    inspected = set(scope["examined_evidence_ids"])
                    if continuation := index.page_continuation((eid for eid in selected if eid in inspected), documents):
                        continuation_windows.append(continuation)
                        window = continuation[1]
                        inspected_body_windows.append(window)
                        new_continuation_ids.update(set(window.evidence_ids) - known)
                        scope["complete"] = False
                        continue
                query = _empirical_context_query(claim, scope, documents) if empirical else None
                if query and query != scope["query"]:
                    scope["body_context_query"] = query
                inspected_body_windows.extend(index.ranked_body_context(
                    [query or scope["query"]], doc_id=scope["doc_id"], doc_version_id=scope["doc_version_id"],
                    page_index=scope["page_index"], section=scope["section"],
                ))
        if second:
            prioritized.clear()
            inspected_body_windows[:] = inspected_body_windows[previous_windows:] + inspected_body_windows[:previous_windows]
        prioritized.extend(selected)
        fresh = [eid for eid in selected if eid not in known]
        follow_up_evidence_ids.extend(fresh)
        if fresh or any(scope.get("complete") for scope in results["scopes"]):
            return True
        if not inspected_body_windows:
            return False
        primary = candidate_order()
        table_ids = [item["evidence_id"] for item in index.table_context(primary)["evidence"]]
        return any(not known.issuperset(group["new_evidence_ids"])
                   for group in index.pack_body_context(context_windows(primary), [*primary, *table_ids]))

    def context_windows(primary_ids):
        if inspected_body_windows:
            # A targeted check adds a source. Keep the original supporting range in the shared text budget.
            windows = list(dict.fromkeys([*inspected_body_windows[:1], *body_windows[:1],
                                          *inspected_body_windows[1:2], *body_windows[1:2]]))
        else:
            windows = list(dict.fromkeys(body_windows))[:2]
        if documents and not cited_abstract and claim["kind"] not in {"background", "purpose"}:
            if anchored := index.anchored_body_context(primary_ids, windows):
                windows.insert(len(windows) if inspected_body_windows else 0, anchored)
        return windows

    def candidate_order() -> list[str]:
        # Keep context from each lane together when combining support and counter searches.
        ordered = dict.fromkeys(prioritized[:primary_limit])
        for unit in _interleave_units(index, breadth):
            if len(ordered.keys() | set(unit)) <= primary_limit:
                ordered.update(dict.fromkeys(unit))
        if not cited_abstract or full_abstract:
            return list(ordered)
        remaining = _CITED_ABSTRACT_MAX_CHARS
        bounded = []
        for eid in ordered:
            if eid in abstract_ids and (size := len(by_id[eid]["text"])) <= remaining:
                bounded.append(eid)
                remaining -= size
        return bounded

    def evidence_boundary(sent_ids: list[str]) -> dict:
        return {"kind": "cited_abstract", "complete": full_abstract and set(sent_ids) == abstract_ids,
                "sentence_count": len(abstract_rows), "supplied_sentence_count": len(sent_ids)}

    initial_queries = (claim["support_query"], claim["counter_query"])
    proposition = claim["proposition"]
    empirical = _requires_empirical_support(claim)
    comparison_required = bool(_COMPARATIVE_EMPHASIS.search(proposition)) and (
        empirical or (claim["kind"] in {"result", "scope"} and bool(_EMPIRICAL_RESULT.search(proposition)))
    )
    empirical = empirical or comparison_required
    # An explicit absence is a scope condition. Keep the ordinary support and counter lanes for it.
    if not full_abstract and (not cited_abstract or abstract_rows):
        gather(initial_queries, prefer_passage=bool(documents) and claim["kind"] in {"method", "result", "scope"}
               and not (_TABLE_QUERY.search(proposition) or _ABSENCE_CONDITION.search(proposition)))
    body_windows = []
    if not cited_abstract and (model is None or claim["kind"] not in {"background", "purpose"}):
        body_windows = index.ranked_body_context(initial_queries)
        if claim["kind"] not in {"background", "purpose"}:
            focused_window, scope = focused_source_context(index, proposition, documents)
            if focused_window is not None:
                body_windows.insert(0, focused_window)
                examined_scopes.append(scope)
    if full_abstract:
        doc_id = abstract_rows[0]["doc_id"]
        prioritized.extend(cited_abstract_ids)
        by_id.update((row["evidence_id"], row) for row in abstract_rows)
        examined_scopes.extend({
            "mode": "direct", "section": "abstract", "doc_id": doc_id, "requirement": field,
            "complete": True, "examined_evidence_ids": list(cited_abstract_ids),
            "locations": [{"doc_id": doc_id, "page_index": 0, "section": "abstract"}],
        } for field in requirements)
    if additional_evidence:
        supplements = index.get(item["evidence_id"] for item in additional_evidence)
        prioritized.extend(item["evidence_id"] for item in supplements)
        by_id.update((item["evidence_id"], item) for item in supplements)
    rounds = []
    if saved_targets and isinstance(saved_targets[0], dict) and "targets" in saved_targets[0]:
        rounds = list(saved_targets[:_MAX_ORIGINAL_FOLLOWUPS])
    elif saved := normalize_targets(saved_targets, documents, initial_queries=initial_queries):
        # Older checkpoints stored one flat list without the originating packet.
        previous = candidate_order()
        context = index.table_context(previous)
        retained = list(saved_targets[0].get("retained_evidence_ids", ())) if isinstance(saved_targets[0], dict) else []
        known = set(previous) | {item["evidence_id"] for item in context["evidence"]} | set(retained)
        rounds = [{"targets": saved, "primary_ids": previous, "sent_ids": sorted(known),
                   "context_ids": [], "retained_ids": retained}]
    for ordinal, plan in enumerate(rounds):
        inspect(plan, second=bool(ordinal))
    followup = bool(rounds or saved_targets or additional_evidence)
    calculation = None
    while True:
        second_followup = bool(documents and len(rounds) == _MAX_ORIGINAL_FOLLOWUPS - 1
                               and not additional_evidence)
        candidates = [by_id[eid] for eid in candidate_order()]
        passages = index.body_passages(item["evidence_id"] for item in candidates)
        context = index.table_context(item["evidence_id"] for item in candidates)
        if cited_abstract and not candidates:
            finding = _pending(claim, "No readable sentences fit the cited-abstract evidence packet.")
            break
        context_by_id = {item["evidence_id"]: item for item in context["evidence"]}
        table_context = [
            {"cell_evidence_id": eid, "evidence": blind_evidence([context_by_id[related] for related in related_ids])}
            for eid, related_ids in sorted(context["by_seed"].items())
        ]
        scoped = {item["evidence_id"]: item for item in [*candidates, *context["evidence"]]}
        body_context = index.pack_body_context(context_windows([item["evidence_id"] for item in candidates]), scoped)
        body_ids = [eid for group in body_context for eid in group["new_evidence_ids"]]
        body_rows = {row["evidence_id"]: row for row in index.get(body_ids)}
        scoped.update(body_rows)
        table_region = None
        linked_table = None
        if documents and claim["kind"] not in {"background", "purpose"}:
            source_ids = [item["evidence_id"] for item in candidates]
            table_region = index.referenced_table_region(claim["support_query"], [*source_ids, *body_ids], scoped)
            if table_region:
                scoped.update((row["evidence_id"], row) for row in table_region["evidence"])
            previous_cell = rounds[-1].get("linked_table_cell_id") if rounds else None
            linked_table = index.linked_table_method(proposition, scoped, retained_cell_id=previous_cell)
            if linked_table:
                cell_id = linked_table["cell_evidence_id"]
                group = next((row for row in table_context if row["cell_evidence_id"] == cell_id), None)
                if group is None:
                    group = {"cell_evidence_id": cell_id, "evidence": []}
                    table_context.append(group)
                group["reference_evidence_id"] = linked_table["reference_evidence_id"]
                group["evidence"].extend(blind_evidence(linked_table["evidence"], preserve_order=True))
                scoped.update((row["evidence_id"], row) for row in linked_table["evidence"])
                if len(linked_table["method_ids"]) > 1:
                    passages.append({"evidence_ids": linked_table["method_ids"]})
        linked_paths = linked_table["paths"] if linked_table else []
        comparison = None if claim.get("logic_review_required") else comparison_for_claim(
            claim["proposition"], with_method_scope(scoped.values(), [*paths, *linked_paths])
        )
        calculation = comparison[0] if comparison else None
        packet_order = list(scoped)
        if documents and claim["kind"] not in {"background", "purpose"} and not _ABSENCE_CONDITION.search(proposition):
            packet_order = index.prioritize_packet(claim["support_query"], scoped)
        priority = packet_order[:8] if packet_order[:8] != list(scoped)[:8] else []
        if model is None:
            finding = _offline(claim, [*candidates, *body_rows.values()], calculation)
            sent_candidate_ids = packet_order
            sent_priority = priority
            sent_body_context = body_context
            sent_passages = passages
            sent_table_context = table_context
            sent_table_region = table_region
            sent_linked_paths = linked_paths
            break
        if claim["kind"] in {"background", "purpose"}:
            finding = _pending(claim, "This sentence is outside original-paper result verification.")
            break
        material = {"claim": claim["proposition"], "requirements": requirements,
                    "evidence": blind_evidence(candidates, preserve_order=bool(documents) or full_abstract),
                    "calculation": calculation,
                    "follow_up_already_performed": followup,
                    "direct_inspections": [scope for scope in examined_scopes if scope.get("mode") == "direct"],
                    "coverage_instructions": (
                        "For each required field, report its status; found requires supplied evidence IDs. "
                        "Read listed body passages in source order. Match the study, model, intervention and conditions. "
                        "Separate methods of the current study from cited or sensitivity analyses. "
                        "For an atomic claim, when the paper states the opposite or provides a witness within the claimed "
                        "domain, return refuted and a direct_contradiction. Put the opposite result or witness in evidence_ids and any "
                        "separate same-study, same-conditions anchors in scope_evidence_ids. Put both sets in one refute "
                        "group. Use subject_and_conditions=same only when the anchors establish that match, and say how "
                        "in scope_reason. For supplementary_fields, use only fields listed in optional_refutation_requirements. "
                        "If scope or another decisive condition is unresolved, request up to two follow_up_targets. "
                        "Use an empty follow_up_queries list when supplying structured targets. "
                        "Name the exact missing field and target the outcome, comparator, setup, or failure mechanism "
                        "using source terms. Specify a source or refine the initial query to trigger a fresh search. "
                        "For universal negatives, search for a "
                        "constructive witness using the same architecture and assumptions. Prefer source_kind methods, "
                        "results or proof when relevant. Name a section, table or page when known. Ordinary ranked "
                        "search is not a complete inspection."
                    )}
        if empirical:
            material["empirical_support_required"] = True
            material["coverage_instructions"] += (
                " To support an empirical claim, supply empirical_support. Identify the observed outcome or comparison "
                "in result_evidence_ids, and the tested population, models, tasks, comparator or conditions in "
                "scope_evidence_ids. Cite both in one support group. Classify result_source as direct_result only when "
                "the anchors give a concrete original observation, analysis or measurement for the claimed effect. "
                "A restatement of the claim or a summary of the authors' conclusions is summary. Use "
                "same_study_and_scope=same when the cited tests cover the asserted scope. If a result or scope is missing, "
                "request a targeted check of the actual results, error analysis or methods, naming the missing item. "
                "For qualitative claims, match the tested scope and direction; an unstated effect size or unit is "
                "supplementary if it cannot change either."
            )
        if priority:
            material["priority_evidence_ids"] = priority
            material["coverage_instructions"] += (
                " Start with priority_evidence_ids. They are ordered together for review, and their original text "
                "appears in evidence, body_context, table_context or table_region. Use other supplied sources as needed."
            )
        if comparison_required:
            material["coverage_instructions"] += (
                " The highlighted condition makes a comparative claim. To support it, include empirical_support.comparison. "
                "Identify both tested conditions and their shared outcome. Cite measured results for each condition, "
                "with comparable model, task and metric. A reported comparison can cover both conditions in one anchor. "
                "If the evidence covers only one condition or changes the comparison setup, request the missing result."
            )
        hints = []
        if documents and (not followup or second_followup):
            proof_claim = claim["kind"] == "theoretical" or _PROOF_NAVIGATION.search(proposition)
            hints = _source_hints(documents, (proposition, *requirements, *initial_queries),
                                  index.proof_navigation_labels if proof_claim else ())
            if second_followup:
                inspected_pages = {(scope["doc_id"], scope["page_index"]) for scope in examined_scopes
                                   if scope["mode"] == "direct" and scope.get("complete")}
                hints = [hint for hint in hints if (hint["doc_id"], hint["page_index"]) not in inspected_pages]
        if hints:
            material["source_hints"] = hints
            material["coverage_instructions"] += (
                " Choose the unresolved required condition and the most relevant source for checking it. Source hints "
                "are navigation labels only. For a specific printed page use its doc_id and zero-based page_index "
                "without a section filter. For a named section that may span pages, set section without restricting "
                "it to the heading's page. A known table can be targeted by table_label. Use a targeted search if "
                "no listed source fits. Base judgments on the returned source anchors."
            )
        if second_followup:
            material["coverage_instructions"] += (
                " One further check is available if a required condition is still unresolved. Use a structured "
                "follow_up_target with the exact requirement and a different named page, section or table."
            )
        if table_context:
            material["table_context"] = table_context
            if linked_table:
                material["coverage_instructions"] += (
                    " A table_context group with reference_evidence_id pairs a table value with a Methods passage "
                    "that explicitly references its table. Read anchors elsewhere in the packet when already supplied."
                )
        if table_region:
            material["table_region"] = {**table_region, "evidence": blind_evidence(table_region["evidence"], preserve_order=True)}
            material["coverage_instructions"] += (
                " table_region contains selected rows from the same-page table named in its reference. "
                "Use the cell locators, headers and notes to check scope. The rows may cover only part of the table."
            )
        if compound := claim.get("compound"):
            material["sentence_context"] = compound["text"]
            material["coverage_instructions"] += (
                " Use sentence_context to resolve shared terms and conditions. Judge only the highlighted proposition "
                "in claim. The other operands have their own findings."
            )
        if passages:
            material["body_passages"] = sorted(passages, key=lambda group: group["evidence_ids"][0])
        if body_context:
            material["body_context"] = []
            for group in body_context:
                first = scoped[group["evidence_ids"][0]]
                material["body_context"].append({
                    "doc_id": first["doc_id"], "page_index": first["locator"]["page_index"],
                    "section": first["locator"].get("section"),
                    "lines": [[eid, body_rows[eid]["text"]] if eid in group["new_evidence_ids"] else [eid]
                              for eid in group["evidence_ids"]],
                })
            material["coverage_instructions"] += (
                " Read body_context in line order. Each entry has an evidence ID and its source text when new. "
                "For a bare ID, read the text already supplied under evidence, table_context or an earlier range."
            )
        if passages or body_context:
            material["coverage_instructions"] += (
                " For one continuous citation, use evidence_ranges with a role and the inclusive start_evidence_id "
                "and end_evidence_id in one displayed body passage or context span. Cite separate evidence in "
                "individual evidence_groups."
            )
        if links := [
            {"from_evidence_id": source, "to_evidence_ids": list(window.evidence_ids)}
            for source, window in continuation_windows
            if source in scoped and scoped.keys() >= set(window.evidence_ids)
        ]:
            material["body_continuations"] = links
            material["coverage_instructions"] += (
                " body_continuations links an unfinished line to source text on the next physical page. "
                "Use separate citations for each page. These links do not make the source inspection exhaustive."
            )
        if claim.get("logic_review_required"):
            material["logic_review_required"] = True
            material["coverage_instructions"] += (
                " This sentence requires a full logic review. Use ordinary evidence groups and leave "
                "direct_contradiction empty."
            )
        if cited_abstract:
            material = {
                "claim": claim["proposition"],
                "requirements": requirements,
                "evidence": [{"evidence_id": row["evidence_id"], "text": row["text"]} for row in material["evidence"]],
                "evidence_boundary": evidence_boundary(packet_order),
                "coverage_instructions": _CITED_ABSTRACT_INSTRUCTIONS,
            }
            if calculation is not None:
                material["calculation"] = calculation
            if empirical:
                material["coverage_instructions"] += (
                    " For empirical support, identify the directly reported result and its conditions in empirical_support, "
                    "and put those citations together in a support group."
                )
            if comparison_required:
                material["coverage_instructions"] += (
                    " Supply empirical_support.comparison with matched outcomes under both conditions."
                )
        material["optional_refutation_requirements"] = _optional_refutation_requirements(claim)
        material["coverage_instructions"] += (
            " Treat every required field as decisive. Cite sources for each decisive field in the same evidence group as "
            "the result. In a refutation, only optional_refutation_requirements may be reported as unchecked supplementary_fields."
        )
        material.update(figure_material(candidates, documents, data_dir))
        try:
            try:
                output, ids, citation_ranges = _judge_attempt(model, material, packet_order, scoped, decision_attempts)
            except InvalidJudgeReference:
                output = None
            sent_candidate_ids = packet_order
            sent_priority = priority
            sent_body_context = body_context
            sent_passages = passages
            sent_table_context = table_context
            sent_table_region = table_region
            sent_linked_paths = linked_paths
            if output is None:
                repair = {**material, "retry": "Review the supplied IDs and citation ranges carefully."}
                output, ids, citation_ranges = _judge_attempt(model, repair, sent_candidate_ids, scoped,
                                                               decision_attempts, repair=True)
        except ModelUnavailable as exc:
            finding = _pending(claim, str(exc))
            finding["retryable"] = isinstance(exc, ModelPaused)
            if isinstance(exc, InvalidJudgeReference):
                stop = "invalid_judge_reference"
            elif isinstance(exc, ModelPaused) and str(exc) in {
                "model budget exhausted", "model time budget exhausted", "request_budget_exhausted", "time_budget_exhausted",
            }:
                stop = "budget_exhausted"
            else:
                stop = "model_unavailable"
            break

        coverage = _coverage(requirements, output, examined_scopes, sent_candidate_ids)
        verdict = output.verdict
        empirical_blocker = None
        if empirical and (verdict == "supported" or (verdict == "pending" and output.empirical_support is not None)):
            empirical_blocker = _empirical_support_blocker(
                output, sent_candidate_ids, scoped, documents, comparison_required=comparison_required,
            )
        elif comparison_required and verdict == "pending":
            empirical_blocker = "unconfirmed_matched_comparison"
        proof_error = proof_blocker = None
        if verdict == "refuted":
            proof_error = _contradiction_error(output, sent_candidate_ids)
            proof_blocker = proof_error or _direct_refutation_blocker(claim, output, coverage)
        direct_proof = verdict == "refuted" and proof_blocker is None
        if not cited_abstract and not direct_proof and (not followup or second_followup):
            targets = _pending_targets(output, requirements, documents, initial_queries=initial_queries)
            if not targets and empirical_blocker and not followup:
                fallback = _empirical_followup(claim, empirical_blocker, hints)
                if fallback is not None:
                    targets = [fallback]
            retained = list(dict.fromkeys([
                *(eid for group in output.evidence_groups if group.role in {"support", "refute"}
                  for eid in group.evidence_ids),
                *(eid for field in coverage if field["status"] == "found" for eid in field["evidence_ids"]),
            ]))[:2]
            results = None
            if second_followup and targets:
                retained = list(dict.fromkeys([*retained, *(
                    eid for eid in rounds[0]["retained_ids"] if eid in scoped
                )]))[:2]
                targets, results = _additional_inspection(index, targets, rounds[0]["targets"], coverage,
                                                         requirements, documents, sent_candidate_ids,
                                                         primary_limit - len(retained))
            if targets:
                if followup or reserve_followup is None or reserve_followup():
                    followup = True
                    plan = {"targets": targets, "primary_ids": [item["evidence_id"] for item in candidates],
                            "sent_ids": sent_candidate_ids, "context_ids": body_ids, "retained_ids": retained}
                    if linked_table:
                        plan["linked_table_cell_id"] = linked_table["cell_evidence_id"]
                    rounds.append(plan)
                    if save_targets is not None:
                        save_targets(list(rounds))
                    if inspect(plan, second=second_followup, results=results):
                        continue
                    stop = "no_progress"
                else:
                    stop = "follow_up_budget"
            elif _pending_targets(output, requirements, documents):
                stop = "no_progress"
        if stop == "no_progress":
            coverage = _coverage(requirements, output, examined_scopes, sent_candidate_ids)
            if verdict == "refuted" and proof_error is None:
                proof_blocker = _direct_refutation_blocker(claim, output, coverage)
                direct_proof = proof_blocker is None
        role = {"supported": "support", "refuted": "refute"}.get(verdict)
        has_evidence = any(group.role == role and group.evidence_ids for group in output.evidence_groups)
        conflicting = {group.role for group in output.evidence_groups if group.evidence_ids}
        opposing = {"support", "refute"} <= conflicting
        checked = {field["status"] for field in coverage}
        follow_up_requested = bool(output.follow_up_queries or output.follow_up_targets)
        decisive_fields_checked = checked <= {"found"} and not follow_up_requested
        invalid_contradiction = verdict == "refuted" and not direct_proof
        required = {_field_key(field) for field in requirements}
        insufficiency_checks = {field["status"] for field in coverage
                                 if not cited_abstract or _field_key(field["requirement"]) in required}
        # In a bounded abstract, explanations in missing_fields do not add new requirements.
        can_call_insufficient = (
            "searched_missing" in insufficiency_checks and insufficiency_checks <= {"found", "searched_missing"}
            and not follow_up_requested and not conflicting & {"support", "refute"}
        )
        if cited_abstract:
            marked_missing = any(check.status == "searched_missing" and _field_key(check.requirement) in required
                                 and set(sent_candidate_ids).issuperset(check.evidence_ids) for check in output.coverage)
            can_call_insufficient = can_call_insufficient and full_abstract and marked_missing
        if role and opposing:
            decision_blockers.append("opposing_evidence")
        if verdict == "supported" and output.direct_contradiction is not None:
            decision_blockers.append("opposite_proof_unresolved")
        if verdict == "supported" and empirical_blocker:
            decision_blockers.append(empirical_blocker)
        if verdict == "supported" and has_evidence and decisive_fields_checked and not any(
            _covers_requirements(coverage, {_field_key(field) for field in requirements}, group.evidence_ids)
            for group in output.evidence_groups if group.role == "support"
        ):
            decision_blockers.append("unlinked_support_requirements")
        if invalid_contradiction:
            decision_blockers.append(proof_blocker)
        calculation_disagrees = (calculation is not None and role is not None
                                 and (verdict == "supported") != (calculation["verification"] == "matched"))
        if calculation_disagrees:
            decision_blockers.append("calculation_disagrees")
        if role and not has_evidence:
            decision_blockers.append("missing_directional_evidence")
        if role and not (decisive_fields_checked or direct_proof):
            decision_blockers.append("unchecked_requirements")
        if verdict == "insufficient" and not can_call_insufficient:
            decision_blockers.append("unverified_insufficiency")
        if verdict == "manual_conflict" and not opposing:
            decision_blockers.append("unpaired_conflicting_evidence")
        if decision_blockers:
            verdict = "pending"
        status = verdict if verdict in {"pending", "manual_conflict"} else "completed"
        if opposing and role:
            reason = "The assessment supplied both supporting and contradicting evidence. Their scopes need review."
        elif calculation_disagrees:
            reason = (f"The proposed verdict disagrees with the scoped calculation. The claim states "
                      f"{calculation['stated']} {calculation['unit']} and the result at that precision is "
                      f"{calculation['rounded_result']} {calculation['unit']}.")
        elif invalid_contradiction:
            reason = f"The proposed refutation did not pass the evidence checks. Proposed assessment: {output.reason}"
        elif "unlinked_support_requirements" in decision_blockers:
            reason = f"The stated conditions are not linked to the supporting result. Proposed assessment: {output.reason}"
        elif empirical_blocker == "unconfirmed_matched_comparison" and output.verdict == "supported":
            reason = f"The comparison between tested conditions needs further verification. Proposed assessment: {output.reason}"
        elif empirical_blocker and output.verdict == "supported":
            reason = f"The observed result and tested scope need further verification. Proposed assessment: {output.reason}"
        elif cited_abstract and output.verdict == "insufficient" and verdict == "pending":
            reason = f"The evidence-insufficiency checks did not pass for this cited abstract. Proposed assessment: {output.reason}"
        elif (output.verdict == "insufficient" or (role and not decisive_fields_checked)) and verdict == "pending":
            reason = f"Required sources have not been fully checked. Proposed assessment: {output.reason}"
        else:
            reason = output.reason
        if cited_abstract and verdict == "refuted" and direct_proof:
            proof = output.direct_contradiction
            ids = list(dict.fromkeys([*proof.evidence_ids, *proof.scope_evidence_ids, *ids]))
        finding = {**claim, "review_status": status, "verdict": verdict if status == "completed" else None,
                   "reason": reason, "evidence_ids": ids,
                   "evidence_groups": [group.model_dump() for group in output.evidence_groups],
                   "missing_fields": output.missing_fields, "coverage": coverage}
        if citation_ranges:
            finding["evidence_ranges"] = citation_ranges
        if direct_proof:
            finding["direct_contradiction"] = output.direct_contradiction.model_dump()
        elif output.verdict == "refuted" and status == "pending":
            finding["refutation_blocker"] = proof_blocker if has_evidence else "missing_refutation_evidence"
        if empirical_blocker:
            finding["empirical_support_blocker"] = empirical_blocker
        elif empirical and verdict == "supported":
            finding["empirical_support"] = output.empirical_support.model_dump(exclude_none=True)
        break
    candidate_ids = [item["evidence_id"] for item in candidates]
    finding["candidate_ids"] = candidate_ids
    finding["sent_candidate_ids"] = sent_candidate_ids
    if decision_attempts:
        unchecked = [field["requirement"] for field in finding.get("coverage", ()) if field["status"] != "found"]
        finding["decision_trace"] = {"attempts": decision_attempts, "blocker_codes": decision_blockers,
                                     "missing_requirements": list(dict.fromkeys(unchecked))}
    if cited_abstract:
        finding["evidence_boundary"] = evidence_boundary(sent_candidate_ids)
    if documents:
        finding["retrieval_trace"] = {
            "retrieved_evidence_ids": list(dict.fromkeys(eid for lane in breadth for eid in lane)),
            "body_window_evidence_ids": list(dict.fromkeys(eid for window in context_windows(candidate_ids)
                                                           for eid in window.expanded_ids)),
            "priority_evidence_ids": sent_priority,
        }
        if continuations := [
            {"from_evidence_id": source, "to_evidence_ids": list(window.evidence_ids)}
            for source, window in continuation_windows
            if source in sent_candidate_ids and set(sent_candidate_ids).issuperset(window.evidence_ids)
        ]:
            finding["retrieval_trace"]["page_continuations"] = continuations
    finding["searched_scope"] = examined_scopes
    finding["retrieval_paths"] = _sent_paths([*sent_linked_paths, *paths, *context["paths"]], sent_candidate_ids)
    if sent_table_context:
        finding["table_context"] = [
            {**{key: value for key, value in group.items() if key != "evidence"},
             "evidence_ids": [item["evidence_id"] for item in group["evidence"]]}
            for group in sent_table_context
        ]
    if sent_table_region:
        finding["table_region"] = {
            "reference_evidence_id": sent_table_region["reference_evidence_id"],
            "evidence_ids": [row["evidence_id"] for row in sent_table_region["evidence"]],
        }
    if sent_passages:
        finding["body_passages"] = sent_passages
    if sent_body_context:
        finding["body_context"] = sent_body_context
    finding["follow_up_performed"] = followup
    finding["follow_up_round_count"] = len(rounds)
    follow_up_evidence_ids.extend(eid for eid in sent_candidate_ids if eid in new_continuation_ids)
    if follow_up_evidence_ids:
        finding["follow_up_evidence_ids"] = list(dict.fromkeys(follow_up_evidence_ids))
    if "coverage" not in finding:
        offline_ids = finding["evidence_ids"] if model is None and finding["review_status"] == "completed" else ()
        if calculation and offline_ids:
            offline_ids = [*offline_ids, *calculation.get("scope_evidence_ids", ())]
        finding["coverage"] = _coverage(requirements, None, examined_scopes, sent_candidate_ids, offline_ids)
    if finding["verdict"] == "insufficient":
        stop = "source_range_exhausted"
    elif finding["review_status"] == "completed":
        stop = "verdict"
    elif any(field["status"] == "unreadable" for field in finding["coverage"]):
        stop = "unreadable"
    finding["stop_reason"] = stop or "requires_review"
    return finding, calculation


def _recover_cached_compounds(report: dict, version: dict) -> bool:
    if (version.get("kind") not in {"pdf_original", "pasted_original"}
            or version.get("version_id") != report.get("original_abstract", {}).get("version_id")
            or any(report.get(key) for key in ("revisions", "verifications", "candidate_proposal"))):
        return False
    claims = report["claims"]
    locked = {row["claim_id"] for row in report["findings"]
              if not row.get("retryable") or row.get("evidence_ids")}
    locked.update(row["claim_id"] for key in ("calculations", "finding_history") for row in report.get(key, ()))
    locked.update(report.get("follow_up_rounds", ()))
    locked.update(report.get("claim_followups", ()))
    ids = {claim["claim_id"] for claim in claims}
    counts = Counter(claim["sentence_id"] for claim in claims)
    updated = []
    replaced = set()
    for claim in claims:
        claim_id = claim["claim_id"]
        if claim_id not in locked and (counts[claim["sentence_id"]] == 1 or claim.get("compound")):
            recovered = recover_compound_claim(claim, version["text"])
            recovered_ids = {member["claim_id"] for member in recovered}
            if recovered and not ((recovered_ids - {claim_id}) & ids):
                replaced.add(claim_id)
                ids.update(recovered_ids)
                updated.extend(recovered)
                continue
        updated.append(claim)
    if not replaced:
        return False
    report["claims"] = updated
    for key in ("findings", "unresolved_items"):
        report[key] = [row for row in report.get(key, ()) if row.get("claim_id") not in replaced]
    return True


def audit_version(
    report: dict, model, commit, *, embedder=None, reranker=None, data_dir=None,
    reserve_followup: Callable[[str], bool] | None = None,
) -> None:
    version = report["reviewed_abstract"]
    if "claims" not in report:
        report["claims"] = extract_claims(version["text"], model)
        refresh_compound_findings(report)
        commit()
    elif _recover_cached_compounds(report, version):
        refresh_compound_findings(report)
        commit()
    index = EvidenceIndex(report["evidence"], embedder=embedder, reranker=reranker)
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
            refresh_compound_findings(report)
            commit()
    finally:
        close = getattr(index, "close", None)
        if close is not None:
            close()
    report["unresolved_items"] = [{"claim_id": row["claim_id"], "reason": row["reason"]}
                                  for row in report["findings"]
                                  if row["review_status"] != "completed" or row["verdict"] == "insufficient"]
    refresh_compound_findings(report)
