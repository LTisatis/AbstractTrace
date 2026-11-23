"""Evidence retrieval, scoped calculations, and independent original verdicts."""

import hashlib
import re
from base64 import b64encode
from collections.abc import Callable, Iterable, Sequence
from decimal import Decimal
from itertools import zip_longest
from math import isfinite
from pathlib import Path
from typing import Literal

from pydantic import Field

from .calculations import comparison_for_claim
from .compound import (
    aggregate_compound_findings, attach_compound, offline_compound_parts, requires_whole_sentence,
    shared_procedure_parts,
)
from .evidence import EvidenceIndex, with_method_scope
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
    source_kind: Literal["methods", "results", "proof"] | None = None


class DirectContradiction(Schema):
    evidence_ids: list[str]
    scope_evidence_ids: list[str] = Field(default_factory=list)
    subject_and_conditions: Literal["same", "different", "uncertain"]
    relation: Literal["opposite_result", "in_scope_counterexample", "uncertain"]
    scope_reason: str
    supplementary_fields: list[str]


class ScopedJudgement(Judgement):
    coverage: list[FieldCheck] = Field(default_factory=list)
    follow_up_targets: list[FollowUpTarget] = Field(default_factory=list)
    direct_contradiction: DirectContradiction | None = None


_SCOPE_KEYS = ("section", "table_label", "page_index", "doc_id")
_SOURCE_SECTIONS = {"methods": "Methods", "results": "Results"}
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
            failures = _page_failures(scope, documents)
            scope["unreadable_locations"] = failures
            scope["complete"] = explicit_location and scope["complete"] and not failures
            section_hint = proof_query and target["section"] is not None and all(
                target[key] is None for key in ("page_index", "table_label")
            )
            if (section_hint or not scope["complete"]) and (not target["table_label"] or not scope["matching_count"]):
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
    required = {_field_key(field["requirement"]) for field in coverage}
    unresolved = {_field_key(field["requirement"]) for field in coverage if field["status"] != "found"}
    core = {_field_key(field) for field in claim["requirements"]} - supplementary
    if not unresolved <= supplementary <= required:
        return "unchecked_decisive_requirement"
    scope_ids = set(proof.scope_evidence_ids or proof.evidence_ids)
    if not any(_field_key(field["requirement"]) in core and field["status"] == "found"
               and scope_ids.intersection(field["evidence_ids"]) for field in coverage):
        return "missing_scope_coverage"
    proof_ids = scope_ids | set(proof.evidence_ids)
    if proof.scope_evidence_ids and any(_field_key(field["requirement"]) in core
                                       and not proof_ids.intersection(field["evidence_ids"]) for field in coverage):
        return "missing_scope_coverage"
    if output.follow_up_queries or any(not target.requirement or _field_key(target.requirement) not in supplementary
                                       for target in output.follow_up_targets):
        return "outstanding_decisive_followup"
    return None


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
    return atoms if all("compound" in atom for atom in atoms) else []


def extract_claims(text: str, model=None) -> list[dict]:
    parts = sentences(text)
    if model is None:
        results = []
        for part in parts:
            split = offline_compound_parts(part["text"])
            atoms = _compound_claims(part, split) if split else []
            results.extend(atoms or [_plain_claim(part)])
        return results

    parsed = model.call("extract", {"sentences": parts}, Extraction)
    by_sentence = {part["sentence_id"]: part for part in parts}
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

    result = {}
    for atom in anchored:
        sentence_id = atom["sentence_id"]
        part = by_sentence[sentence_id]
        whole = sentence_id in whole_sentences
        unsplit = (counts[sentence_id] == 1 and atom["kind"] not in {"background", "purpose"}
                   and atom["quote"].rstrip(".?!") == part["text"].rstrip(".?!"))
        if whole or unsplit:
            split = shared_procedure_parts(part["text"])
            recovered = _compound_claims(part, split) if split else []
            if recovered:
                result.update((member["claim_id"], member) for member in recovered)
                continue
        if whole:
            atom = _plain_claim(part)
            atom["logic_review_required"] = True
        result[atom["claim_id"]] = atom
    missing = whole_sentences - {atom["sentence_id"] for atom in anchored}
    for part in parts:
        if part["sentence_id"] in missing:
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


def blind_evidence(evidence: list[dict]) -> list[dict]:
    permitted = {"evidence_id", "doc_id", "kind", "text", "locator", "row_header", "column_header",
                 "column_header_path", "cell_role", "note_marker", "note_scope"}
    return [{key: item[key] for key in sorted(item.keys() & permitted)}
            for item in sorted(evidence, key=lambda item: item["evidence_id"])]


def validate_ids(ids: list[str], evidence: Iterable[dict]) -> None:
    allowed = {item["evidence_id"] for item in evidence}
    if not allowed.issuperset(ids):
        raise ModelUnavailable("model cited evidence outside the supplied scope")


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


def _offline(claim: dict, candidates: list[dict], comparison) -> dict:
    if claim.get("logic_review_required"):
        return _pending(claim, "This logical relationship needs a scope-aware review.")
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
    follow_up_evidence_ids = []
    sent_candidate_ids = []
    sent_body_context = []
    sent_passages = []
    sent_table_context = []
    inspected_body_windows = []
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

    def inspect(targets: Sequence[dict], previous: list[str], known: set[str], *, context_ids=()) -> bool:
        results = targeted_evidence(index, targets, documents=documents, prior_ids=sorted(known - set(context_ids)))
        examined_scopes.extend(results["scopes"])
        for scope in results["scopes"]:
            if scope["mode"] == "direct" and scope["matching_count"] and scope["table_label"] is None:
                inspected_body_windows.extend(index.ranked_body_context(
                    [scope["query"]], doc_id=scope["doc_id"], doc_version_id=scope["doc_version_id"],
                    page_index=scope["page_index"], section=scope["section"],
                ))
        remember(results)
        selected = [item["evidence_id"] for item in results["evidence"]]
        if previous and len(selected) == 8 and previous[0] not in selected:
            # A partial check can add a missing condition while an initial hit is
            # still needed to judge the result. Keep that hit if space allows.
            exhaustive = {eid for scope in results["scopes"] if scope.get("complete")
                          for eid in scope["examined_evidence_ids"]}
            grouped = {eid for unit in index.evidence_units(selected) if len(unit) > 1 for eid in unit}
            for offset in reversed(range(len(selected))):
                if selected[offset] not in exhaustive and selected[offset] not in grouped:
                    selected[offset] = previous[0]
                    break
        prioritized.extend(selected)
        follow_up_evidence_ids.extend(eid for eid in selected if eid not in known)
        if follow_up_evidence_ids or any(scope.get("complete") for scope in results["scopes"]):
            return True
        if not inspected_body_windows:
            return False
        primary = candidate_order()
        table_ids = [item["evidence_id"] for item in index.table_context(primary)["evidence"]]
        return any(not known.issuperset(group["new_evidence_ids"])
                   for group in index.pack_body_context(context_windows(), [*primary, *table_ids]))

    def context_windows():
        return list(dict.fromkeys(inspected_body_windows or body_windows))[:2]

    def candidate_order() -> list[str]:
        # Keep context from each lane together when combining support and counter searches.
        ordered = dict.fromkeys(prioritized[:8])
        for unit in _interleave_units(index, breadth):
            if len(ordered.keys() | set(unit)) <= 8:
                ordered.update(dict.fromkeys(unit))
        return list(ordered)

    gather([claim["support_query"], claim["counter_query"]])
    body_windows = []
    if model is None or claim["kind"] not in {"background", "purpose"}:
        body_windows = index.ranked_body_context([claim["support_query"], claim["counter_query"]])
    if additional_evidence:
        supplements = index.get(item["evidence_id"] for item in additional_evidence)
        prioritized.extend(item["evidence_id"] for item in supplements)
        by_id.update((item["evidence_id"], item) for item in supplements)
    followup = bool(saved_targets or additional_evidence)
    if saved_targets:
        previous = candidate_order()
        context = index.table_context(previous)
        inspect(saved_targets, previous, set(previous) | {item["evidence_id"] for item in context["evidence"]})
    calculation = None
    while True:
        candidates = [by_id[eid] for eid in candidate_order()]
        passages = index.body_passages(item["evidence_id"] for item in candidates)
        context = index.table_context(item["evidence_id"] for item in candidates)
        context_by_id = {item["evidence_id"]: item for item in context["evidence"]}
        table_context = [
            {"cell_evidence_id": eid, "evidence": blind_evidence([context_by_id[related] for related in related_ids])}
            for eid, related_ids in sorted(context["by_seed"].items())
        ]
        scoped = {item["evidence_id"]: item for item in [*candidates, *context["evidence"]]}
        comparison = None if claim.get("logic_review_required") else comparison_for_claim(
            claim["proposition"], with_method_scope(scoped.values(), paths)
        )
        calculation = comparison[0] if comparison else None
        body_context = index.pack_body_context(context_windows(), scoped)
        body_ids = [eid for group in body_context for eid in group["new_evidence_ids"]]
        body_rows = {row["evidence_id"]: row for row in index.get(body_ids)}
        scoped.update(body_rows)
        if model is None:
            finding = _offline(claim, [*candidates, *body_rows.values()], comparison)
            sent_candidate_ids = list(scoped)
            sent_body_context = body_context
            sent_passages = passages
            sent_table_context = table_context
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
                        "Read listed body passages in source order. Match the study, model, intervention and conditions. "
                        "Separate methods of the current study from cited or sensitivity analyses. "
                        "For an atomic claim, when the paper states the opposite or provides a witness within the claimed "
                        "domain, return refuted and a direct_contradiction. Put the opposite result or witness in evidence_ids and any "
                        "separate same-study, same-conditions anchors in scope_evidence_ids. Put both sets in one refute "
                        "group. Use subject_and_conditions=same only when the anchors establish that match, and say how "
                        "in scope_reason. Give the exact names of optional unchecked fields in supplementary_fields. "
                        "An unreported instrument or operational detail can be optional for a directly stated qualitative "
                        "contradiction. An explicit population, model, intervention, comparator, or claim condition is essential. "
                        "If scope or another decisive condition is unresolved, request up to two follow_up_targets. "
                        "Use the exact missing field and concise source terms. For universal negatives, search for a "
                        "constructive witness using the same architecture and assumptions. Prefer source_kind methods, "
                        "results or proof when relevant. Name a section, table or page when known. Ordinary ranked "
                        "search is not a complete inspection."
                    )}
        if table_context:
            material["table_context"] = table_context
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
        if claim.get("logic_review_required"):
            material["logic_review_required"] = True
            material["coverage_instructions"] += (
                " This sentence requires a full logic review. Use ordinary evidence groups and leave "
                "direct_contradiction empty."
            )
        material.update(figure_material(candidates, documents, data_dir))
        try:
            output = model.call("judge", material, ScopedJudgement)
            sent_candidate_ids = list(dict.fromkeys([
                *(item["evidence_id"] for item in material["evidence"]),
                *(item["evidence_id"] for group in table_context for item in group["evidence"]),
                *body_ids,
            ]))
            sent_body_context = body_context
            sent_passages = passages
            sent_table_context = table_context
            ids = list(dict.fromkeys(eid for group in output.evidence_groups for eid in group.evidence_ids))
            try:
                validate_ids(ids, scoped.values())
            except ModelUnavailable:
                output = model.call("judge", {**material, "retry": "Review the supplied IDs carefully."}, ScopedJudgement)
                ids = list(dict.fromkeys(eid for group in output.evidence_groups for eid in group.evidence_ids))
                validate_ids(ids, scoped.values())
        except ModelUnavailable as exc:
            finding = _pending(claim, str(exc))
            finding["retryable"] = isinstance(exc, ModelPaused)
            break

        coverage = _coverage(requirements, output, examined_scopes, sent_candidate_ids)
        verdict = output.verdict
        proof_error = proof_blocker = None
        if verdict == "refuted":
            proof_error = _contradiction_error(output, sent_candidate_ids)
            proof_blocker = proof_error or _direct_refutation_blocker(claim, output, coverage)
        direct_proof = verdict == "refuted" and proof_blocker is None
        targets = [] if direct_proof else _pending_targets(output, requirements, documents)
        if targets and not followup:
            if reserve_followup is None or reserve_followup():
                followup = True
                if save_targets is not None:
                    save_targets(targets)
                if inspect(targets, [item["evidence_id"] for item in candidates], set(sent_candidate_ids),
                           context_ids=body_ids):
                    continue
                stop = "no_progress"
            else:
                stop = "follow_up_budget"
        if stop == "no_progress":
            coverage = _coverage(requirements, output, examined_scopes, sent_candidate_ids)
            if verdict == "refuted" and proof_error is None:
                proof_blocker = _direct_refutation_blocker(claim, output, coverage)
                direct_proof = proof_blocker is None
        role = {"supported": "support", "refuted": "refute"}.get(verdict)
        has_evidence = any(group.role == role and group.evidence_ids for group in output.evidence_groups)
        conflicting = {group.role for group in output.evidence_groups if group.evidence_ids}
        checked = {field["status"] for field in coverage}
        follow_up_requested = bool(output.follow_up_queries or output.follow_up_targets)
        decisive_fields_checked = checked <= {"found"} and not follow_up_requested
        invalid_contradiction = (verdict == "refuted" and output.direct_contradiction is not None
                                 and (proof_error is not None or proof_blocker in {
                                     "missing_scope_coverage", "conflicting_support",
                                 }))
        can_call_insufficient = (
            "searched_missing" in checked and checked <= {"found", "searched_missing"}
            and not follow_up_requested
        )
        if (invalid_contradiction or (role and not (has_evidence and (decisive_fields_checked or direct_proof)))
                or (verdict == "insufficient" and not can_call_insufficient)
                or (verdict == "manual_conflict" and not {"support", "refute"} <= conflicting)):
            verdict = "pending"
        status = verdict if verdict in {"pending", "manual_conflict"} else "completed"
        if invalid_contradiction:
            reason = f"The proposed refutation did not pass the evidence checks. Proposed assessment: {output.reason}"
        elif (output.verdict == "insufficient" or (role and not decisive_fields_checked)) and verdict == "pending":
            reason = f"Required sources have not been fully checked. Proposed assessment: {output.reason}"
        else:
            reason = output.reason
        finding = {**claim, "review_status": status, "verdict": verdict if status == "completed" else None,
                   "reason": reason, "evidence_ids": ids,
                   "evidence_groups": [group.model_dump() for group in output.evidence_groups],
                   "missing_fields": output.missing_fields, "coverage": coverage}
        if direct_proof:
            finding["direct_contradiction"] = output.direct_contradiction.model_dump()
        elif output.verdict == "refuted" and status == "pending":
            finding["refutation_blocker"] = proof_blocker if has_evidence else "missing_refutation_evidence"
        break
    candidate_ids = [item["evidence_id"] for item in candidates]
    finding["candidate_ids"] = candidate_ids
    finding["sent_candidate_ids"] = sent_candidate_ids
    finding["searched_scope"] = examined_scopes
    finding["retrieval_paths"] = _sent_paths([*paths, *context["paths"]], sent_candidate_ids)
    if sent_table_context:
        finding["table_context"] = [
            {"cell_evidence_id": group["cell_evidence_id"],
             "evidence_ids": [item["evidence_id"] for item in group["evidence"]]}
            for group in sent_table_context
        ]
    if sent_passages:
        finding["body_passages"] = sent_passages
    if sent_body_context:
        finding["body_context"] = sent_body_context
    finding["follow_up_performed"] = followup
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


def audit_version(
    report: dict, model, commit, *, embedder=None, reranker=None, data_dir=None,
    reserve_followup: Callable[[str], bool] | None = None,
) -> None:
    version = report["reviewed_abstract"]
    if "claims" not in report:
        report["claims"] = extract_claims(version["text"], model)
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
