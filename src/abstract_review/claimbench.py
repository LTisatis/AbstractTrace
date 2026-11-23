"""Local CLAIM-BENCH passage alignment against the PDF review extractor."""

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

from .ingest import PARSER_VERSION, parse_pdf
from .scoring import _DEFAULT_RANK_CUTOFFS, _ordered_ids, _parse_predictions, _ratio


_DATASET = "claimbench-quoted-passages"


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).replace("\u00ad", "").split())


def _annotations(content: object):
    flat = isinstance(content, list)
    rows = content if flat else content.get("annotations") if isinstance(content, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("expected a nonempty CLAIM-BENCH annotation array")
    seen = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"annotation {index} must be an object")
        if flat:
            claim_id, claim, excerpts = row.get("claim_id"), row.get("claim_text"), [row.get("evidence_text")]
        else:
            claim_id, claim, excerpts = index, row.get("claim"), row.get("evidences")
        if type(claim_id) is int:
            claim_id = str(claim_id)
        if not isinstance(claim_id, str) or not claim_id.strip() or claim_id != claim_id.strip() or claim_id in seen:
            raise ValueError(f"annotation {index} has a missing or repeated claim ID")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(f"annotation {index} has no claim text")
        if (not isinstance(excerpts, list) or not excerpts
                or any(not isinstance(item, str) or not _normalized(item) for item in excerpts)):
            raise ValueError(f"annotation {index} has no quoted evidence text")
        seen.add(claim_id)
        yield claim_id, claim, excerpts


def _line_wrap(text: str, before: dict, next_text: str, after: dict) -> bool:
    if len(text) < 2 or not (text.endswith("-") and text[-2].isalpha() and next_text[0].isalpha()):
        return False
    left, right = before["locator"], after["locator"]
    if left.get("section") != right.get("section"):
        return False
    x0, y0, _, y1 = left["bbox"]
    next_x0, next_y0, _, next_y1 = right["bbox"]
    line_height = min(y1 - y0, next_y1 - next_y0)
    return abs(next_x0 - x0) <= line_height and 0 < next_y0 - y0 <= 1.75 * line_height


def _pages(evidence: list[dict], *, join_line_wraps: bool = False) -> dict[int, tuple[str, list[tuple[int, int, dict]]]]:
    grouped = {}
    for item in evidence:
        if item["kind"] != "body_text":
            continue
        text = _normalized(item["text"])
        if text:
            grouped.setdefault(item["locator"]["page_index"], []).append((text, item))
    pages = {}
    for page_index, parts in grouped.items():
        text_parts, spans = [], []
        offset = 0
        for index, (text, item) in enumerate(parts):
            join = join_line_wraps and index + 1 < len(parts) and _line_wrap(text, item, *parts[index + 1])
            if join:
                text = text[:-1]
            text_parts.append(text)
            spans.append((offset, offset + len(text), item))
            offset += len(text)
            if not join:
                text_parts.append(" ")
                offset += 1
        pages[page_index] = ("".join(text_parts), spans)
    return pages


def _align(excerpt: str, pages: dict, *, normalization: str) -> dict:
    pattern = re.compile(r"(?<!\w)(?=(" + re.escape(_normalized(excerpt)) + r")(?!\w))")
    match = None
    for page_index, (text, spans) in pages.items():
        for occurrence in pattern.finditer(text):
            if match is not None:
                return {"status": "ambiguous", "evidence_ids": [], "anchors": []}
            first, last = occurrence.span(1)
            records = [item for start, end, item in spans if start < last and end > first]
            match = {
                "status": "aligned", "normalization": normalization,
                "evidence_ids": [item["evidence_id"] for item in records],
                "anchors": [{"evidence_id": item["evidence_id"], "doc_id": item["doc_id"], "page_index": page_index,
                             "bbox": item["locator"]["bbox"]} for item in records],
            }
    return match or {"status": "unmatched", "evidence_ids": [], "anchors": []}


def _align_passage(excerpt: str, plain: dict, joined: dict) -> dict:
    result = _align(excerpt, plain, normalization="normalized_exact")
    if result["status"] == "unmatched":
        return _align(excerpt, joined, normalization="line_wrap_hyphen")
    return result


def prepare_claimbench(
    pdf: Path, annotations: Path, *, output_dir: Path, paper_family_id: str, split: str, dataset_version: str,
) -> tuple[dict, dict]:
    """Convert one matching author PDF and annotation file without fetching either."""
    if not all(isinstance(value, str) and value.strip() for value in (paper_family_id, split, dataset_version)):
        raise ValueError("paper family, split, and dataset version must be nonempty")
    try:
        content = annotations.read_bytes()
        claims = list(_annotations(json.loads(content.decode("utf-8-sig"))))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{annotations}: invalid UTF-8 annotation JSON") from exc
    document = parse_pdf(pdf)
    pages = _pages(document.evidence)
    joined = _pages(document.evidence, join_line_wraps=True)
    pdf_sha256 = document.document["sha256"]
    sample_id = f"claimbench-{pdf_sha256}"
    try:
        pdf_path = os.path.relpath(pdf.resolve(), output_dir.resolve())
    except ValueError:  # Windows paths on different drives cannot share a relative path.
        pdf_path = str(pdf.resolve())
    samples = [{
        "sample_id": sample_id, "paper_family_id": paper_family_id.strip(), "split": split.strip(),
        "pdf": pdf_path, "pdf_sha256": pdf_sha256,
        "claims": [{"claim_id": claim_id, "text": claim} for claim_id, claim, _ in claims],
    }]
    labels = [
        {"sample_id": sample_id, "claim_id": claim_id,
         "passages": [_align_passage(excerpt, pages, joined) for excerpt in excerpts]}
        for claim_id, _, excerpts in claims
    ]
    return {"schema_version": 1, "samples": samples}, {
        "schema_version": 1, "dataset": _DATASET, "dataset_version": dataset_version.strip(),
        "parser_version": PARSER_VERSION, "pdf_sha256": pdf_sha256,
        "annotations_sha256": hashlib.sha256(content).hexdigest(), "labels": labels,
    }


def score_claimbench(predictions: object, gold: object) -> dict:
    """Count full annotated passage hits, separately from exact alignment coverage."""
    predicted = _parse_predictions(predictions)
    if (not isinstance(gold, dict) or gold.get("dataset") != _DATASET or type(gold.get("schema_version")) is not int
            or gold["schema_version"] != 1 or not isinstance(gold.get("labels"), list)):
        raise ValueError("expected CLAIM-BENCH aligned passage gold")
    seen, cases = set(), []
    ranked_hits = dict.fromkeys(_DEFAULT_RANK_CUTOFFS, 0)
    within_cutoff = dict.fromkeys(_DEFAULT_RANK_CUTOFFS, 0)
    totals = dict.fromkeys(("aligned", "unmatched", "ambiguous"), 0)
    cited_hits = rank_unavailable = 0
    for label in gold["labels"]:
        if (not isinstance(label, dict) or not isinstance(label.get("sample_id"), str)
                or not isinstance(label.get("claim_id"), str) or not isinstance(label.get("passages"), list)
                or not label["passages"]):
            raise ValueError("invalid CLAIM-BENCH claim gold record")
        key = label["sample_id"], label["claim_id"]
        if key in seen:
            raise ValueError(f"duplicate CLAIM-BENCH gold claim {key}")
        seen.add(key)
        prediction = predicted.get(key)
        cited = prediction.evidence_ids if prediction else frozenset()
        candidates = prediction.candidate_ids if prediction else ()
        for number, passage in enumerate(label["passages"], 1):
            status = passage.get("status") if isinstance(passage, dict) else None
            if not isinstance(status, str) or status not in totals:
                raise ValueError("invalid CLAIM-BENCH passage alignment status")
            ids = set(_ordered_ids(passage.get("evidence_ids"), "passage.evidence_ids"))
            if bool(ids) != (status == "aligned"):
                raise ValueError("only aligned CLAIM-BENCH passages may have evidence IDs")
            totals[status] += 1
            cited_hit = ids <= cited if ids else None
            cited_hits += bool(cited_hit)
            ranked = None
            if ids:
                for k in within_cutoff:
                    within_cutoff[k] += len(ids) <= k
                if candidates is None:
                    rank_unavailable += 1
                else:
                    ranked = {str(k): ids <= set(candidates[:k]) for k in _DEFAULT_RANK_CUTOFFS}
                    for k in ranked_hits:
                        ranked_hits[k] += ranked[str(k)]
            cases.append({
                "sample_id": key[0], "claim_id": key[1], "passage_number": number, "alignment_status": status,
                "evidence_ids": sorted(ids), "evidence_count": len(ids), "citation_hit": cited_hit, "ranked_hit": ranked,
            })
    if predicted.keys() - seen:
        raise ValueError("some predictions have no CLAIM-BENCH gold claim")
    return {
        "schema_version": 1, "dataset": _DATASET, "alignment": totals,
        "alignment_coverage": _ratio(totals["aligned"], sum(totals.values())),
        "cited_passage_recall": _ratio(cited_hits, totals["aligned"]),
        "quoted_passage_recall_at_k": {
            str(k): None if rank_unavailable else _ratio(hits, totals["aligned"]) for k, hits in ranked_hits.items()
        },
        "passages_within_k": {str(k): _ratio(count, totals["aligned"]) for k, count in within_cutoff.items()},
        "cases": cases,
    }
