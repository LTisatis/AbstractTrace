"""Derive narrowly scoped source probes without changing the claim under review."""

import re
from dataclasses import dataclass, replace

from .evidence import _normalize, _terms


_NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_PERCENT_UNIT = r"(?:%|percentage\s+points?\b|percent\b|p\.?p\.?\b)"
_PAIR = re.compile(
    rf"(?<![\w.])({_NUMBER})\s*(?:{_PERCENT_UNIT})?\s+and\s+({_NUMBER})\s*{_PERCENT_UNIT}", re.I,
)
_COUNT = (r"(?P<count>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?:\s*(?P<scale>[km]))?\s+"
          r"(?P<unit>tokens?|words?|characters?|examples?|samples?|records?|test cases?)\b")
_COUNT_ON_LINE = re.compile(rf"(?<![\w.,]){_COUNT}", re.I)
_COUNT_UPPER_BOUND = re.compile(rf"\b(?:up to|at most|as many as|a maximum of)\s+{_COUNT}", re.I)
_CAPACITY_CONTEXT_LINES = 4
_COMPARISON = re.compile(
    r"\b(?:against|baselines?|compar(?:e|ed|ison|ing)|decreas\w*|differen\w*|gain\w*|improv\w*|"
    r"increas\w*|outperform\w*|previous|prior|reduc\w*)\b", re.I,
)
_REMOVAL_ACTION = r"remov\w*|ablat\w*|disabl\w*|omit\w*|exclud\w*"
_REMOVAL = re.compile(rf"\b(?:{_REMOVAL_ACTION}|without)\b", re.I)
_INTERVENTION = re.compile(
    r"\b(?P<action>removing|ablating|disabling|omitting|excluding)\s+"
    r"(?P<subject>[\w-]+(?:\s+[\w-]+){0,4}?)\s+"
    r"(?:(?:does|did)\s+not\s+)?"
    r"(?:lead(?:s)? to|causes?|yields?|results? in|prevents?|avoids?|eliminates?|increases?|reduces?)\s+"
    r"(?P<outcome>[^.,;!?]+)", re.I,
)
_STOP = frozenset({"a", "an", "and", "of", "the", "to"})


def _count_key(match: re.Match) -> tuple[str, str, str]:
    return match["count"].replace(",", ""), match["scale"] or "", match["unit"].removesuffix("s")


@dataclass(frozen=True)
class _Probe:
    query: str
    values: tuple[str, ...] = ()
    subject: frozenset[str] = frozenset()
    outcome: frozenset[str] = frozenset()
    capacity: tuple[str, str, str] | None = None

    def has_anchor(self, text: str) -> bool:
        if self.capacity is not None:
            return any(_count_key(match) == self.capacity for match in _COUNT_ON_LINE.finditer(_normalize(text)))
        if self.values:
            numbers = set(re.findall(rf"(?<![\w.])({_NUMBER})(?![\w.])", _normalize(text)))
            return set(self.values) <= numbers and bool(_COMPARISON.search(text))
        return self.outcome <= _terms(text)

    def has_context(self, text: str) -> bool:
        if self.capacity is not None or self.values:
            return self.has_anchor(text)
        return self.outcome | self.subject <= _terms(text) and bool(_REMOVAL.search(text))


def _probe(proposition: str) -> _Probe | None:
    text = _normalize(proposition)
    if (pair := _PAIR.search(text)) and pair[1] != pair[2] and _COMPARISON.search(text):
        first, second = pair.groups()
        # A unit may be written differently in the source. These aliases are only for retrieval.
        return _Probe(f"{first}% {second}% {first} {second} comparison results", (first, second))
    if match := _INTERVENTION.search(text):
        subject = frozenset(_terms(match["subject"]) - _STOP)
        outcome = frozenset(_terms(match["outcome"]) - _STOP)
        if subject and outcome:
            return _Probe(f"{match['action']} {match['subject']} {match['outcome']}", subject=subject, outcome=outcome)
    if match := _COUNT_UPPER_BOUND.search(text):
        start = match.start("count")
        return _Probe(text[start:match.end()], capacity=_count_key(match))
    return None


def focused_source_context(index, proposition: str, documents):
    """Select a same-page span only after finding and confirming its lexical anchor."""
    probe = _probe(proposition)
    main = [doc["doc_id"] for doc in documents if doc.get("role") == "main"]
    if probe is None or len(main) != 1:
        return None, None
    result = index.search(probe.query, limit=12, expand=False, doc_id=main[0])
    visited = set()
    for anchor in result["evidence"]:
        page = anchor["locator"]["page_index"]
        if page in visited or anchor.get("kind") != "body_text" or not probe.has_anchor(anchor["text"]):
            continue
        visited.add(page)
        windows = index.ranked_body_context([probe.query], doc_id=main[0], page_index=page)
        for window in windows:
            anchor_id = anchor["evidence_id"]
            if anchor_id in window.evidence_ids and probe.has_context(window.text):
                if probe.capacity is not None:
                    center = window.expanded_ids.index(anchor_id)
                    start = max(0, min(center - 1, len(window.expanded_ids) - _CAPACITY_CONTEXT_LINES))
                    ids = window.expanded_ids[start:start + _CAPACITY_CONTEXT_LINES]
                    body = "\n".join(record["text"] for record in index.get(ids))
                    window = replace(window, text=body, evidence_ids=ids, expanded_ids=ids)
                scope = {**result["scope"], "mode": "ranked", "focused_page_index": page,
                         "focus_anchor_id": anchor_id}
                return window, scope
        if len(visited) == 2:
            break
    return None, None
