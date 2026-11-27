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
_HEADING_NUMBER = re.compile(r"^(?:\d+(?:\.\d+)*|[ivxlcdm]+)[.)]?\s+", re.I)
_FAILURE_HEADING = re.compile(
    r"(?:errors? (?:analys[ie]s|breakdowns?|categories)|failures? (?:analys[ie]s|breakdowns?|modes?|cases)|"
    r"analysis of (?:errors|failures)|错误分析|失败分析|错误分类)", re.I,
)
_RESULT_HEADING = re.compile(
    r"(?:(?:main|primary|overall|experimental|evaluation) )?results(?: and discussion)?|(?:主要|总体|实验|评估)?结果", re.I,
)
_FAILURE = re.compile(r"\b(?:errors?|fail(?:s|ed|ure[s]?)?|limitations?|shortcomings?)\b|错误|失败|局限", re.I)
_EMPIRICAL = re.compile(r"\b(?:evaluat\w*|experiments?|analys[ie]s|observ\w*|results?)\b|实验|评估|测得", re.I)
_ABLATION = re.compile(r"\bablation\b|消融", re.I)
_RESULT_EFFECT = re.compile(
    r"\b(?:improv\w*|outperform\w*|gains?|accuracy|precision|recall|scores?|performance)\b|提高|改进|优于|得分", re.I,
)
_ACROSS_TASKS = re.compile(
    r"\b(?:across|generaliz\w*|wide range|multiple|various|all)\b.{0,64}\b(?:tasks?|benchmarks?|datasets?)\b|"
    r"(?:跨|多个|所有|不同)(?:任务|基准|数据集)", re.I,
)
_STANDALONE = re.compile(r"\b(?:alone|naively|naïvely|standalone|in isolation)\b|单独|仅使用", re.I)
_SMALL_EFFECT = re.compile(r"\b(?:small|marginal|modest|limited|minor)\b.{0,28}\b(?:improv\w*|gains?|effects?)\b|有限提升|小幅提升", re.I)


@dataclass(frozen=True)
class _HeadingRoute:
    role: str
    label: re.Pattern[str]
    passage_terms: re.Pattern[str]
    cue: str = ""


_FAILURE_ROUTE = _HeadingRoute("error_analysis", _FAILURE_HEADING, _FAILURE,
                               "evaluation failure cases and error categories")
_GENERALIZATION_ROUTE = _HeadingRoute("comparative_results", _RESULT_HEADING, _RESULT_EFFECT)
_STANDALONE_ROUTE = _HeadingRoute("standalone_results", _RESULT_HEADING, _RESULT_EFFECT,
                                  "standalone small marginal improvement over baseline")


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


def _heading_role(proposition: str) -> _HeadingRoute | None:
    text = _normalize(proposition)
    if _ABLATION.search(text):
        return None  # An explicitly named study design needs its own source, not the general results page.
    if _EMPIRICAL.search(text) and _FAILURE.search(text):
        return _FAILURE_ROUTE
    if _RESULT_EFFECT.search(text):
        if _STANDALONE.search(text) and _SMALL_EFFECT.search(text):
            return _STANDALONE_ROUTE
        if _ACROSS_TASKS.search(text):
            return _GENERALIZATION_ROUTE
    return None


def _heading_context(index, route: _HeadingRoute, document: dict):
    matches = []
    for page in document.get("pages", ()):
        if not isinstance(page, dict) or page.get("status") in {"unreadable", "no_text"}:
            continue
        page_index, headings = page.get("page_index"), page.get("source_headings")
        if type(page_index) is not int or page_index < 0 or not isinstance(headings, (list, tuple)):
            continue
        for printed in headings:
            if not isinstance(printed, str) or not 0 < len(printed) <= 100:
                continue
            title = _HEADING_NUMBER.sub("", " ".join(printed.split())).rstrip(":.：")
            if route.label.fullmatch(title):
                matches.append((page_index, printed, title))
    if len({page for page, _, _ in matches}) != 1:
        return None, None
    page, heading, title = matches[0]
    query = f"{title} {route.cue}".strip()
    doc_id = document["doc_id"]
    windows = index.ranked_body_context([query], doc_id=doc_id, page_index=page)
    for window in windows:
        if route.passage_terms.search(window.text):
            scope = {"mode": "ranked", "query": query, "doc_id": doc_id, "document_ids": [doc_id],
                     "focused_page_index": page, "focused_source_heading": heading, "routing_role": route.role,
                     "examined_evidence_ids": list(window.evidence_ids)}
            return window, scope
    return None, None


def focused_source_context(index, proposition: str, documents):
    """Select at most one same-page context window from an anchor or a matching printed heading."""
    probe = _probe(proposition)
    main = [doc for doc in documents if doc.get("role") == "main"]
    if len(main) != 1:
        return None, None
    document, = main
    if probe is None:
        if route := _heading_role(proposition):
            return _heading_context(index, route, document)
        return None, None
    doc_id = document["doc_id"]
    result = index.search(probe.query, limit=12, expand=False, doc_id=doc_id)
    visited = set()
    for anchor in result["evidence"]:
        page = anchor["locator"]["page_index"]
        if page in visited or anchor.get("kind") != "body_text" or not probe.has_anchor(anchor["text"]):
            continue
        visited.add(page)
        windows = index.ranked_body_context([probe.query], doc_id=doc_id, page_index=page)
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
