"""Rank source table cells together with their row and column context."""

import math
import re
from collections import Counter
from collections.abc import Iterable

from .evidence import _cell_position, _exact_terms, _table_key, _terms


_COMMON_TERMS = frozenset(
    "the a an was is when whether are in of how to has had were this among on with for similar from against that "
    "both each and or by at their these which not continue italic bold empty table result results between over about "
    "around nearly denotes none".split()
)
_MEASUREMENT = re.compile(r"[\s+−.(-]*\d[\d\s−+.,/%∼~()<>±*a-dEe-]*")
_MISSING_VALUES = frozenset({"", "[empty]", "-", "–", "—", "n/a", "na", "nd"})


def _label_terms(text: str) -> set[str]:
    return {term for term in _terms(text) if term not in _COMMON_TERMS and not term.isdecimal()}


def _measurement_or_missing(text: str) -> bool:
    text = text.strip().casefold()
    return text in _MISSING_VALUES or _MEASUREMENT.fullmatch(text) is not None


def _matching_weight(terms: set[str], query: set[str], frequencies: Counter, population: int) -> float:
    return sum(math.log1p(population / frequencies[term]) for term in sorted(terms & query))


def rank_table_cells(query: str, cells: Iterable[dict], ranked_ids: Iterable[str], *, limit: int) -> list[str]:
    """Rank a single table's values with original row and column headers under a cell budget."""
    positioned = [(position, cell) for cell in cells
                  if (position := _cell_position(cell)) is not None and _table_key(cell) is not None]
    if limit <= 0 or not positioned:
        return []
    by_position = dict(positioned)
    source_ids = {cell["evidence_id"] for _, cell in positioned}
    seed_rank = list(dict.fromkeys(eid for eid in ranked_ids if eid in source_ids))
    if len(by_position) != len(positioned) or len({_table_key(cell) for _, cell in positioned}) != 1:
        return seed_rank[:limit]

    row_ids = sorted({row for row, _ in by_position if row})
    column_ids = sorted({col for _, col in by_position})
    descriptions = Counter(col for (row, col), cell in positioned if row and not _measurement_or_missing(cell["text"]))
    descriptive = sorted(col for col, count in descriptions.items() if count / len(row_ids) >= 0.7)
    labels = {position: _label_terms(cell["text"]) for position, cell in by_position.items()}
    row_labels = {row: [labels.get((row, col), set()) for col in descriptive] for row in row_ids}
    column_labels = {col: labels.get((0, col), set()) for col in column_ids}
    row_frequency = Counter(term for entries in row_labels.values() for term in set().union(*entries))
    column_frequency = Counter(term for entries in column_labels.values() for term in entries)
    query_labels = _label_terms(query)
    row_matches = {
        row: max((_matching_weight(terms, query_labels, row_frequency, len(row_ids)) for terms in entries), default=0)
        for row, entries in row_labels.items()
    }
    column_matches = {col: _matching_weight(terms, query_labels, column_frequency, len(column_ids))
                      for col, terms in column_labels.items()}
    exact_query, _ = _exact_terms(query)
    best_row_match = max(row_matches.values(), default=0)
    row_max = best_row_match or 1
    best_column_match = max((column_matches[col] for col in column_ids if col), default=0)
    if not seed_rank and not any(row_matches.values()) and not best_column_match and not exact_query:
        return []
    column_max = best_column_match or 1
    seeds = {eid: (len(seed_rank) - index) / len(seed_rank) for index, eid in enumerate(seed_rank)}

    def priority(position: tuple[int, int]) -> float:
        row, col = position
        cell = by_position[position]
        header = by_position.get((0, col), {}).get("text", "")
        exact_cell, _ = _exact_terms(cell["text"], header=header)
        exact_value = (bool(exact_query & exact_cell)
                       and row_matches[row] >= 0.8 * best_row_match
                       and (not best_column_match or column_matches[col] > 0))
        return (row_matches[row] / row_max + column_matches[col] / column_max
                + exact_value + 0.25 * seeds.get(cell["evidence_id"], 0)
                - 0.7 * (col in descriptive))

    values = (position for position, cell in by_position.items() if min(position) > 0
              and cell.get("cell_role", "value") == "value")
    chosen = {}
    for row, col in sorted(values, key=lambda position: (-priority(position), position)):
        for position in ((row, col), (row, 0), (0, col)):
            if position not in by_position:
                continue
            chosen[by_position[position]["evidence_id"]] = None
            if len(chosen) == limit:
                return list(chosen)
    return list(dict.fromkeys((*chosen, *seed_rank)))[:limit]
