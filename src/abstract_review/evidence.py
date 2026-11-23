"""Scoped paper retrieval with exact matching and source-anchored links."""

import math
import re
import sqlite3
import unicodedata
from collections import defaultdict, deque
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


_HAN = re.compile(r"([\u3400-\u9fff\U00020000-\U0003134f]+)")
_WORD = re.compile(r"[^\W_]+")
_TABLE_NUMBER = r"(?:[a-z]+\d+|\d+(?:[.-]\d+)*[a-z]?)(?![a-z0-9])"
_TABLE_MENTION = rf"(?:(?<![a-z0-9])(?:table|tab\.)\s+|表\s*)({_TABLE_NUMBER})"
_TABLE_REFERENCE = re.compile(_TABLE_MENTION)
_TABLE_CAPTION = re.compile(rf"^\s*{_TABLE_MENTION}\s*(?:[:.—-]|$)")
_FIGURE_MENTION = rf"(?:(?<![a-z0-9])(?:figure|fig\.)\s+|图\s*)({_TABLE_NUMBER})"
_FIGURE_REFERENCE = re.compile(_FIGURE_MENTION)
_FIGURE_PANEL = re.compile(rf"{_FIGURE_MENTION}\s*\(([a-z])\)")
_EXPERIMENT = re.compile(
    r"(?:(?<![a-z0-9])(?:experiment\s+|exp\.\s*)|实验(?:编号|号)?\s*)"
    r"[:#]?\s*([a-z]*\d+(?:[-.]\d+)*[a-z]?)(?![a-z0-9])"
)
_UNIT = r"%|百分点|percentage\s+points?|p\.?p\.?\b"
_NUMBER = re.compile(rf"(?<![a-z0-9_.])([+-]?(?:\d+(?:\.\d+)?|\.\d+))(?:\s*({_UNIT}))?")
_MAX_CANDIDATES = 24
_SECTION_NUMBER = re.compile(r"^(?:\d+(?:\.\d+)*|[ivx]+)[.)]?\s+")
_LINK_PRIORITY = {
    "same_experiment": 0,
    "table_reference": 1,
    "figure_reference": 1,
    "table_caption": 2,
    "table_continuation": 2,
    "table_note": 2,
    "figure_caption": 2,
    "figure_panel": 2,
    "row_header": 3,
    "column_header": 4,
    "same_table": 5,
}


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> Iterable[Iterable[float]]: ...


class SentenceTransformerEmbedder:
    """Use an explicitly installed model snapshot and disable model downloads."""

    def __init__(self, model_path: Path | str) -> None:
        path = Path(model_path).resolve()
        if not path.is_dir() or not (path / "modules.json").is_file():
            raise ValueError(f"expected a local Sentence Transformers model snapshot: {path}")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ValueError("local model retrieval requires sentence-transformers") from exc
        self.model = SentenceTransformer(
            str(path), local_files_only=True, trust_remote_code=False, device="cpu"
        )

    def encode(self, texts: list[str]) -> Iterable[Iterable[float]]:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().replace("−", "-")


def _terms(text: str, *, index: bool = False) -> set[str]:
    parts = _HAN.split(_normalize(text))
    tokens = set()
    for offset, part in enumerate(parts):
        if offset % 2 == 0:
            tokens.update(_WORD.findall(part))
        else:
            tokens.update(part[i : i + 2] for i in range(len(part) - 1))
            if index or len(part) == 1:
                tokens.update(part)
    return tokens


def _exact_terms(text: str, *, header: str = "") -> tuple[set[str], str]:
    normalized = _normalize(text)
    tokens = {f"table:{label}" for label in _TABLE_REFERENCE.findall(normalized)}
    tokens.update(f"figure-panel:{figure}:{panel}" for figure, panel in _FIGURE_PANEL.findall(normalized))
    without_tables = _TABLE_REFERENCE.sub(" ", _FIGURE_PANEL.sub(" ", normalized))
    header_unit = re.search(_UNIT, _normalize(header))
    for number, unit in _NUMBER.findall(without_tables):
        tokens.add(f"number:{number}:{'pp' if unit and unit != '%' else unit}")
        if not unit and header_unit:
            tokens.add(f"number:{number}:{'%' if header_unit.group() == '%' else 'pp'}")
    return tokens, _NUMBER.sub(" ", without_tables)


def _text_field(record: dict, key: str) -> str:
    value = record.get(key)
    return value if isinstance(value, str) else ""


def _document_key(record: dict) -> tuple[str, str]:
    version = _text_field(record, "doc_version_id") or _text_field(record, "document_version_id")
    return record["doc_id"], version


def _section_key(section: str) -> str:
    return _SECTION_NUMBER.sub("", _normalize(section).strip()).rstrip(":：")


def _table_key(record: dict) -> tuple[tuple[str, str], int, int] | None:
    locator = record["locator"]
    table_index = locator.get("table_index")
    if type(table_index) is not int or table_index < 0:
        return None
    return _document_key(record), locator["page_index"], table_index


def _cell_position(record: dict) -> tuple[int, int] | None:
    row = record["locator"].get("row_index")
    column = record["locator"].get("column_index")
    if type(row) is int and type(column) is int and row >= 0 and column >= 0:
        return row, column
    return None


def _table_label(record: dict) -> str | None:
    value = _text_field(record, "table_label") or _text_field(record["locator"], "table_label")
    return _printed_table_label(value)


def _printed_table_label(value: str) -> str | None:
    label = _normalize(value).strip()
    match = _TABLE_REFERENCE.fullmatch(label)
    if match:
        return match[1]
    return label if re.fullmatch(_TABLE_NUMBER, label) else None


def _bbox(record: dict) -> tuple[float, float, float, float] | None:
    box = record["locator"].get("bbox")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if not all(type(value) in (int, float) and math.isfinite(value) for value in box):
        return None
    x0, y0, x1, y1 = box
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def _near_caption(caption: tuple, boxes: list[tuple]) -> bool:
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    left, top, right, bottom = caption
    if min(right, x1) <= max(left, x0):
        return False
    return 0 <= y0 - bottom <= 36 or 0 <= top - y1 <= 36


@dataclass(frozen=True, slots=True)
class _Edge:
    target_id: str
    relation: str
    anchors: tuple[str, ...]


def _unit_vectors(embedder: Embedder, texts: list[str]) -> list[tuple[float, ...]]:
    vectors = []
    for values in embedder.encode(texts):
        vector = tuple(float(value) for value in values)
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("embedding vectors must be finite and nonempty")
        norm = math.hypot(*vector)
        vectors.append(tuple(value / norm for value in vector) if norm else vector)
    if len(vectors) != len(texts) or len({len(vector) for vector in vectors}) > 1:
        raise ValueError("embedder returned an unexpected number or dimension of vectors")
    return vectors


class EvidenceIndex:
    """Index only the records authorized for this run and retain their original anchors."""

    def __init__(self, evidence: Iterable[dict], *, embedder: Embedder | None = None) -> None:
        self._records: dict[str, dict] = {}
        self._exact: dict[str, set[str]] = defaultdict(set)
        self._graph: dict[str, set[_Edge]] = defaultdict(set)
        self._table_indexes: dict[tuple, list[str]] = defaultdict(list)
        self._named_tables: dict[str, set[str]] = defaultdict(set)
        self._embedder = embedder
        texts = []
        lexical_rows = []
        for raw in evidence:
            record = deepcopy(raw)
            eid = self._register(record)
            text_parts = [record["text"], _text_field(record, "row_header"), _text_field(record, "column_header")]
            text = " ".join(part for part in text_parts if part)
            texts.append(text)
            lexical_rows.append((eid, " ".join(sorted(_terms(text, index=True)))))
            exact, _ = _exact_terms(record["text"], header=text_parts[-1])
            panel = _text_field(record["locator"], "panel_label")
            if panel and (figure := _text_field(record, "figure_label")):
                exact.add(f"figure-panel:{_normalize(figure)}:{_normalize(panel)}")
            for token in exact:
                self._exact[token].add(eid)
            table = _table_key(record)
            if record.get("kind") == "table_cell" and table is not None:
                self._table_indexes[table].append(eid)

        vectors = _unit_vectors(embedder, texts) if embedder is not None and texts else []
        self._vectors = dict(zip(self._records, vectors))
        self._build_tables()
        self._build_experiments()
        self._build_ingest_links()
        self._fts = sqlite3.connect(":memory:")
        try:
            self._fts.execute("CREATE VIRTUAL TABLE lexical USING fts5(evidence_id UNINDEXED, text, tokenize='unicode61')")
            self._fts.executemany("INSERT INTO lexical(evidence_id, text) VALUES (?, ?)", lexical_rows)
        except sqlite3.Error:
            self._fts.close()
            raise

    def close(self) -> None:
        self._fts.close()

    def _register(self, record: dict) -> str:
        if not isinstance(record, dict):
            raise ValueError("an evidence record must be an object")
        for key in ("evidence_id", "doc_id", "text"):
            if not _text_field(record, key).strip():
                raise ValueError(f"evidence requires a nonempty {key}")
        locator = record.get("locator")
        if not isinstance(locator, dict) or type(locator.get("page_index")) is not int:
            raise ValueError("evidence requires a page locator")
        if locator["page_index"] < 0:
            raise ValueError("evidence page_index must be nonnegative")
        eid = record["evidence_id"]
        if eid in self._records:
            raise ValueError(f"duplicate evidence ID: {eid!r}")
        self._records[eid] = record
        return eid

    def _connect(self, first: str, second: str, relation: str, *anchors: str) -> None:
        if first == second or _document_key(self._records[first]) != _document_key(self._records[second]):
            return
        self._graph[first].add(_Edge(second, relation, anchors))
        self._graph[second].add(_Edge(first, relation, anchors))

    def _connect_cells(self, ids: list[str]) -> None:
        positions = {}
        for eid in ids:
            position = _cell_position(self._records[eid])
            if position is not None:
                positions[position] = eid
        for (row, column), eid in positions.items():
            cell = self._records[eid]
            for next_position in ((row + 1, column), (row, column + 1)):
                nearby = positions.get(next_position)
                if nearby is not None:
                    self._connect(eid, nearby, "same_table", eid, nearby)
            for key, target in (("row_header", (row, 0)), ("column_header", (0, column))):
                header_id = positions.get(target)
                if header_id is None:
                    continue
                header = _normalize(_text_field(cell, key)).strip()
                if header and header == _normalize(self._records[header_id]["text"]).strip():
                    self._connect(eid, header_id, key, header_id)

    def _build_tables(self) -> None:
        table_labels: dict[tuple, set[str]] = defaultdict(set)
        captions: dict[tuple, list[str]] = defaultdict(list)
        page_tables: dict[tuple, list[tuple]] = defaultdict(list)
        for table, ids in self._table_indexes.items():
            self._connect_cells(ids)
            page_tables[table[:2]].append(table)
            for eid in ids:
                label = _table_label(self._records[eid])
                if label is not None:
                    table_labels[table].add(label)

        for eid, record in self._records.items():
            if record.get("kind") == "table_cell":
                continue
            caption_match = _TABLE_CAPTION.match(_normalize(record["text"]))
            label = caption_match[1] if caption_match else None
            if label is None and record.get("kind") == "table_caption":
                label = _table_label(record)
            if label is None:
                continue
            document = _document_key(record)
            captions[(document, label)].append(eid)
            page_key = document, record["locator"]["page_index"]
            explicit_table = _table_key(record)
            possible_tables = [explicit_table] if explicit_table in self._table_indexes else []
            if not possible_tables:
                possible_tables = self._nearby_tables(record, page_tables[page_key], table_labels, label)
            if len(possible_tables) == 1:
                table = possible_tables[0]
                table_labels[table].add(label)
                self._named_tables[label].add(eid)
                for cell_id in self._table_indexes[table]:
                    self._connect(eid, cell_id, "table_caption", eid)

        labeled_tables: dict[tuple, list[tuple]] = defaultdict(list)
        for table, labels in table_labels.items():
            if len(labels) == 1:
                label = next(iter(labels))
                for eid in self._table_indexes[table]:
                    self._exact[f"table:{label}"].add(eid)
                    self._named_tables[label].add(eid)
                labeled_tables[(table[0], label)].append(table)

        for eid, record in self._records.items():
            if record.get("kind") == "table_cell":
                continue
            for label in set(_TABLE_REFERENCE.findall(_normalize(record["text"]))):
                key = _document_key(record), label
                targets = captions.get(key, [])
                if len(targets) == 1:
                    self._connect(eid, targets[0], "table_reference", eid)
                elif not targets and len(labeled_tables.get(key, [])) == 1:
                    table = labeled_tables[key][0]
                    first_cell = min(self._table_indexes[table])
                    self._connect(eid, first_cell, "table_reference", eid, first_cell)

    def _nearby_tables(self, record: dict, tables: list[tuple], labels: dict, label: str) -> list[tuple]:
        caption_box = _bbox(record)
        if caption_box is None:
            return []
        nearby = []
        for table in tables:
            if labels[table] and labels[table] != {label}:
                continue
            boxes = [box for eid in self._table_indexes[table] if (box := _bbox(self._records[eid]))]
            if boxes and _near_caption(caption_box, boxes):
                nearby.append(table)
        return nearby

    def _build_experiments(self) -> None:
        groups: dict[tuple, list[str]] = defaultdict(list)
        for eid, record in self._records.items():
            for reference in set(_EXPERIMENT.findall(_normalize(record["text"]))):
                groups[(_document_key(record), reference)].append(eid)
        for ids in groups.values():
            def method_first(eid: str) -> tuple[bool, str]:
                section = _normalize(_text_field(self._records[eid]["locator"], "section"))
                return not ("method" in section or "方法" in section), eid

            center = min(ids, key=method_first)
            for eid in ids:
                self._connect(center, eid, "same_experiment", center, eid)

    def _build_ingest_links(self) -> None:
        figures: dict[tuple, list[str]] = defaultdict(list)
        for eid, record in self._records.items():
            locator = record["locator"]
            kind = record.get("kind")
            caption_id = record.get("caption_evidence_id")
            caption = self._records.get(caption_id) if isinstance(caption_id, str) else None
            if caption is not None and (_document_key(record) != _document_key(caption)
                                        or caption["locator"]["page_index"] != locator["page_index"]):
                caption = None
            if caption is not None:
                relation = {"figure": "figure_caption", "table_cell": "table_caption",
                            "table_note": "table_note"}.get(kind)
                if relation:
                    self._connect(eid, caption_id, relation, caption_id)
            if kind == "figure" and (label := _text_field(record, "figure_label")):
                if not locator.get("panel_label"):
                    figures[(_document_key(record), _normalize(label))].append(eid)
                elif isinstance(parent_id := record.get("parent_figure_evidence_id"), str) and parent_id in self._records:
                    parent = self._records[parent_id]
                    if parent.get("kind") == "figure" and parent.get("figure_label") == label and caption is not None:
                        self._connect(eid, parent_id, "figure_panel", caption_id, eid)
            if kind == "table_note" and (label := _table_label(record)):
                self._named_tables[label].add(eid)
            previous = locator.get("continued_from")
            if not isinstance(previous, dict) or caption is None:
                continue
            previous_id = previous.get("caption_evidence_id")
            previous_caption = self._records.get(previous_id) if isinstance(previous_id, str) else None
            if previous_caption is None:
                continue
            previous_loc = previous_caption["locator"]
            group = locator.get("table_group_id")
            if (group and group == previous_loc.get("table_group_id")
                    and group == caption["locator"].get("table_group_id")
                    and previous_loc["page_index"] == previous.get("page_index")
                    and previous_loc.get("table_index") == previous.get("table_index")
                    and previous_loc["page_index"] < locator["page_index"]):
                self._connect(caption_id, previous_id, "table_continuation", caption_id, previous_id)
        for eid, record in self._records.items():
            if record.get("kind") == "figure":
                continue
            for label in set(_FIGURE_REFERENCE.findall(_normalize(record["text"]))):
                targets = figures.get((_document_key(record), label), [])
                if len(targets) == 1 and eid != self._records[targets[0]].get("caption_evidence_id"):
                    self._connect(eid, targets[0], "figure_reference", eid)

    def get(self, ids: Iterable[str]) -> list[dict]:
        """Return only records owned by this index, in the requested order."""
        if isinstance(ids, str):
            raise ValueError("evidence IDs must be an iterable of IDs, not a string")
        result = []
        for eid in ids:
            if not isinstance(eid, str) or eid not in self._records:
                raise ValueError(f"evidence is outside the current scope: {eid!r}")
            result.append(deepcopy(self._records[eid]))
        return result

    @staticmethod
    def _rank(postings: dict[str, set[str]], tokens: set[str], active: set[str]) -> list[str]:
        scores: dict[str, float] = defaultdict(float)
        for token in sorted(tokens):
            matched = postings.get(token, set()) & active
            weight = 1 + math.log((len(active) + 1) / (len(matched) + 1))
            for eid in matched:
                scores[eid] += weight
        return sorted(scores, key=lambda eid: (-scores[eid], eid))

    def _semantic_rank(self, query: str, active: set[str]) -> list[str]:
        if self._embedder is None or not active or not self._vectors:
            return []
        query_vector, = _unit_vectors(self._embedder, [query])
        scores = {}
        for eid in active:
            vector = self._vectors[eid]
            if len(vector) != len(query_vector):
                raise ValueError("query embedding dimension differs from the evidence index")
            score = math.fsum(a * b for a, b in zip(query_vector, vector))
            if score > 0:
                scores[eid] = score
        return sorted(scores, key=lambda eid: (-scores[eid], eid))

    def _lexical_rank(self, tokens: set[str], active: set[str]) -> list[str]:
        if not tokens or not active:
            return []
        # Quote the generated tokens so words such as OR are searched literally.
        query = " OR ".join(f'"{token}"' for token in sorted(tokens)[:96])
        rows = self._fts.execute(
            "SELECT evidence_id FROM lexical WHERE lexical MATCH ? ORDER BY bm25(lexical), evidence_id", (query,),
        )
        return [eid for (eid,) in rows if eid in active]

    def _walk(self, seed: str, active: set[str]) -> Iterable[tuple[tuple[str, ...], tuple[_Edge, ...]]]:
        queue = deque([((seed,), ())])
        while queue:
            nodes, path = queue.popleft()
            if path:
                yield nodes, path
            if len(path) == 2:
                continue
            for edge in self._graph.get(nodes[-1], ()):
                if edge.target_id in active and edge.target_id not in nodes:
                    queue.append(((*nodes, edge.target_id), (*path, edge)))

    def _expand(self, ranked: list[str], limit: int, active: set[str]) -> tuple[list[str], list[dict]]:
        if limit <= 1:
            return ranked[:limit], []
        seeds = ranked[: max(1, limit - 2)]
        selected = dict.fromkeys(seeds)
        paths = []
        walks = []
        for rank, seed in enumerate(seeds):
            for nodes, edges in self._walk(seed, active):
                walks.append((rank, nodes, edges))
        walks.sort(key=lambda walk: (
            _LINK_PRIORITY[walk[2][-1].relation], len(walk[2]), walk[0], walk[1],
            tuple((edge.relation, edge.anchors) for edge in walk[2]),
        ))
        for _, nodes, edges in walks:
            if len(selected) >= limit:
                break
            if nodes[-1] in selected:
                continue
            additions = [eid for eid in nodes[1:] if eid not in selected]
            if len(selected) + len(additions) > limit:
                continue
            selected.update(dict.fromkeys(additions))
            paths.append({"evidence_ids": list(nodes), "edges": self._serialize_path(nodes, edges)})

        for eid in ranked:
            if len(selected) >= limit:
                break
            selected[eid] = None
        return list(selected), paths

    def _serialize_path(self, nodes: tuple[str, ...], edges: tuple[_Edge, ...]) -> list[dict]:
        result = []
        for source, edge in zip(nodes, edges):
            result.append({
                "from_id": source,
                "to_id": edge.target_id,
                "relation": edge.relation,
                "doc_id": self._records[source]["doc_id"],
                "anchors": [
                    {"evidence_id": aid, "locator": deepcopy(self._records[aid]["locator"]),
                     "text": self._records[aid]["text"]}
                    for aid in edge.anchors
                ],
            })
        return result

    def search(
        self, query: str, *, limit: int = 8, section: str | None = None, expand: bool = True,
        doc_id: str | None = None,
    ) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("evidence query must be nonempty text")
        if type(limit) is not int or limit < 0:
            raise ValueError("evidence limit must be a nonnegative integer")
        if section is not None and (not isinstance(section, str) or not section.strip()):
            raise ValueError("section must be nonempty text when supplied")
        limit = min(limit, _MAX_CANDIDATES)
        section_key = _normalize(section).strip() if section else None
        active = {
            eid for eid, record in self._records.items()
            if (doc_id is None or record["doc_id"] == doc_id)
            and (section_key is None or section_key in _normalize(_text_field(record["locator"], "section")))
        }
        exact, query_without_exact = _exact_terms(query)
        lexical = _terms(query_without_exact if exact else query)
        ranks = {
            "lexical": self._lexical_rank(lexical, active),
            "exact": self._rank(self._exact, exact, active),
        }
        if self._embedder is not None:
            ranks["semantic"] = self._semantic_rank(query, active)
        scores: dict[str, float] = defaultdict(float)
        for channel in ranks.values():
            for rank, eid in enumerate(channel[:_MAX_CANDIDATES], start=1):
                scores[eid] += 1 / (60 + rank)
        ranked = sorted(scores, key=lambda eid: (-scores[eid], eid))[:_MAX_CANDIDATES]
        chosen, paths = self._expand(ranked, limit, active) if expand else (ranked[:limit], [])
        locations = {
            (self._records[eid]["doc_id"], self._records[eid]["locator"]["page_index"],
             _text_field(self._records[eid]["locator"], "section"))
            for eid in active
        }
        return {
            "evidence": self.get(chosen),
            "paths": paths,
            "scope": {
                "query": query,
                "section": section,
                "doc_id": doc_id,
                "document_ids": sorted({doc_id for doc_id, _, _ in locations}),
                "locations": [
                    {"doc_id": doc_id, "page_index": page, "section": section_name}
                    for doc_id, page, section_name in sorted(locations)
                ],
                "examined_evidence_ids": sorted(active),
                "channels": list(ranks),
                "lexical_engine": "sqlite_fts5_unicode61",
                "candidate_limit": limit,
            },
        }

    def inspect(
        self, *, section: str | None = None, table_label: str | None = None,
        page_index: int | None = None, doc_id: str | None = None, limit: int = 8,
    ) -> dict:
        """Read a named local source directly, independent of relevance ranking."""
        if section is None and table_label is None and page_index is None:
            raise ValueError("direct inspection needs a section, table label or page")
        if not 0 <= limit <= _MAX_CANDIDATES:
            raise ValueError("invalid direct inspection limit")
        label = _printed_table_label(table_label) if table_label else None
        table_ids = self._named_tables.get(label, set())
        section_key = _section_key(section) if section is not None else None
        ids = []
        for eid, record in self._records.items():
            locator = record["locator"]
            if doc_id is not None and record["doc_id"] != doc_id:
                continue
            if page_index is not None and locator["page_index"] != page_index:
                continue
            if section_key is not None and _section_key(_text_field(locator, "section")) != section_key:
                continue
            if table_label is not None and eid not in table_ids:
                continue
            ids.append(eid)
        ids.sort(key=lambda eid: (
            self._records[eid]["doc_id"], self._records[eid]["locator"]["page_index"], eid,
        ))
        chosen = self.get(ids[:limit])
        locators = {
            (record["doc_id"], record["locator"]["page_index"], _text_field(record["locator"], "section"))
            for record in chosen
        }
        return {"evidence": chosen, "scope": {
            "mode": "direct", "section": section, "table_label": table_label,
            "page_index": page_index, "doc_id": doc_id, "matching_count": len(ids),
            "complete": bool(ids) and len(ids) <= limit,
            "examined_evidence_ids": [item["evidence_id"] for item in chosen],
            "document_ids": sorted({doc for doc, _, _ in locators}),
            "locations": [
                {"doc_id": doc, "page_index": page, "section": name} for doc, page, name in sorted(locators)
            ],
        }}
