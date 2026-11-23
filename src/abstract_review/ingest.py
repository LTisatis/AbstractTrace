"""Local PDF text, table and figure extraction with original-page anchors."""

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path

import pymupdf


# PyMuPDF otherwise prints a layout recommendation before the CLI's JSON.
pymupdf.no_recommend_layout()

MAX_PDF_BYTES = 64 * 1024 * 1024
MAX_PDF_PAGES = 200
_PARSER_FAMILY = f"pymupdf-{pymupdf.__version__}"
PARSER_VERSION = f"{_PARSER_FAMILY}-v3"
# Abstract cleanup does not change the physical locator of an existing source.
_ANCHOR_VERSION = f"{_PARSER_FAMILY}-v2"

_PDF_ERRORS = (pymupdf.mupdf.FzErrorBase, RuntimeError, ValueError)
_ABSTRACT_HEADING = re.compile(
    r"^(?:abstract|摘\s*要)(?:\s*[:：.—–-]\s*(?P<text>.*))?$", re.IGNORECASE
)
_KEYWORDS_HEADING = re.compile(r"^(?:keywords?\b|index terms\b|关键词)", re.IGNORECASE)
_ABSTRACT_WORD = re.compile(r"[^\W\d_]+")
_WRAP_SUFFIX = re.compile(r"([^\W\d_]+)([-\u00ad])$")
_OBJECT_LABEL = r"[A-Z]?\d+(?:[.-]\d+)*[A-Z]?"
_LABEL_END = r"(?=$|\s|[:：.。—–(（])"
_TABLE_CAPTION = re.compile(rf"^\s*(?P<prefix>table\s+|续表\s*|表\s*)(?P<label>{_OBJECT_LABEL}){_LABEL_END}", re.I)
_TABLE_CONTINUED = re.compile(
    r"^[\s:：.—–-]*(?:continued|cont\.?|续)(?:\s*[:：.。].*|\s*)$|[(（]\s*(?:continued|cont\.?|续)\s*[)）]",
    re.I,
)
_FIGURE_CAPTION = re.compile(
    rf"^\s*(?:fig(?:ure)?\.?\s+|图\s*)(?P<label>{_OBJECT_LABEL}){_LABEL_END}", re.I,
)
_PANEL_MARKER = re.compile(r"[(（]\s*([a-z])\s*[)）]", re.I)
_OBJECT_MENTION = re.compile(
    r"^\s*(?:(?:shows?|summari[sz]es?|illustrates?|presents?|reports?|lists?|gives?|contains?)\b|所示|显示|列出|给出)",
    re.I,
)
_NUMERIC_START = re.compile(r"^[+−-]?(?:\d|\.\d)")
_DATA_NUMBER = re.compile(r"[+−-]?(?:\d+(?:\.\d+)?|\.\d+)\s*%?")
_EXPERIMENT_REF = re.compile(
    r"(?:(?<![a-z0-9])(?:experiment\s+|exp\.\s*)|实验(?:编号|号)?\s*)"
    r"[:：#]?\s*([a-z]*\d+(?:[-.]\d+)*[a-z]?)(?![a-z0-9])", re.I,
)
_SHORT_MARKER = r"(?:[a-z]|\d{1,2})"
_MARKER = rf"\({_SHORT_MARKER}\)|\[{_SHORT_MARKER}\]|\*{{1,3}}|†|‡|[⁰¹²³⁴⁵⁶⁷⁸⁹ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ]"
_NOTE_MARKER = re.compile(rf"^\s*(?P<marker>{_MARKER})\s*(?:[:：.]\s*)?(?P<body>\S.*)$", re.I)
_MARKED_VALUE = re.compile(rf"(?P<base>.*?)\s*(?P<marker>{_MARKER})\s*$", re.I)
_NOTE_PREFIX = re.compile(r"^\s*(?:notes?|注释?|说明)\s*[:：.]\s*", re.I)
_NAMED_NOTE = re.compile(
    rf"^\s*(?:notes?\s+(?:for|to)\s+(?:table\s+|表\s*)(?P<first>{_OBJECT_LABEL})"
    rf"|(?:table\s+|表\s*)(?P<second>{_OBJECT_LABEL})\s*(?:notes?\b|注释?))\s*[:：.]\s*", re.I,
)
_SECTION_HEADING = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*|[IVX]+)[.)]?\s+)?"
    r"(?:introduction|background|related work|"
    r"(?:materials and )?methods?|methodology|experimental setup|"
    r"experiments?|evaluation|results(?: and discussion)?|discussion|"
    r"limitations|conclusions?|concluding remarks|"
    r"acknowledg(?:e)?ments|references|bibliography|appendix(?: [A-Z0-9]+)?|"
    r"引言|绪论|前言|背景|相关工作|材料与方法|方法|实验设置|实验|结果与讨论|结果|"
    r"讨论|局限性|结论|致谢|参考文献|附录(?:\s*[A-Z0-9]+)?)[:：]?$",
    re.IGNORECASE,
)
_SOURCE_SECTION_NUMBER = r"[1-9]\d?(?:\.[1-9]\d?){0,3}[.)]?"
_SOURCE_NUMBER_ONLY = re.compile(_SOURCE_SECTION_NUMBER)
_SOURCE_NUMBERED_TITLE = re.compile(rf"({_SOURCE_SECTION_NUMBER})\s+(.+)")
_SOURCE_CONTENTS_TITLE = re.compile(r"(?:table of contents|contents|目录)[:：]?", re.I)
_SOURCE_BODY_OPENING = re.compile(r"(?:we|our|this|these|they|it|there)\b", re.I)
_SOURCE_LEADER = re.compile(r"(?:\.\s*){2,}|[…⋯]")

type BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ParsedPdf:
    data: bytes
    document: dict
    abstract: str | None
    evidence: list[dict]


@dataclass(frozen=True, slots=True)
class _Word:
    text: str
    bbox: BBox


@dataclass(frozen=True, slots=True)
class _Line:
    block_index: int
    line_index: int
    words: tuple[_Word, ...]

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)

    @property
    def bbox(self) -> BBox:
        return (
            min(word.bbox[0] for word in self.words),
            min(word.bbox[1] for word in self.words),
            max(word.bbox[2] for word in self.words),
            max(word.bbox[3] for word in self.words),
        )


@dataclass(frozen=True, slots=True)
class _Cell:
    table_index: int
    row_index: int
    column_index: int
    column_count: int
    bbox: BBox
    text: str
    row_header: str | None
    column_header: str | None
    column_header_path: tuple[str, ...] = ()
    column_header_cells: tuple[tuple[int, int], ...] = ()
    role: str = "value"


@dataclass(frozen=True, slots=True)
class _TableLabel:
    label: str
    page_index: int
    table_index: int
    caption_id: str
    headers: tuple[str, ...] | None
    group_id: str | None
    continued: bool
    previous: "_TableLabel | None" = None


@dataclass(frozen=True, slots=True)
class _Caption:
    label: str
    bbox: BBox
    evidence: dict
    continued: bool = False


@dataclass(frozen=True, slots=True)
class _HeadingFragment:
    text: str
    bbox: BBox
    baseline: float
    size: float
    font: str
    bold: bool


def _bounded_box(rect, width: float, height: float) -> BBox | None:
    x0, y0, x1, y1 = map(float, rect)
    if not all(map(isfinite, (x0, y0, x1, y1))):
        return None
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def _covers_column(outer: BBox, inner: BBox) -> bool:
    return outer[0] <= inner[0] + 0.5 and outer[2] >= inner[2] - 0.5


def _column_headers(
    values: list[list[str | None]], grid: dict[tuple[int, int], BBox], width: int,
) -> tuple[int, dict[int, tuple[tuple[int, int], ...]]]:
    ordinary = {col: ((0, col),) for col in range(1, width) if (0, col) in grid and values[0][col]}
    depth = next((row for row in range(1, min(5, len(values))) if any(
        _NUMERIC_START.match(text.strip()) for text in values[row][1:] if text
    )), 1)
    if depth < 2 or any((depth, col) not in grid for col in range(1, width)):
        return 1, ordinary

    columns: dict[int, list[tuple[int, int]]] = {col: [] for col in range(1, width)}
    merged = False
    for level in range(depth):
        anchors = [(level, col) for col in range(1, width) if (level, col) in grid and values[level][col]]
        assigned = []
        for col in columns:
            options = [anchor for anchor in anchors if _covers_column(grid[anchor], grid[depth, col])]
            if len(options) != 1:
                return 1, ordinary
            anchor = options[0]
            if level:
                upper = grid[columns[col][-1]]
                lower = grid[anchor]
                if not _covers_column(upper, lower) or upper[3] > lower[1] + 0.5:
                    return 1, ordinary
            columns[col].append(anchor)
            assigned.append(anchor)
        shared = len(set(assigned)) != len(assigned)
        if level == depth - 1 and shared:
            return 1, ordinary
        merged = merged or shared
    return (depth, {col: tuple(path) for col, path in columns.items()}) if merged else (1, ordinary)


def _table_cells(page: pymupdf.Page, width: float, height: float) -> tuple[list[_Cell], list[BBox]]:
    rotation = page.rotation
    try:
        # Table detection and PDF text extraction share the unrotated crop frame.
        if rotation:
            page.set_rotation(0)
        tables = page.find_tables(strategy="lines_strict", use_layout=False).tables
        cells = []
        table_boxes = []
        for table_index, table in enumerate(sorted(tables, key=lambda table: (table.bbox[1], table.bbox[0]))):
            if table.row_count < 2 or table.col_count < 2:
                continue
            values = table.extract()
            table_box = _bounded_box(table.bbox, width, height)
            if table_box is None:
                continue
            grid = {
                (row_index, column_index): box for row_index, row in enumerate(table.rows)
                for column_index, rect in enumerate(row.cells)
                if rect is not None and (box := _bounded_box(rect, width, height)) is not None
            }
            depth, columns = _column_headers(values, grid, table.col_count)
            previous_cell_count = len(cells)
            for (row_index, column_index), bbox in grid.items():
                value = values[row_index][column_index]
                if value is None or not value.strip():
                    continue
                role = "column_header" if row_index < depth else "row_header" if column_index == 0 else "value"
                row_header = values[row_index][0] if role == "value" else None
                anchors = columns.get(column_index, ()) if role == "value" else ()
                if anchors and not _covers_column(grid[anchors[-1]], bbox):
                    anchors = ()
                header_path = tuple(values[row][col].strip() for row, col in anchors)
                cells.append(_Cell(
                    table_index, row_index, column_index, table.col_count, bbox, value.strip(),
                    row_header.strip() if row_header else None, header_path[-1] if header_path else None,
                    header_path, anchors, role,
                ))
            if len(cells) > previous_cell_count:
                table_boxes.append(table_box)
        return cells, table_boxes
    finally:
        if rotation:
            page.set_rotation(rotation)


def _text_lines(raw_words, width: float, height: float, excluded: list[BBox]) -> list[_Line]:
    lines: dict[tuple[int, int], list[tuple[int, _Word]]] = {}
    for x0, y0, x1, y1, text, block_index, line_index, word_index in raw_words:
        bbox = _bounded_box((x0, y0, x1, y1), width, height)
        if bbox is None or not text.strip():
            continue
        x_center, y_center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        if any(left <= x_center <= right and top <= y_center <= bottom
               for left, top, right, bottom in excluded):
            continue
        lines.setdefault((block_index, line_index), []).append((word_index, _Word(text, bbox)))

    ordered = [
        _Line(block, line, tuple(word for _, word in sorted(words, key=lambda entry: entry[0])))
        for (block, line), words in lines.items()
    ]
    return _reading_order(ordered, width)


def _reading_order(lines: list[_Line], width: float) -> list[_Line]:
    vertical = sorted(lines, key=lambda line: (line.bbox[1], line.bbox[0], line.block_index, line.line_index))
    middle = width / 2
    left = [line for line in vertical if line.bbox[2] < middle]
    right = [line for line in vertical if line.bbox[0] > middle]
    if len({line.block_index for line in right}) < 2 or len({line.block_index for line in left}) < 2:
        return vertical
    if max(line.bbox[1] for line in left) < min(line.bbox[1] for line in right):
        return vertical

    ordered = []
    band = []
    for line in vertical:
        if line.bbox[0] <= middle <= line.bbox[2]:
            ordered.extend(sorted(band, key=lambda item: item.bbox[0] > middle))
            ordered.append(line)
            band.clear()
        else:
            band.append(line)
    ordered.extend(sorted(band, key=lambda item: item.bbox[0] > middle))
    return ordered


def _caption(pattern: re.Pattern, text: str) -> re.Match | None:
    match = pattern.match(text)
    if match is not None and not _OBJECT_MENTION.match(text[match.end():]):
        return match
    return None


def _ocr_words(page: pymupdf.Page, details: dict, language: str) -> list[tuple]:
    details["ocr_language"] = language
    dpi = min(250, int(72 * sqrt(25_000_000 / (details["width"] * details["height"]))))
    if dpi < 72:
        details["ocr_status"] = "page_too_large"
        return []
    try:
        textpage = page.get_textpage_ocr(language=language, dpi=dpi, full=True)
        words = [word for word in page.get_text("words", textpage=textpage) if _usable_word(word)]
    except _PDF_ERRORS:
        details["ocr_status"] = "unavailable_or_failed"
        return []
    details["ocr_status"] = "completed" if words else "no_text"
    return words


def _usable_word(word: tuple) -> bool:
    return bool(word[4].strip()) and "\ufffd" not in word[4]


def _image_boxes(page: pymupdf.Page, width: float, height: float) -> list[BBox]:
    boxes = set()
    for image in page.get_image_info():
        rect = image.get("bbox")
        if rect is None:
            continue
        box = _bounded_box(rect, width, height)
        if box == tuple(rect) and box[2] - box[0] >= 24 and box[3] - box[1] >= 24:
            boxes.add(box)
    return sorted(boxes, key=lambda box: (box[1], box[0], box[3], box[2]))


def _heading_fragments(page: pymupdf.Page, width: float, height: float, excluded: list[BBox]) -> list[_HeadingFragment]:
    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    fragments = []
    for block in page.get_text("dict", flags=flags)["blocks"]:
        for line in block.get("lines", ()):
            if line["dir"][0] < 0.99 or abs(line["dir"][1]) > 0.01:
                continue
            spans = [span for span in line["spans"] if span["text"].strip()]
            bbox = _bounded_box(line["bbox"], width, height)
            if not spans or bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            if any(x0 < right and left < x1 and y0 < bottom and top < y1 for left, top, right, bottom in excluded):
                continue
            text = " ".join(unicodedata.normalize("NFKC", " ".join(span["text"] for span in spans)).split())
            first = spans[0]
            bold = all(span["flags"] & pymupdf.TEXT_FONT_BOLD for span in spans)
            fragments.append(_HeadingFragment(text, bbox, first["origin"][1], first["size"], first["font"], bold))
    return sorted(fragments, key=lambda fragment: (fragment.baseline, fragment.bbox[0]))


def _source_title(text: str) -> bool:
    words = text.split()
    return bool(
        words and len(words) <= 12 and len(text) <= 100 and text[0].isalpha() and not words[-1].isdigit()
        and text[-1] not in ".!?。！？:：;；" and not _SOURCE_LEADER.search(text)
        and not _SOURCE_BODY_OPENING.match(text) and not _TABLE_CAPTION.match(text) and not _FIGURE_CAPTION.match(text)
    )


def _source_headings(page: pymupdf.Page, width: float, height: float, excluded: list[BBox]) -> list[str]:
    fragments = _heading_fragments(page, width, height, excluded)
    if any(_SOURCE_CONTENTS_TITLE.fullmatch(fragment.text) for fragment in fragments):
        return []

    headings = []
    consumed = set()
    for fragment in fragments:
        if fragment in consumed or not fragment.bold or "\ufffd" in fragment.text:
            continue
        same_line = [other for other in fragments if other.bbox[0] >= fragment.bbox[2] - 0.5
                     and abs(other.baseline - fragment.baseline) <= 0.15 * fragment.size]
        right = fragment
        if numbered := _SOURCE_NUMBERED_TITLE.fullmatch(fragment.text):
            number, title = numbered.groups()
        elif _SOURCE_NUMBER_ONLY.fullmatch(fragment.text):
            if not same_line:
                continue
            right = min(same_line, key=lambda other: other.bbox[0])
            if (not right.bold or fragment.font != right.font or abs(fragment.size - right.size) > 0.5
                    or right.bbox[0] - fragment.bbox[2] > 3.5 * fragment.size):
                continue
            number, title = fragment.text, right.text
        elif _SECTION_HEADING.fullmatch(fragment.text):
            number, title = "", fragment.text.rstrip(":：")
        else:
            continue
        if not _source_title(title) or "\ufffd" in title:
            continue
        if any(other.bbox[0] > right.bbox[2] + fragment.size and other.text.isdigit() for other in same_line):
            # A separately positioned page number is common in contents entries.
            continue
        if right is not fragment:
            consumed.add(right)
            continuations = [other for other in fragments if other.bold and other.font == right.font
                             and abs(other.size - right.size) <= 0.5
                             and abs(other.bbox[0] - right.bbox[0]) <= 0.25 * right.size
                             and 0.9 * right.size <= other.baseline - right.baseline <= 1.4 * right.size
                             and _source_title(f"{title} {other.text}") and other.text[0].isalpha()]
            if len(continuations) == 1 and len(title) >= 30:
                (continuation,) = continuations
                title = f"{title} {continuation.text}"
                consumed.add(continuation)
        headings.append(f"{number} {title}".strip())
    return list(dict.fromkeys(headings))


def _read_page(
    pdf: pymupdf.Document, page_index: int, ocr_language: str | None,
) -> tuple[dict, list[_Line], list[_Cell], list[BBox]]:
    details = {
        "page_index": page_index,
        "page_number": page_index + 1,
        "page_label": None,
        "width": None,
        "height": None,
        "rotation": None,
        "status": "unreadable",
    }
    try:
        page = pdf.load_page(page_index)
        width, height = page.cropbox.width, page.cropbox.height
        details.update(
            page_label=page.get_label() or None, width=width, height=height, rotation=page.rotation,
        )
        if not (isfinite(width) and isfinite(height) and width > 0 and height > 0):
            return details, [], [], []
        raw_words = page.get_text("words")
    except _PDF_ERRORS:
        return details, [], [], []

    native_words = [word for word in raw_words if _usable_word(word)]
    ocr_words = (
        _ocr_words(page, details, ocr_language)
        if ocr_language and (not native_words or 2 * len(native_words) < len(raw_words))
        else []
    )
    words = ocr_words or native_words
    if not words:
        if not raw_words and not ocr_language:
            details["status"] = "no_text"
        return details, [], [], []

    cells, table_boxes = [], []
    if not ocr_words:
        try:
            cells, table_boxes = _table_cells(page, width, height)
        except _PDF_ERRORS:
            details["table_status"] = "unreadable"
    lines = _text_lines(words, width, height, table_boxes)
    images = []
    if any(_caption(_FIGURE_CAPTION, line.text) for line in lines):
        try:
            images = _image_boxes(page, width, height)
        except _PDF_ERRORS:
            details["figure_status"] = "unreadable"
    details["status"] = ("ocr" if ocr_words else "parsed") if lines or cells else "unreadable"
    if details["status"] == "parsed":
        try:
            if headings := _source_headings(page, width, height, [*table_boxes, *images]):
                details["source_headings"] = headings
        except _PDF_ERRORS:
            # Page clues are optional when the native font metadata cannot be read.
            pass
    return details, lines, cells, images


def _evidence(doc_id: str, kind: str, identity: str, text: str, page: dict, bbox: BBox, section: str | None) -> dict:
    key = f"{_ANCHOR_VERSION}/{doc_id}/{kind}/{page['page_index']}/{identity}"
    if page["status"] == "ocr":
        key = f"{key}/ocr/{page['ocr_language']}"
    return {
        "evidence_id": "ev-" + hashlib.sha256(key.encode("utf-8")).hexdigest(),
        "doc_id": doc_id,
        "kind": kind,
        "text": text,
        "locator": {
            "page_index": page["page_index"],
            "page_number": page["page_number"],
            "page_label": page["page_label"],
            "bbox": list(bbox),
            "section": section or "Body",
        },
    }


def _near_caption(caption: BBox, item: BBox, max_gap: float) -> bool:
    overlap = min(caption[2], item[2]) - max(caption[0], item[0])
    gap = max(caption[1] - item[3], item[1] - caption[3])
    return overlap > 0 and 0 <= gap <= max_gap


def _caption_matches(captions: list[_Caption], boxes: dict[int, BBox], max_gap: float) -> dict[int, _Caption]:
    matches = defaultdict(list)
    counts = Counter()
    for caption_index, caption in enumerate(captions):
        for number, box in boxes.items():
            if _near_caption(caption.bbox, box, max_gap):
                matches[number].append(caption_index)
                counts[caption_index] += 1
    return {
        number: captions[indices[0]]
        for number, indices in matches.items()
        if len(indices) == 1 and counts[indices[0]] == 1
    }


def _table_boxes(tables: dict[int, list[_Cell]]) -> dict[int, BBox]:
    return {
        number: (
            min(cell.bbox[0] for cell in cells), min(cell.bbox[1] for cell in cells),
            max(cell.bbox[2] for cell in cells), max(cell.bbox[3] for cell in cells),
        )
        for number, cells in tables.items()
    }


def _header_signature(cells: list[_Cell]) -> tuple[str, ...] | None:
    layered = {cell.column_index: " / ".join(cell.column_header_path)
               for cell in cells if len(cell.column_header_path) > 1}
    if layered:
        stub = next((cell.text for cell in cells if cell.row_index == 0 and cell.column_index == 0), None)
        if stub is None or len(layered) != cells[0].column_count - 1:
            return None
        paths = [stub, *(layered[col] for col in range(1, cells[0].column_count))]
        return tuple(" ".join(unicodedata.normalize("NFKC", item).casefold().split()) for item in paths)
    headers = {
        cell.column_index: " ".join(unicodedata.normalize("NFKC", cell.text).casefold().split())
        for cell in cells if cell.row_index == 0
    }
    if len(headers) != cells[0].column_count or not all(headers.values()):
        return None
    return tuple(headers[index] for index in range(cells[0].column_count))


def _table_labels(
    doc_id: str, page_index: int, cells: list[_Cell],
    captions: list[_Caption], previous: dict[str, _TableLabel],
) -> dict[int, _TableLabel]:
    tables = defaultdict(list)
    for cell in cells:
        tables[cell.table_index].append(cell)

    labels = {}
    for number, caption in _caption_matches(captions, _table_boxes(tables), 45).items():
        headers = _header_signature(tables[number])
        prior = previous.get(caption.label.casefold()) if caption.continued else None
        if prior is not None and (headers is None or headers != prior.headers):
            prior = None
        group_id = f"table-{doc_id}-{page_index}-{number}" if not caption.continued else None
        if prior is not None:
            group_id = prior.group_id
        labels[number] = _TableLabel(
            caption.label, page_index, number, caption.evidence["evidence_id"],
            headers, group_id, caption.continued, prior,
        )
        _locate_table(caption.evidence, labels[number])
    return labels


def _locate_table(entry: dict, label: _TableLabel) -> None:
    locator = entry["locator"]
    locator["table_index"] = label.table_index
    if label.group_id:
        locator.update(table_label=label.label, table_group_id=label.group_id)
        entry["table_label"] = label.label
    elif label.continued:
        locator.update(declared_table_label=label.label, table_continuation="unverified")
    if label.previous is not None:
        previous = label.previous
        locator["continued_from"] = {
            "page_index": previous.page_index, "table_index": previous.table_index,
            "caption_evidence_id": previous.caption_id,
        }
    if entry["evidence_id"] != label.caption_id:
        entry["caption_evidence_id"] = label.caption_id


def _union_box(boxes: list[BBox]) -> BBox:
    return (min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes))


def _experiment_refs(entry: dict, line: _Line | None = None) -> None:
    if not (matches := list(_EXPERIMENT_REF.finditer(entry["text"]))):
        return
    spans = []
    if line is not None:
        offset = 0
        for word in line.words:
            end = offset + len(word.text)
            spans.append((offset, end, word.bbox))
            offset = end + 1
    refs = []
    for match in matches:
        boxes = [box for start, end, box in spans if start < match.end() and end > match.start()]
        refs.append({"experiment_id": match[1].casefold(), "text": match[0],
                     "bbox": list(_union_box(boxes)) if boxes else entry["locator"]["bbox"]})
    entry["experiment_refs"] = refs


def _marker_key(marker: str) -> str:
    return unicodedata.normalize("NFKC", marker).strip("()[]").casefold()


def _parse_note(text: str) -> tuple[str | None, str | None] | None:
    named = _NAMED_NOTE.match(text)
    prefix = named or _NOTE_PREFIX.match(text)
    body = text[prefix.end():] if prefix else text
    label = (named["first"] or named["second"]).casefold() if named else None
    if marked := _NOTE_MARKER.match(body):
        return label, _marker_key(marked["marker"])
    return (label, None) if prefix and body.strip() else None


def _suffix(text: str) -> tuple[str, str] | None:
    match = _MARKED_VALUE.fullmatch(text)
    if not match or not (base := match["base"].strip()):
        return None
    return base, _marker_key(match["marker"])


def _near_table_footer(note: dict, box: BBox) -> bool:
    left, top, right, _ = note["locator"]["bbox"]
    horizontal = min(right, box[2]) - max(left, box[0])
    return horizontal >= 6 and 0 <= top - box[3] <= 32


def _attach_notes(notes: list[tuple[dict, str | None, str | None]], cells: list[_Cell],
                  entries: dict[tuple[int, int, int], dict], labels: dict[int, _TableLabel]) -> None:
    by_table: dict[int, list[_Cell]] = defaultdict(list)
    for cell in cells:
        by_table[cell.table_index].append(cell)
    boxes = _table_boxes(by_table)
    bindings = []
    for note, declared, marker in notes:
        if declared is not None:
            choices = [table for table, label in labels.items() if label.group_id and label.label.casefold() == declared]
        else:
            choices = [table for table, box in boxes.items() if _near_table_footer(note, box)]
        if len(choices) == 1:
            bindings.append((choices[0], note, marker))

    duplicates = Counter((table, marker) for table, _, marker in bindings if marker is not None)
    for table, note, marker in bindings:
        table_cells = by_table[table]
        marks = {(cell.row_index, cell.column_index): parsed for cell in table_cells
                 if (parsed := _suffix(cell.text)) is not None and parsed[1] == marker}
        if marker is not None and (not marks or duplicates[(table, marker)] > 1):
            continue
        note["kind"] = "table_note"
        note["note_marker"] = marker
        note["note_scope"] = "marked_cells" if marker is not None else "table"
        note["locator"]["table_index"] = table
        label = labels.get(table)
        if label is not None:
            _locate_table(note, label)
        first = table_cells[0]
        table_anchor = label.caption_id if label else entries[(table, first.row_index, first.column_index)]["evidence_id"]
        for cell in table_cells:
            if cell.role != "value":
                continue
            row, col = cell.row_index, cell.column_index
            entry = entries[(table, row, col)]
            anchor_id = table_anchor
            if marker is not None:
                anchored = next((position for position in ((row, col), (row, 0), *cell.column_header_cells)
                                 if position in marks), None)
                if anchored is None:
                    continue
                anchor_id = entries[(table, *anchored)]["evidence_id"]
                if anchored == (row, col) and _DATA_NUMBER.fullmatch(marks[anchored][0]):
                    entry["numeric_text"] = marks[anchored][0]
            entry.setdefault("note_evidence_ids", []).append(note["evidence_id"])
            entry.setdefault("note_links", []).append({"note_evidence_id": note["evidence_id"],
                                                        "anchor_evidence_id": anchor_id, "marker": marker})


def _panel_near_corner(marker: BBox, image: BBox) -> bool:
    x0, y0, x1, y1 = image
    x, y = (marker[0] + marker[2]) / 2, (marker[1] + marker[3]) / 2
    return x0 - 10 <= x <= x0 + min(28, (x1 - x0) / 3) and y0 - 16 <= y <= y0 + min(30, (y1 - y0) / 3)


def _explicit_panels(caption: _Caption, images: list[BBox], lines: list[_Line]) -> dict[str, tuple[int, BBox]]:
    labels = list(dict.fromkeys(_PANEL_MARKER.findall(caption.evidence["text"].casefold())))
    if not 2 <= len(labels) <= 8:
        return {}
    panel_markers = defaultdict(list)
    for line in lines:
        if marker := _PANEL_MARKER.fullmatch(line.text):
            panel_markers[marker[1].casefold()].append(line.bbox)
    panels = {}
    for label in labels:
        matches = [(index, mark) for mark in panel_markers[label] for index, image in enumerate(images)
                   if _panel_near_corner(mark, image)]
        if len(matches) != 1:
            return {}
        panels[label] = matches[0]
    if len({number for number, _ in panels.values()}) != len(panels):
        return {}
    boxes = [images[number] for number, _ in panels.values()]
    union = (min(box[0] for box in boxes), min(box[1] for box in boxes),
             max(box[2] for box in boxes), max(box[3] for box in boxes))
    return panels if _near_caption(caption.bbox, union, 45) else {}


def _figures(
    doc_id: str, page: dict, images: list[BBox], captions: list[_Caption], lines: list[_Line],
) -> list[dict]:
    def figure(identity: str, caption: _Caption, box: BBox) -> dict:
        entry = _evidence(
            doc_id, "figure", identity, caption.evidence["text"], page, box,
            caption.evidence["locator"]["section"],
        )
        entry.update(figure_label=caption.label, caption_evidence_id=caption.evidence["evidence_id"],
                     source_type="embedded_image")
        entry["locator"]["figure_label"] = caption.label
        if page["status"] == "ocr":
            entry["text_source"] = "ocr"
        return entry

    simple = _caption_matches(captions, dict(enumerate(images)), 45)
    figures = []
    for number, caption in simple.items():
        entry = figure(str(number), caption, images[number])
        entry["locator"]["figure_index"] = number
        figures.append(entry)
    if page["status"] == "ocr":
        return figures

    simple_captions = {caption.evidence["evidence_id"] for caption in simple.values()}
    possibilities = [
        (caption, panels) for caption in captions if caption.evidence["evidence_id"] not in simple_captions
        if (panels := _explicit_panels(caption, images, lines))
    ]
    uses = Counter(number for _, panels in possibilities for number, _ in panels.values())
    for caption, panels in possibilities:
        if any(number in simple or uses[number] != 1 for number, _ in panels.values()):
            continue
        boxes = [images[number] for number, _ in panels.values()]
        union = (min(box[0] for box in boxes), min(box[1] for box in boxes),
                 max(box[2] for box in boxes), max(box[3] for box in boxes))
        parent = figure(f"panels-{caption.evidence['evidence_id']}", caption, union)
        parent["source_type"] = "embedded_panels"
        figures.append(parent)
        for label, (number, mark) in panels.items():
            panel = figure(f"panel-{number}-{label}", caption, images[number])
            panel["parent_figure_evidence_id"] = parent["evidence_id"]
            panel["locator"].update(figure_index=number, panel_label=label, panel_label_bbox=list(mark))
            figures.append(panel)
    return figures


def _abstract_column(heading: _Line, lines: list[_Line], width: float) -> tuple[float, float] | None:
    """Recognize a broad text column aligned with the abstract heading."""
    heading_left, _, heading_right, heading_bottom = heading.bbox
    blocks = defaultdict(list)
    for line in lines:
        if line.bbox[1] >= heading_bottom - 1:
            blocks[line.block_index].append(line)
    margin = 5.0
    for block in sorted(blocks.values(), key=lambda items: min(item.bbox[1] for item in items)):
        first_y = min(item.bbox[1] for item in block)
        if first_y > heading_bottom + 72:
            break
        left, right = min(item.bbox[0] for item in block), max(item.bbox[2] for item in block)
        # A narrow abstract column may continue in the next column. Filter only
        # when the text establishes a broad, heading-aligned main column.
        if right - left >= width / 2 and left <= heading_right + margin and right >= heading_left - margin:
            return left - margin, right + margin
    return None


def _abstract_text(parts: list[tuple[str, bool]]) -> str | None:
    words = {word.casefold() for text, _ in parts for word in _ABSTRACT_WORD.findall(text)}
    joined: list[str] = []
    for text, same_paragraph in parts:
        suffix = _WRAP_SUFFIX.search(joined[-1]) if same_paragraph and joined else None
        continuation = _ABSTRACT_WORD.match(text)
        if suffix and continuation and text[0].islower():
            word = (suffix[1] + continuation[0]).casefold()
            if suffix[2] == "\u00ad" or word in words:
                joined[-1] = joined[-1][:-1]
            joined[-1] += text
        else:
            joined.append(text)
    return " ".join(joined) or None


def _extract(pdf: pymupdf.Document, doc_id: str, ocr_language: str | None) -> tuple[list[dict], str | None, list[dict]]:
    pages = []
    abstract_parts: list[tuple[str, bool]] = []
    evidence = []
    section = None
    collecting_abstract = False
    previous_abstract_line = None
    previous_tables: dict[str, _TableLabel] = {}

    for page_index in range(pdf.page_count):
        page, lines, cells, images = _read_page(pdf, page_index, ocr_language)
        pages.append(page)
        page_start_section = section
        headings = []
        table_captions = []
        figure_captions = []
        table_notes = []
        abstract_column = None
        for line in lines:
            text = line.text
            if page_index < 2 and section is None and not collecting_abstract and not abstract_parts:
                match = _ABSTRACT_HEADING.fullmatch(text)
                if match:
                    collecting_abstract = True
                    abstract_column = _abstract_column(line, lines, page["width"])
                    if inline_text := match.group("text"):
                        abstract_parts.append((inline_text, False))
                        previous_abstract_line = (page_index, line.block_index, line.line_index)
                    continue
            if collecting_abstract and abstract_column is not None:
                left, right = abstract_column
                if line.bbox[2] <= left or line.bbox[0] >= right:
                    continue
            if _KEYWORDS_HEADING.match(text):
                collecting_abstract = False
                continue
            if _SECTION_HEADING.fullmatch(text):
                collecting_abstract = False
                section = text
                headings.append((line.bbox[1], section))
                continue
            if collecting_abstract:
                same_paragraph = previous_abstract_line == (page_index, line.block_index, line.line_index - 1)
                abstract_parts.append((text, same_paragraph))
                previous_abstract_line = (page_index, line.block_index, line.line_index)
                continue
            entry = _evidence(
                doc_id, "body_text", f"{line.block_index}/{line.line_index}", text, page, line.bbox, section,
            )
            entry["word_spans"] = [{"text": word.text, "bbox": list(word.bbox)} for word in line.words]
            if page["status"] == "ocr":
                entry["text_source"] = "ocr"
            _experiment_refs(entry, line)
            evidence.append(entry)
            if note := _parse_note(text):
                table_notes.append((entry, *note))
            elif match := _caption(_TABLE_CAPTION, text):
                continued = match.group("prefix").startswith("续表") or bool(_TABLE_CONTINUED.search(text[match.end():]))
                table_captions.append(_Caption(match.group("label"), line.bbox, entry, continued))
            if match := _caption(_FIGURE_CAPTION, text):
                figure_captions.append(_Caption(match.group("label"), line.bbox, entry))

        table_labels = _table_labels(doc_id, page_index, cells, table_captions, previous_tables)
        counts = Counter(label.label.casefold() for label in table_labels.values())
        previous_tables = {
            label.label.casefold(): label for label in table_labels.values()
            if label.group_id and counts[label.label.casefold()] == 1
        }
        cell_entries = {}
        for cell in cells:
            cell_section = next(
                (title for top, title in reversed(headings) if top <= cell.bbox[1]), page_start_section,
            )
            entry = _evidence(
                doc_id, "table_cell", f"{cell.table_index}/{cell.row_index}/{cell.column_index}",
                cell.text, page, cell.bbox, cell_section,
            )
            entry["locator"].update(
                table_index=cell.table_index, row_index=cell.row_index, column_index=cell.column_index,
            )
            entry["cell_role"] = cell.role
            if cell.row_header:
                entry["row_header"] = cell.row_header
            if cell.column_header:
                entry["column_header"] = cell.column_header
            if cell.column_header_path:
                entry["column_header_path"] = list(cell.column_header_path)
            if cell.table_index in table_labels:
                _locate_table(entry, table_labels[cell.table_index])
            _experiment_refs(entry)
            cell_entries[(cell.table_index, cell.row_index, cell.column_index)] = entry
            evidence.append(entry)
        for cell in cells:
            entry = cell_entries[(cell.table_index, cell.row_index, cell.column_index)]
            if cell.row_header and (header := cell_entries.get((cell.table_index, cell.row_index, 0))):
                entry["row_header_evidence_id"] = header["evidence_id"]
            if cell.column_header_cells:
                entry["column_header_evidence_ids"] = [cell_entries[(cell.table_index, *position)]["evidence_id"]
                                                       for position in cell.column_header_cells]
        _attach_notes(table_notes, cells, cell_entries, table_labels)
        evidence.extend(_figures(doc_id, page, images, figure_captions, lines))

    return pages, _abstract_text(abstract_parts), evidence


def parse_pdf(path: Path, role: str = "main", *, ocr_language: str | None = None) -> ParsedPdf:
    """Parse local page text, supported figures and clearly marked bordered tables."""
    if role not in {"main", "supplement", "external"}:
        raise ValueError(f"unknown PDF role {role!r}")
    with path.open("rb") as source:
        data = source.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise ValueError(f"{path.name}: PDF exceeds {MAX_PDF_BYTES // (1024 * 1024)} MiB")
    if b"%PDF-" not in data[:1024]:
        raise ValueError(f"{path.name}: invalid PDF header")
    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except _PDF_ERRORS as exc:
        raise ValueError(f"{path.name}: invalid PDF ({exc})") from exc

    with pdf:
        if not pdf.is_pdf:
            raise ValueError(f"{path.name}: input is not a PDF")
        if pdf.needs_pass:
            raise ValueError(f"{path.name}: password-protected PDF")
        if pdf.page_count < 1:
            raise ValueError(f"{path.name}: invalid PDF with no pages")
        if pdf.page_count > MAX_PDF_PAGES:
            raise ValueError(f"{path.name}: PDF exceeds {MAX_PDF_PAGES} pages")

        sha256 = hashlib.sha256(data).hexdigest()
        doc_id = f"{role}-{sha256}"
        pages, abstract, evidence = _extract(pdf, doc_id, ocr_language)
        document = {
            "doc_id": doc_id,
            "sha256": sha256,
            "name": path.name,
            "role": role,
            "parser_version": PARSER_VERSION,
            "pages": pages,
        }
        if pdf.is_repaired:
            document["repaired"] = True
    return ParsedPdf(data, document, abstract, evidence)
