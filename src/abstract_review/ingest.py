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
PARSER_VERSION = f"pymupdf-{pymupdf.__version__}-v2"

_PDF_ERRORS = (pymupdf.mupdf.FzErrorBase, RuntimeError, ValueError)
_ABSTRACT_HEADING = re.compile(
    r"^(?:abstract|摘\s*要)(?:\s*[:：.—–-]\s*(?P<text>.*))?$", re.IGNORECASE
)
_KEYWORDS_HEADING = re.compile(r"^(?:keywords?\b|index terms\b|关键词)", re.IGNORECASE)
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


def _bounded_box(rect, width: float, height: float) -> BBox | None:
    x0, y0, x1, y1 = map(float, rect)
    if not all(map(isfinite, (x0, y0, x1, y1))):
        return None
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


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
            previous_cell_count = len(cells)
            for row_index, row in enumerate(table.rows):
                for column_index, rect in enumerate(row.cells):
                    value = values[row_index][column_index]
                    if rect is None or value is None or not value.strip():
                        continue
                    bbox = _bounded_box(rect, width, height)
                    if bbox is None:
                        continue
                    row_header = values[row_index][0] if row_index and column_index else None
                    column_header = values[0][column_index] if row_index else None
                    cells.append(_Cell(
                        table_index, row_index, column_index, table.col_count, bbox, value.strip(),
                        row_header.strip() if row_header else None,
                        column_header.strip() if column_header else None,
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
    return details, lines, cells, images


def _evidence(doc_id: str, kind: str, identity: str, text: str, page: dict, bbox: BBox, section: str | None) -> dict:
    key = f"{PARSER_VERSION}/{doc_id}/{kind}/{page['page_index']}/{identity}"
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


def _extract(pdf: pymupdf.Document, doc_id: str, ocr_language: str | None) -> tuple[list[dict], str | None, list[dict]]:
    pages = []
    abstract_parts: list[str] = []
    evidence = []
    section = None
    collecting_abstract = False
    previous_tables: dict[str, _TableLabel] = {}

    for page_index in range(pdf.page_count):
        page, lines, cells, images = _read_page(pdf, page_index, ocr_language)
        pages.append(page)
        page_start_section = section
        headings = []
        table_captions = []
        figure_captions = []
        for line in lines:
            text = line.text
            if page_index < 2 and section is None and not collecting_abstract and not abstract_parts:
                match = _ABSTRACT_HEADING.fullmatch(text)
                if match:
                    collecting_abstract = True
                    if inline_text := match.group("text"):
                        abstract_parts.append(inline_text)
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
                abstract_parts.append(text)
                continue
            entry = _evidence(
                doc_id, "body_text", f"{line.block_index}/{line.line_index}", text, page, line.bbox, section,
            )
            entry["word_spans"] = [{"text": word.text, "bbox": list(word.bbox)} for word in line.words]
            if page["status"] == "ocr":
                entry["text_source"] = "ocr"
            evidence.append(entry)
            if match := _caption(_TABLE_CAPTION, text):
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
            if cell.row_header:
                entry["row_header"] = cell.row_header
            if cell.column_header:
                entry["column_header"] = cell.column_header
            if cell.table_index in table_labels:
                _locate_table(entry, table_labels[cell.table_index])
            evidence.append(entry)
        evidence.extend(_figures(doc_id, page, images, figure_captions, lines))

    return pages, " ".join(abstract_parts) or None, evidence


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
