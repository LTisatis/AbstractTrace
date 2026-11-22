"""Local PDF extraction and source anchors for the first parsing milestone."""

import hashlib
import re
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import pymupdf


# PyMuPDF otherwise prints a layout recommendation before the CLI's JSON.
pymupdf.no_recommend_layout()

MAX_PDF_BYTES = 64 * 1024 * 1024
MAX_PDF_PAGES = 200
PARSER_VERSION = f"pymupdf-{pymupdf.__version__}-d02-v1"

_PDF_ERRORS = (pymupdf.mupdf.FzErrorBase, RuntimeError, ValueError)
_ABSTRACT_HEADING = re.compile(
    r"^(?:abstract|摘\s*要)(?:\s*[:：.—–-]\s*(?P<text>.*))?$", re.IGNORECASE
)
_KEYWORDS_HEADING = re.compile(r"^(?:keywords?\b|index terms\b|关键词)", re.IGNORECASE)
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
    bbox: BBox
    text: str
    row_header: str | None
    column_header: str | None


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
                        table_index, row_index, column_index, bbox, value.strip(),
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
    return sorted(ordered, key=lambda line: (line.bbox[1], line.bbox[0], line.block_index, line.line_index))


def _read_page(pdf: pymupdf.Document, page_index: int) -> tuple[dict, list[_Line], list[_Cell]]:
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
            return details, [], []
        raw_words = page.get_text("words")
    except _PDF_ERRORS:
        return details, [], []

    if not raw_words:
        details["status"] = "no_text"
        return details, [], []

    try:
        cells, table_boxes = _table_cells(page, width, height)
    except _PDF_ERRORS:
        details["table_status"] = "unreadable"
        cells, table_boxes = [], []
    lines = _text_lines(raw_words, width, height, table_boxes)
    details["status"] = "parsed" if lines or cells else "no_text"
    return details, lines, cells


def _evidence(doc_id: str, kind: str, identity: str, text: str, page: dict, bbox: BBox, section: str | None) -> dict:
    key = f"{PARSER_VERSION}/{doc_id}/{kind}/{page['page_index']}/{identity}"
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


def _extract(pdf: pymupdf.Document, doc_id: str) -> tuple[list[dict], str | None, list[dict]]:
    pages = []
    abstract_parts: list[str] = []
    evidence = []
    section = None
    collecting_abstract = False

    for page_index in range(pdf.page_count):
        page, lines, cells = _read_page(pdf, page_index)
        pages.append(page)
        page_start_section = section
        headings = []
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
            evidence.append(entry)

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
            evidence.append(entry)

    return pages, " ".join(abstract_parts) or None, evidence


def parse_pdf(path: Path, role: str = "main") -> ParsedPdf:
    """Parse native text and bordered tables in a bounded, locally supplied PDF."""
    if role not in {"main", "supplement"}:
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
        pages, abstract, evidence = _extract(pdf, doc_id)
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
