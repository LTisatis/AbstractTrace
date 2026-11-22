from pathlib import Path

import pymupdf
import pytest

from abstract_review.ingest import parse_pdf


def save(pdf, path: Path) -> Path:
    path.write_bytes(pdf.tobytes())
    return path


def table(page, caption, headings, value):
    page.insert_text((42, 100), caption)
    for x in (40, 140, 240, 340):
        page.draw_line((x, 118), (x, 196))
    for y in (118, 144, 170, 196):
        page.draw_line((40, y), (340, y))
    for y, row in zip((137, 163, 189), (headings, ("A", "80%", value), ("B", "75%", "79%"))):
        for x, text in zip((48, 148, 248), row):
            page.insert_text((x, y), text)


def test_continued_table_needs_an_explicit_marker_and_the_same_headers(tmp_path):
    ordinary_headers = ("Dataset", "Baseline", "Final")
    with pymupdf.open() as paper:
        table(paper.new_page(), "Table 2: Mobility by dataset", ordinary_headers, "88%")
        table(paper.new_page(), "Table 2 (continued)", ordinary_headers, "89%")
        table(paper.new_page(), "Table 2 (continued)", ("Dataset", "Final", "Baseline"), "90%")
        table(paper.new_page(), "Table 2 (continued)", ordinary_headers, "91%")
        table(paper.new_page(), "Table 2: Follow-up", ordinary_headers, "92%")
        path = save(paper, tmp_path / "continued.pdf")

    parsed = parse_pdf(path)
    results = [entry for entry in parsed.evidence if entry["kind"] == "table_cell"
               and entry["locator"]["row_index"] == 1 and entry["locator"]["column_index"] == 2]
    first, matching, different_header, no_confirmed_predecessor, separate = results
    assert first["table_label"] == matching["table_label"] == "2"
    assert first["locator"]["table_group_id"] == matching["locator"]["table_group_id"]
    assert matching["locator"]["continued_from"] == {
        "page_index": 0, "table_index": first["locator"]["table_index"],
        "caption_evidence_id": first["caption_evidence_id"],
    }
    assert separate["locator"]["table_group_id"] != first["locator"]["table_group_id"]
    for unlinked in (different_header, no_confirmed_predecessor):
        assert unlinked["locator"]["declared_table_label"] == "2"
        assert unlinked["locator"]["table_continuation"] == "unverified"
        assert "table_group_id" not in unlinked["locator"]
    with pymupdf.open(path) as original:
        for page_index, item in enumerate(results):
            locator = item["locator"]
            assert (locator["page_index"], locator["page_number"]) == (page_index, page_index + 1)
            assert original[page_index].get_textbox(pymupdf.Rect(locator["bbox"])) == item["text"]
    assert parse_pdf(path).evidence == parsed.evidence


def test_figure_requires_a_unique_visible_image_and_caption(tmp_path):
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
    pixmap.clear_with(140)
    first_box = pymupdf.Rect(80, 35, 210, 140)
    with pymupdf.open() as paper:
        page = paper.new_page(width=520, height=700)
        page.set_cropbox(pymupdf.Rect(20, 30, 480, 670))
        page.insert_image(first_box, pixmap=pixmap, keep_proportion=False)
        page.insert_text((85, 158), "Fig. 1. A visible sample.")
        page.set_rotation(90)

        page = paper.new_page()
        page.insert_text((40, 100), "Figure 2: Image unavailable.")
        page = paper.new_page()
        page.draw_rect(first_box)
        page.insert_text((85, 158), "Figure 3: Vector drawing.")
        page = paper.new_page()
        page.insert_image(first_box, pixmap=pixmap, keep_proportion=False)
        page.insert_image(pymupdf.Rect(100, 165, 220, 270), pixmap=pixmap, keep_proportion=False)
        page.insert_text((110, 156), "Figure 4: Two nearby images.")
        page = paper.new_page()
        page.insert_image(first_box, pixmap=pixmap, keep_proportion=False)
        page.insert_text((85, 158), "Figure 5 shows the sample.")
        path = save(paper, tmp_path / "figures.pdf")

    parsed = parse_pdf(path)
    figure, = [entry for entry in parsed.evidence if entry["kind"] == "figure"]
    assert figure["figure_label"] == "1"
    assert figure["source_type"] == "embedded_image"
    assert figure["locator"]["page_index"] == 0
    assert figure["locator"]["bbox"] == pytest.approx(tuple(first_box), abs=0.01)
    assert parsed.document["pages"][0]["rotation"] == 90
    caption = next(entry for entry in parsed.evidence if entry["evidence_id"] == figure["caption_evidence_id"])
    assert caption["text"] == figure["text"] == "Fig. 1. A visible sample."
    with pymupdf.open(path) as original:
        assert figure["locator"]["bbox"] == pytest.approx(original[0].get_image_info()[0]["bbox"], abs=0.01)


def test_detects_left_column_before_right_column_and_keeps_their_coordinates(tmp_path):
    columns = [
        ((35, 50), "Left first."),
        ((320, 50), "Right first."),
        ((35, 105), "Left second."),
        ((320, 105), "Right second."),
    ]
    with pymupdf.open() as paper:
        page = paper.new_page()
        for position, text in columns:
            page.insert_text(position, text)
        path = save(paper, tmp_path / "columns.pdf")
    parsed = parse_pdf(path)
    assert [entry["text"] for entry in parsed.evidence] == [
        "Left first.", "Left second.", "Right first.", "Right second.",
    ]
    with pymupdf.open(path) as original:
        for item in parsed.evidence:
            assert original[0].get_textbox(pymupdf.Rect(item["locator"]["bbox"])) == item["text"]


def test_optional_ocr_records_success_and_does_not_touch_native_pages(tmp_path, monkeypatch):
    with pymupdf.open() as paper:
        paper.new_page()
        page = paper.new_page()
        page.insert_text((40, 100), "Native text.")
        path = save(paper, tmp_path / "scan.pdf")
    assert parse_pdf(path).document["pages"][0]["status"] == "no_text"

    original_get_text = pymupdf.Page.get_text
    calls = []
    sentinel = object()

    def ocr(page, *, language, dpi, full):
        calls.append((page.number, language, dpi, full))
        return sentinel

    def text(page, option, **kwargs):
        if kwargs.get("textpage") is sentinel:
            return [(40, 80, 72, 100, "Scanned", 0, 0, 0), (74, 80, 95, 100, "line.", 0, 0, 1)]
        return original_get_text(page, option, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", ocr)
    monkeypatch.setattr(pymupdf.Page, "get_text", text)
    parsed = parse_pdf(path, ocr_language="eng+chi_sim")
    assert calls == [(0, "eng+chi_sim", 250, True)]
    first, second = parsed.document["pages"]
    assert (first["status"], first["ocr_status"], first["ocr_language"]) == ("ocr", "completed", "eng+chi_sim")
    assert second["status"] == "parsed" and "ocr_status" not in second
    scanned, native = parsed.evidence
    assert scanned["text_source"] == "ocr" and scanned["text"] == "Scanned line."
    assert scanned["locator"]["bbox"] == [40, 80, 95, 100]
    assert "text_source" not in native
    assert parse_pdf(path, ocr_language="eng").evidence[0]["evidence_id"] != scanned["evidence_id"]


def test_failed_ocr_is_unreadable_and_creates_no_evidence(tmp_path, monkeypatch):
    with pymupdf.open() as paper:
        paper.new_page()
        path = save(paper, tmp_path / "unreadable.pdf")

    def unavailable(*args, **kwargs):
        raise RuntimeError("Tesseract language data unavailable")

    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", unavailable)
    parsed = parse_pdf(path, ocr_language="eng")
    assert parsed.evidence == []
    page, = parsed.document["pages"]
    assert page["status"] == "unreadable"
    assert page["ocr_status"] == "unavailable_or_failed"
