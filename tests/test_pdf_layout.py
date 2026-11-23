from pathlib import Path

import pymupdf
import pytest

from abstract_review.evidence import EvidenceIndex
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


def layered_table(page, label):
    page.insert_text((43, 83), label, fontsize=9)
    for x in (40, 120, 254, 388):
        page.draw_line((x, 100), (x, 204))
    for x in (188, 322):
        page.draw_line((x, 125), (x, 204))
    for y in (100, 125, 150, 177, 204):
        page.draw_line((40, y), (388, y))
    for x, text in ((48, "Dataset"), (130, "Accuracy"), (261, "F1")):
        page.insert_text((x, 119), text, fontsize=9)
    for y, values in ((144, ("", "Baseline", "Final", "Baseline", "Final")),
                      (169, ("A", "80%", "88%(a)", "80%", "88%(b)")),
                      (195, ("B", "44%", "48%", "41%", "49%"))):
        for x, text in zip((48, 128, 196, 264, 330), values):
            if text:
                page.insert_text((x, y), text, fontsize=9)


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


def test_layered_headers_notes_and_experiment_anchors_follow_actual_pdf_boxes(tmp_path):
    with pymupdf.open() as paper:
        methods = paper.new_page(width=435, height=500)
        methods.insert_text((41, 42), "Methods")
        methods.insert_text((41, 68), "For Experiment E1, we used a fixed test split.")
        methods.insert_text((41, 93), "For Experiment E2, we used the tuning split.")
        results = paper.new_page(width=435, height=500)
        results.insert_text((41, 42), "Results")
        layered_table(results, "Table 2. Results for Experiment E1.")
        results.insert_text((42, 219), "(a) Accuracy used the fixed test split.", fontsize=9)
        results.insert_text((42, 232), "(b) F1 used the tuning split.", fontsize=9)
        path = save(paper, tmp_path / "anchored.pdf")

    found = parse_pdf(path).evidence
    ids = {item["evidence_id"]: item for item in found}
    a = next(item for item in found if item["text"] == "88%(a)")
    b = next(item for item in found if item["text"] == "88%(b)")
    caption = ids[a["caption_evidence_id"]]
    assert a["column_header_path"] == ["Accuracy", "Final"]
    assert b["column_header_path"] == ["F1", "Final"]
    assert a["column_header"] == b["column_header"] == "Final"
    for cell, marker in ((a, "a"), (b, "b")):
        note_link, = cell["note_links"]
        note = ids[note_link["note_evidence_id"]]
        assert note["kind"] == "table_note" and note["note_marker"] == marker
        assert note["note_scope"] == "marked_cells"
        assert note["locator"]["page_index"] == cell["locator"]["page_index"] == 1
        assert note["locator"]["table_index"] == cell["locator"]["table_index"]
        assert note_link["anchor_evidence_id"] == cell["evidence_id"]
        assert cell["numeric_text"] == "88%" and cell["text"].endswith(f"({marker})")
        assert ids[cell["row_header_evidence_id"]]["text"] == "A"
        assert [ids[eid]["text"] for eid in cell["column_header_evidence_ids"]] == cell["column_header_path"]

    assert [ref["experiment_id"] for ref in caption["experiment_refs"]] == ["e1"]
    matching = next(item for item in found if item["locator"]["section"] == "Methods" and "Experiment E1" in item["text"])
    foil = next(item for item in found if item["locator"]["section"] == "Methods" and "Experiment E2" in item["text"])
    assert [ref["experiment_id"] for ref in foil["experiment_refs"]] == ["e2"]
    with pymupdf.open(path) as pdf:
        for item in (a, b, caption, *(ids[eid] for eid in a["column_header_evidence_ids"]),
                     *(ids[eid] for eid in a["note_evidence_ids"])):
            locator = item["locator"]
            assert pdf[locator["page_index"]].get_textbox(pymupdf.Rect(locator["bbox"])).strip() == item["text"]
        ref, = matching["experiment_refs"]
        assert ref["text"] == "Experiment E1"
        assert "Experiment E1" in pdf[0].get_textbox(pymupdf.Rect(ref["bbox"]))
    assert parse_pdf(path).evidence == found


def test_ambiguous_duplicate_and_unlinked_markers_leave_numeric_text_raw(tmp_path):
    with pymupdf.open() as paper:
        page = paper.new_page(width=435, height=500)
        layered_table(page, "Table 2. Duplicate marker.")
        page.insert_text((42, 219), "(a) Accuracy used the fixed test split.", fontsize=9)
        page.insert_text((42, 235), "(a) Accuracy used a different split.", fontsize=9)
        page.insert_text((42, 374), "Note for Table 9: This refers elsewhere.", fontsize=9)
        path = save(paper, tmp_path / "unproven.pdf")
    evidence = parse_pdf(path).evidence
    assert not any(item["kind"] == "table_note" for item in evidence)
    marked = [item for item in evidence if item["text"] in {"88%(a)", "88%(b)"}]
    assert len(marked) == 2 and all("numeric_text" not in item and "note_links" not in item for item in marked)


def test_explicit_table_note_links_outside_the_footer_only_when_label_is_unique(tmp_path):
    with pymupdf.open() as paper:
        page = paper.new_page(width=435, height=500)
        layered_table(page, "Table 2. Named note reference.")
        page.insert_text((42, 374), "Note for Table 2: (a) Accuracy used the fixed test split.", fontsize=9)
        path = save(paper, tmp_path / "named-note.pdf")
    evidence = parse_pdf(path).evidence
    note, = [item for item in evidence if item["kind"] == "table_note"]
    assert note["table_label"] == "2" and note["note_marker"] == "a"
    assert note["text"].startswith("Note for Table 2:")
    marked = next(item for item in evidence if item["text"] == "88%(a)")
    assert marked["note_evidence_ids"] == [note["evidence_id"]]
    assert marked["numeric_text"] == "88%"


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


def test_embedded_subfigures_need_both_caption_and_unambiguous_panel_labels(tmp_path):
    pixels = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
    pixels.clear_with(140)
    left = pymupdf.Rect(40, 75, 170, 180)
    right = pymupdf.Rect(200, 75, 330, 180)
    with pymupdf.open() as paper:
        page = paper.new_page()
        for box, label in ((left, "a"), (right, "b")):
            page.insert_image(box, pixmap=pixels, keep_proportion=False)
            page.insert_text((box.x0 + 5, box.y0 + 13), f"({label})")
        page.insert_text((43, 202), "Figure 8. (a) First approximation and (b) the final response.")

        page = paper.new_page()
        combined = pymupdf.Rect(40, 75, 330, 180)
        page.insert_image(combined, pixmap=pixels, keep_proportion=False)
        page.insert_text((45, 89), "(a)")
        page.insert_text((204, 89), "(b)")
        page.insert_text((43, 202), "Figure 9. (a) and (b) in one embedded bitmap.")
        path = save(paper, tmp_path / "panels.pdf")

    records = parse_pdf(path).evidence
    figures = [item for item in records if item["kind"] == "figure"]
    parent = next(item for item in figures if item["source_type"] == "embedded_panels")
    panels = {item["locator"]["panel_label"]: item for item in figures if item["locator"].get("panel_label")}
    assert set(panels) == {"a", "b"}
    for label, box in (("a", left), ("b", right)):
        panel = panels[label]
        assert panel["parent_figure_evidence_id"] == parent["evidence_id"]
        assert panel["locator"]["bbox"] == pytest.approx(tuple(box), abs=0.01)
        assert panel["caption_evidence_id"] == parent["caption_evidence_id"]
    assert [item["source_type"] for item in figures if item["figure_label"] == "9"] == ["embedded_image"]

    index = EvidenceIndex(records)
    try:
        result = index.search("Figure 8(b)", limit=1)
        assert result["evidence"] == [panels["b"]]
        assert index.search("Figure 9(b)", limit=1)["evidence"] == []
    finally:
        index.close()


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
