import pymupdf

from abstract_review import ingest


def _bold(page, x, y, text):
    page.insert_text((x, y), text, fontname="hebo", fontsize=11)


def test_native_page_headings_recover_split_title_without_rewriting_evidence(tmp_path, monkeypatch):
    path = tmp_path / "headings.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        _bold(page, 70, 75, "1 Introduction")
        page.insert_text((70, 97), "The study starts here.")
        _bold(page, 70, 150, "3.1")
        _bold(page, 108, 150, "Task Definition")
        page.insert_text((70, 177), "The benchmark uses controlled tasks.")
        _bold(page, 70, 227, "3.2")
        _bold(page, 108, 227, "Comparison with Other Long-Context")
        _bold(page, 108, 241, "Benchmarks")
        page.insert_text((70, 267), "The source evidence keeps its original location.")
        document.save(path)

    with_headings = ingest.parse_pdf(path)
    (page,) = with_headings.document["pages"]
    assert page["source_headings"] == [
        "1 Introduction", "3.1 Task Definition", "3.2 Comparison with Other Long-Context Benchmarks",
    ]

    monkeypatch.setattr(ingest, "_source_headings", lambda *_: [])
    without_headings = ingest.parse_pdf(path)
    assert with_headings.evidence == without_headings.evidence
    assert with_headings.abstract == without_headings.abstract
    assert with_headings.document["parser_version"] == without_headings.document["parser_version"]


def test_headings_omit_contents_page_and_non_heading_text(tmp_path):
    path = tmp_path / "layout-noise.pdf"
    with pymupdf.open() as document:
        contents = document.new_page()
        _bold(contents, 70, 65, "Contents")
        _bold(contents, 70, 100, "1 Introduction")
        _bold(contents, 70, 125, "2 Method")

        page = document.new_page()
        _bold(page, 70, 75, "2.1 Tasks ........ 15")
        _bold(page, 70, 100, "3 Evaluation Results")
        page.insert_text((360, 100), "37")
        page.insert_text((70, 125), "4.1 Experimental Details")
        _bold(page, 70, 150, "5.1 We calculated summaries")
        _bold(page, 70, 175, "6.1 Figure 3: A schematic")
        _bold(page, 70, 200, "6.2")
        _bold(page, 335, 200, "Text in Another Column")
        page.draw_rect((60, 230, 290, 310))
        page.draw_line((200, 230), (200, 310))
        page.draw_line((60, 270), (290, 270))
        _bold(page, 70, 252, "7.1 Table Cell")
        _bold(page, 210, 252, "Value")
        page.insert_text((70, 292), "Example")
        page.insert_text((210, 292), "18")
        document.save(path)

    parsed = ingest.parse_pdf(path)
    assert all("source_headings" not in page for page in parsed.document["pages"])
    assert any(item["kind"] == "table_cell" and item["text"] == "7.1 Table Cell" for item in parsed.evidence)
