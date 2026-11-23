import json

import pymupdf

from abstract_review.claimbench import prepare_claimbench
from abstract_review.ingest import parse_pdf


TABLE = "Outcome Score Exact 73.5 Partial 20.0"
CAPTION = "Table 1: Scores for the evaluation."
PROSE = "The reviewers summarized two outcomes."


def _paper(path, *, repeat=False):
    with pymupdf.open() as document:
        for _ in range(2 if repeat else 1):
            page = document.new_page()
            page.insert_text((60, 75), PROSE)
            page.draw_rect((60, 100, 300, 190))
            page.draw_line((200, 100), (200, 190))
            for y in (130, 160):
                page.draw_line((60, y), (300, y))
            for y, left, right in ((120, "Outcome", "Score"), (150, "Exact", "73.5"), (180, "Partial", "20.0")):
                page.insert_text((68, y), left)
                page.insert_text((210, y), right)
            page.insert_text((60, 212), CAPTION)
        document.save(path)


def _prepare(pdf, output_dir, excerpts):
    annotations = output_dir / "annotations.json"
    annotations.write_text(json.dumps({"annotations": [{"claim": "Example.", "evidences": excerpts}]}), encoding="utf-8")
    _, gold = prepare_claimbench(
        pdf, annotations, output_dir=output_dir, paper_family_id="table", split="dev", dataset_version="fixture",
    )
    return gold["labels"][0]["passages"]


def test_table_and_adjacent_prose_require_exact_tokens_and_use_existing_anchors(tmp_path):
    pdf = tmp_path / "table.pdf"
    _paper(pdf)
    table, mixed, different_value, wrong_order, absent_middle = _prepare(pdf, tmp_path, [
        TABLE,
        f"{PROSE}\n{TABLE}\n{CAPTION}",
        TABLE.replace("73.5", "74.5"),
        "Outcome Exact Partial Score 73.5 20.0",
        f"{PROSE} Outcome Score {CAPTION}",
    ])

    assert [table["status"], mixed["status"], different_value["status"], wrong_order["status"],
            absent_middle["status"]] == ["aligned", "aligned", "unmatched", "unmatched", "unmatched"]
    assert table["normalization"] == mixed["normalization"] == "table_layout_exact"
    assert len(table["evidence_ids"]) == 6
    assert len(mixed["evidence_ids"]) == 8
    source = {row["evidence_id"]: row for row in parse_pdf(pdf).evidence}
    assert [source[eid]["kind"] for eid in table["evidence_ids"]] == ["table_cell"] * 6
    assert [source[eid]["text"] for eid in mixed["evidence_ids"] if source[eid]["kind"] == "body_text"] == [PROSE, CAPTION]
    for anchor in mixed["anchors"]:
        item = source[anchor["evidence_id"]]
        assert anchor["doc_id"] == item["doc_id"]
        assert anchor["page_index"] == item["locator"]["page_index"] == 0
        assert anchor["bbox"] == item["locator"]["bbox"]


def test_repeat_exact_table_remains_ambiguous_and_cross_page_quote_stays_unmatched(tmp_path):
    pdf = tmp_path / "repeated-table.pdf"
    _paper(pdf, repeat=True)
    repeated, crosses_pages = _prepare(pdf, tmp_path, [TABLE, f"{CAPTION} {PROSE} {TABLE}"])
    assert repeated == {"status": "ambiguous", "evidence_ids": [], "anchors": []}
    assert crosses_pages == {"status": "unmatched", "evidence_ids": [], "anchors": []}
