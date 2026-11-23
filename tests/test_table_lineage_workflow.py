import pymupdf
from fastapi.testclient import TestClient

from abstract_review.calculations import check_revision_numbers
from abstract_review.evidence import EvidenceIndex, with_method_scope
from abstract_review.ingest import parse_pdf
from abstract_review.reporting import render_markdown
from abstract_review.review import PdfReviewInput, ReviewService
from abstract_review.web import create_app


CLAIMS = (
    "In Experiment E1, on the test split of dataset A, accuracy increased by 8 percentage points.",
    "In Experiment E1, on the test split of dataset A, accuracy showed a relative increase of 10%.",
)
TARGET_QUERY = "Table 2 88%"
METHOD = "Experiment E1 used a fixed test split for its final evaluation."
FOIL_METHOD = "Experiment E2 used a separate validation split during tuning."
NOTE_A = "(a) Accuracy was evaluated using the fixed test split."
NOTE_B = "(b) F1 was measured on the tuning validation split."
CAPTION = "Table 2. Results for Experiment E1."
SCOPE = "test split and aligned accuracy operands"


def _table(page, matrix, *, merged=False):
    xs = [60 + 92 * column for column in range(len(matrix[0]) + 1)]
    ys = [100 + 26 * row for row in range(len(matrix) + 1)]
    for column, x in enumerate(xs):
        start = ys[1] if merged and column in {2, 4} else ys[0]
        page.draw_line((x, start), (x, ys[-1]))
    for row, y in enumerate(ys):
        start = xs[1] if merged and row == 1 else xs[0]
        page.draw_line((start, y), (xs[-1], y))
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            if value:
                page.insert_text((xs[column] + 8, ys[row] + 17), value, fontsize=10)


def _paper(path, *, notes=True, foil_method=FOIL_METHOD):
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((60, 50), "Synthetic evaluation", fontsize=18)
        page.insert_text((60, 90), "Abstract", fontsize=13)
        page.insert_text((60, 115), CLAIMS[0], fontsize=10)
        page.insert_text((60, 135), CLAIMS[1], fontsize=10)
        page.insert_text((60, 180), "1. Introduction", fontsize=13)

        page = pdf.new_page()
        page.insert_text((60, 50), "2. Methods", fontsize=13)
        page.insert_text((60, 86), METHOD)
        page.insert_text((60, 121), foil_method)

        page = pdf.new_page()
        page.insert_text((60, 50), "3. Results", fontsize=13)
        page.insert_text((60, 80), CAPTION)
        _table(page, (
            ("Dataset", "Accuracy", None, "F1", None),
            (None, "Baseline", "Final", "Baseline", "Final"),
            ("A", "80%", "88%(a)" if notes else "88%", "36%", "42%(b)" if notes else "42%"),
        ), merged=True)
        if notes:
            page.insert_text((60, 191), NOTE_A)
            page.insert_text((60, 204), NOTE_B)

        page = pdf.new_page()
        page.insert_text((60, 80), "Table 3. Results for Experiment E2.")
        _table(page, (("Dataset", "Baseline accuracy", "Final accuracy"), ("A", "80%", "88%")))
        path.write_bytes(pdf.tobytes(no_new_id=True))


def _actual_material(payload):
    rows = list(payload["evidence"])
    for group in payload.get("table_context", ()):
        rows.extend(group["evidence"])
    return {row["evidence_id"]: row for row in rows}


class _ReviewModel:
    elapsed_seconds = 0

    def __init__(self):
        self.events = []
        self.judgements = []

    def call(self, stage, material, schema):
        if stage == "extract":
            assert [part["text"] for part in material["sentences"]] == list(CLAIMS)
            return schema(claims=[
                {"sentence_id": f"s{index}", "quote": claim, "proposition": claim, "kind": "quantitative",
                 "requirements": [SCOPE], "support_query": TARGET_QUERY, "counter_query": "Table 2 80%"}
                for index, claim in enumerate(CLAIMS, start=1)
            ])
        if stage == "judge":
            self.judgements.append(material)
            sent = _actual_material(material)
            method_id, = [eid for eid, row in sent.items() if row["text"] == METHOD]
            note_id, = [eid for eid, row in sent.items() if row["text"] == NOTE_A]
            calculation = material["calculation"]
            assert calculation is not None, material["claim"]
            operands = calculation["evidence_ids"]
            assert [sent[eid]["text"] for eid in operands] == ["80%", "88%(a)"]
            ids = [*operands, method_id, note_id]
            return schema(verdict="supported", reason="The measured result uses the test split identified in Methods.",
                          evidence_groups=[{"role": "support", "evidence_ids": ids}], missing_fields=[],
                          follow_up_queries=[], coverage=[{"requirement": SCOPE, "status": "found", "evidence_ids": ids}])
        assert stage == "verify_abstract"
        assert material["original_abstract"] == material["revised_abstract"]
        return schema(status="passed", reason="The unchanged abstract is consistent.", evidence_ids=[], issues=[])


def test_pdf_accuracy_result_reaches_its_header_note_and_experiment_method(tmp_path, monkeypatch):
    path = tmp_path / "experiments.pdf"
    _paper(path)
    model = _ReviewModel()
    service = ReviewService(tmp_path / "state.sqlite3")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(service, "_responses", lambda *args, **kwargs: model)

    run = service.start(PdfReviewInput(path, engine="openai"))
    report = service.export(run["run_id"], run["report_version_id"])
    assert run["run_status"] == "report_ready", report.get("runtime_error")
    assert report["usage"]["external_requests"] == 0
    difference_calculation, relative_calculation = report["calculations"]
    assert [(calc["result"], calc["unit"]) for calc in report["calculations"]] == [
        ("8", "percentage_points"), ("10", "percent"),
    ]
    assert difference_calculation["evidence_ids"] == relative_calculation["evidence_ids"]
    baseline_id, result_id = difference_calculation["evidence_ids"]
    evidence = {row["evidence_id"]: row for row in report["evidence"]}
    base, result = evidence[baseline_id], evidence[result_id]
    assert base["locator"]["page_index"] == result["locator"]["page_index"] == 2
    foil = [row for row in evidence.values() if row["locator"]["page_index"] == 3 and row.get("row_header") == "A"]
    assert [(row["column_header"], row["text"]) for row in foil] == [
        ("Baseline accuracy", "80%"), ("Final accuracy", "88%"),
    ]
    assert (result["text"], result["numeric_text"]) == ("88%(a)", "88%")
    assert [evidence[eid]["text"] for eid in result["column_header_evidence_ids"]] == ["Accuracy", "Final"]
    assert evidence[result["row_header_evidence_id"]]["text"] == "A"
    note_link, = result["note_links"]
    assert (note_link["marker"], note_link["anchor_evidence_id"]) == ("a", result_id)
    note_a = note_link["note_evidence_id"]
    note_b, = [eid for eid, row in evidence.items() if row["kind"] == "table_note" and row["text"] == NOTE_B]
    f1_result, = [row for row in evidence.values() if row["text"] == "42%(b)"]
    assert [link["note_evidence_id"] for link in f1_result["note_links"]] == [note_b]
    method, = [eid for eid, row in evidence.items() if row["text"] == METHOD]
    wrong_method, = [eid for eid, row in evidence.items() if row["text"] == FOIL_METHOD]
    caption, = [eid for eid, row in evidence.items() if row["text"] == CAPTION]
    for eid in (caption, method):
        assert [ref["experiment_id"] for ref in evidence[eid]["experiment_refs"]] == ["e1"]
    assert evidence[note_a]["text"] == NOTE_A
    scope_ids = {note_a, *base["column_header_evidence_ids"], *result["column_header_evidence_ids"]}
    for calculation in report["calculations"]:
        assert scope_ids <= set(calculation["scope_evidence_ids"])
        assert note_b not in calculation["scope_evidence_ids"]

    index = EvidenceIndex(report["evidence"])
    try:
        unexpanded = [row["evidence_id"] for row in index.search(TARGET_QUERY, expand=False)["evidence"]]
    finally:
        index.close()
    assert unexpanded[0] == result_id and method not in unexpanded
    assert foil[1]["evidence_id"] in unexpanded

    assert len(model.judgements) == len(report["findings"]) == 2
    for finding, payload in zip(report["findings"], model.judgements):
        assert finding["verdict"] == "supported"
        actual = _actual_material(payload)
        assert set(finding["sent_candidate_ids"]) == actual.keys()
        assert {baseline_id, result_id, method, note_a} <= actual.keys()
        context, = [group for group in payload["table_context"] if group["cell_evidence_id"] == result_id]
        context_ids = {row["evidence_id"] for row in context["evidence"]}
        assert {note_a, *result["column_header_evidence_ids"]} <= context_ids
        assert {note_b, wrong_method}.isdisjoint(context_ids)
        paths = [walk for walk in finding["retrieval_paths"] if result_id in walk["evidence_ids"]]
        assert any(walk["evidence_ids"][0] == result_id and walk["evidence_ids"][-1] == method for walk in paths)
        assert all(wrong_method not in walk["evidence_ids"] for walk in paths)
        for walk in paths:
            assert all(eid in actual for eid in walk["evidence_ids"])
            for edge in walk["edges"]:
                for anchor in edge["anchors"]:
                    assert (anchor["locator"], anchor["text"]) == (
                        evidence[anchor["evidence_id"]]["locator"], evidence[anchor["evidence_id"]]["text"],
                    )

    markdown = render_markdown(report)
    with pymupdf.open(path) as original:
        for eid in (baseline_id, result_id, note_a, method, *result["column_header_evidence_ids"]):
            row = evidence[eid]
            locator = row["locator"]
            box = pymupdf.Rect(locator["bbox"]) + (-1, -1, 1, 1)
            assert row["text"] in original[locator["page_index"]].get_textbox(box)
            assert f"`{eid}`" in markdown


def test_report_links_open_the_anchored_header_note_and_method_pages(tmp_path, monkeypatch):
    pdf = tmp_path / "experiments.pdf"
    _paper(pdf)
    model = _ReviewModel()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(ReviewService, "_responses", lambda *args, **kwargs: model)
    service = ReviewService(tmp_path / "state.sqlite3")
    run = service.start(PdfReviewInput(pdf, engine="openai"))
    report = service.export(run["run_id"], run["report_version_id"])
    by_text = {row["text"]: row for row in report["evidence"]}
    result = by_text["88%(a)"]
    linked = [by_text["Accuracy"], by_text[NOTE_A], by_text[METHOD]]
    params = {"version_id": run["report_version_id"], "evidence_id": result["evidence_id"]}

    with TestClient(create_app(tmp_path)) as client:
        html = client.get(f"/runs/{run['run_id']}", params=params)
        assert html.status_code == 200
        assert "同一实验" in html.text and "表注" in html.text and "表头" in html.text
        for row in linked:
            assert f"evidence_id={row['evidence_id']}" in html.text
            page = row["locator"]["page_index"]
            preview = client.get(f"/runs/{run['run_id']}/pages/{row['doc_id']}/{page}.png", params={
                **params, "evidence_id": row["evidence_id"],
            })
            assert preview.status_code == 200 and preview.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_only_the_verified_experiment_method_can_supply_an_unnoted_test_split(tmp_path):
    path = tmp_path / "no-notes.pdf"
    foil = "Experiment E2 used the fixed test split of dataset A during tuning."
    _paper(path, notes=False, foil_method=foil)
    evidence = parse_pdf(path).evidence
    methods = {row["text"]: row["evidence_id"] for row in evidence if row["text"] in {METHOD, foil}}
    index = EvidenceIndex(evidence)
    try:
        paths = index.search(TARGET_QUERY)["paths"]
    finally:
        index.close()
    assert any(path["evidence_ids"][-1] == methods[METHOD] for path in paths)
    primary = [row for row in evidence if row["locator"]["page_index"] != 3]
    matched, = check_revision_numbers("", CLAIMS[0], with_method_scope(primary, paths))
    assert (matched["verification"], matched["result"]) == ("matched", "8")
    assert methods[METHOD] in matched["scope_evidence_ids"] and methods[foil] not in matched["scope_evidence_ids"]

    without_method = [row for row in primary if row["evidence_id"] != methods[METHOD]]
    unresolved, = check_revision_numbers("", CLAIMS[0], with_method_scope(without_method, paths))
    assert unresolved["verification"] == "unresolved"
