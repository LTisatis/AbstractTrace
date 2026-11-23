import pymupdf

from abstract_review.reporting import render_markdown
from abstract_review.review import PdfReviewInput, ReviewService


ORIGINAL = "Against the same baseline, accuracy increased by 10 percentage points on datasets A, B, and C."
REVISED = "Against the same baseline, accuracy increased by 10 percentage points on dataset A and by 4 percentage points on dataset B."
TRIAL_NOTE = "The separate trial is still being prepared."
TABLE = (("Dataset", "Baseline", "Final"), ("A", "80%", "90%"), ("B", "70%", "74%"))


def _paper(path):
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((55, 50), "An accuracy evaluation", fontsize=18)
        page.insert_text((55, 90), "Abstract", fontsize=13)
        page.insert_text((55, 115), ORIGINAL, fontsize=10)
        page.insert_text((55, 155), "1. Methods", fontsize=13)
        page.insert_text((55, 180), "Both completed evaluations use the same baseline.")

        page = pdf.new_page()
        page.insert_text((55, 50), "2. Results", fontsize=13)
        page.insert_text((55, 75), "Table 2. Accuracy by dataset")
        xs, ys = (55, 155, 255, 355), (100, 128, 156, 184)
        for x in xs:
            page.draw_line((x, ys[0]), (x, ys[-1]))
        for y in ys:
            page.draw_line((xs[0], y), (xs[-1], y))
        for row, values in enumerate(TABLE):
            for column, value in enumerate(values):
                page.insert_text((xs[column] + 10, ys[row] + 18), value)

        page = pdf.new_page()
        page.insert_text((55, 50), "3. Unfinished trial", fontsize=13)
        page.insert_text((55, 75), TRIAL_NOTE)
        path.write_bytes(pdf.tobytes(no_new_id=True))


def _cell_ids(material, dataset):
    cells = {
        row["column_header"]: row
        for row in material["evidence"]
        if row["kind"] == "table_cell" and row.get("row_header") == dataset
    }
    operands = [cells[column] for column in ("Baseline", "Final")]
    assert tuple(cell["text"] for cell in operands) == next(row[1:] for row in TABLE if row[0] == dataset)
    return [cell["evidence_id"] for cell in operands]


class _ReviewModel:
    elapsed_seconds = 0

    def __init__(self):
        self.calls = []
        self.events = []

    def call(self, stage, material, schema):
        self.calls.append((stage, material))
        if stage == "extract":
            assert material["sentences"][0]["text"] == ORIGINAL
            return schema.model_validate({
                "claims": [
                    {"sentence_id": "s1", "quote": f"{dataset}{suffix}", "kind": "quantitative",
                     "proposition": f"Against the same baseline, accuracy increased by 10 percentage points on dataset {dataset}.",
                     "requirements": [f"dataset {dataset} accuracy result"],
                     "support_query": f"Table 2 Dataset {dataset} accuracy " + " ".join(values),
                     "counter_query": f"Table 2 Dataset {dataset} baseline " + " ".join(values)}
                    for dataset, suffix, values in zip("ABC", (",", ",", "."), (TABLE[1][1:], TABLE[2][1:], ()))
                ],
                "compounds": [{"sentence_id": "s1", "operator": "and", "claim_indexes": [0, 1, 2]}],
            })
        if stage == "judge":
            return self._judge(material, schema)
        if stage == "revise":
            a, b = (_cell_ids(material, dataset) for dataset in "AB")
            note = next(row["evidence_id"] for row in material["evidence"] if row["text"] == TRIAL_NOTE)
            return schema(replacement=REVISED, evidence_ids=[*a, *b, note], semantic_diff=[
                {"kind": "retained", "field": "dataset A accuracy", "before": "10", "after": "10", "evidence_ids": a},
                {"kind": "modified", "field": "dataset B accuracy", "before": "10", "after": "4", "evidence_ids": b},
                {"kind": "removed", "field": "dataset C accuracy", "before": "10", "after": None, "evidence_ids": [note]},
            ])
        if stage == "verify_sentence":
            assert material["original_sentence"] == ORIGINAL
            assert material["revised_sentence"] == REVISED
            assert {"findings", "verification_feedback", "semantic_diff"}.isdisjoint(material)
            evidence_ids = [eid for dataset in "AB" for eid in _cell_ids(material, dataset)]
            return schema(status="passed", reason="Both retained values match Table 2.", evidence_ids=evidence_ids)
        assert stage == "verify_abstract"
        assert material["original_abstract"] == ORIGINAL and material["revised_abstract"] == REVISED
        evidence_ids = [eid for dataset in "AB" for eid in _cell_ids(material, dataset)]
        return schema(status="passed", reason="Only the measured datasets are named.", evidence_ids=evidence_ids, issues=[])

    @staticmethod
    def _judge(material, schema):
        requirement, = material["requirements"]
        dataset = requirement.split()[1]
        if dataset == "C":
            if material["follow_up_already_performed"]:
                assert any(scope["complete"] for scope in material["direct_inspections"])
                assert TRIAL_NOTE in {row["text"] for row in material["evidence"]}
                targets = []
            else:
                targets = [{"query": "Unfinished trial", "requirement": requirement, "page_index": 2}]
            return schema(verdict="insufficient", reason="Dataset C has no reported accuracy result.",
                          evidence_groups=[], missing_fields=[requirement], follow_up_queries=[], follow_up_targets=targets)
        evidence_ids = _cell_ids(material, dataset)
        refuted = dataset == "B"
        return schema(verdict="refuted" if refuted else "supported", reason="The aligned Table 2 cells give the change.",
                      evidence_groups=[{"role": "refute" if refuted else "support", "evidence_ids": evidence_ids}],
                      missing_fields=[], follow_up_queries=[],
                      coverage=[{"requirement": requirement, "status": "found", "evidence_ids": evidence_ids}])


def test_compound_pdf_review_keeps_the_counterexample_and_verifies_the_scoped_edit(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    _paper(pdf)
    model = _ReviewModel()
    service = ReviewService(tmp_path / "state.sqlite3")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(service, "_responses", lambda *args, **kwargs: model)

    run = service.start(PdfReviewInput(pdf, engine="openai"))
    report = service.export(run["run_id"], run["report_version_id"])
    assert run["run_status"] == "report_ready", report.get("runtime_error")
    assert report["usage"]["external_requests"] == 0
    findings = {row["requirements"][0].split()[1]: row for row in report["findings"]}
    assert {dataset: row["verdict"] for dataset, row in findings.items()} == {
        "A": "supported", "B": "refuted", "C": "insufficient",
    }
    assert findings["C"]["coverage"][0]["status"] == "searched_missing"
    original_calculations = {record["claim_id"]: record for record in report["calculations"]}
    for dataset, amount in (("A", "10"), ("B", "4")):
        finding = findings[dataset]
        record = original_calculations[finding["claim_id"]]
        assert (record["result"], record["evidence_ids"]) == (amount, finding["evidence_ids"])
    compound, = report["compound_findings"]
    assert (compound["operator"], compound["verdict"], compound["text"]) == ("and", "refuted", ORIGINAL)
    assert compound["member_claim_ids"] == [findings[dataset]["claim_id"] for dataset in "ABC"]
    assert set(findings["B"]["evidence_ids"]) <= set(compound["evidence_ids"])
    sources = {row["evidence_id"]: row for row in report["evidence"]}
    assert [(sources[eid]["text"], sources[eid]["locator"]["page_index"])
            for eid in findings["B"]["evidence_ids"]] == [("70%", 1), ("74%", 1)]

    assert report["reviewed_abstract"]["text"] == REVISED
    assert report["abstract_review"]["version_id"] == run["report_version_id"]
    sentence_check, = report["abstract_review"]["sentence_checks"]
    assert sentence_check["status"] == report["abstract_review"]["status"] == "passed"
    assert [(record["result"], record["verification"], record["evidence_ids"])
            for record in sentence_check["calculations"]] == [
        ("10", "matched", findings["A"]["evidence_ids"]),
        ("4", "matched", findings["B"]["evidence_ids"]),
    ]
    assert [stage for stage, _ in model.calls if stage.startswith("verify_")] == ["verify_sentence", "verify_abstract"]
    markdown = render_markdown(report)
    assert ORIGINAL in markdown and REVISED in markdown
    assert "## Compound findings" in markdown
