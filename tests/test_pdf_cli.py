import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pymupdf
import pytest


GENERATOR = Path(__file__).resolve().parents[1] / "examples" / "d02" / "make_sample.py"
PDF_ABSTRACT = "Accuracy increased from 80% to 88% on dataset A."
BODY_TEXT = "Dataset A uses a fixed test split for the primary evaluation."
SUPPLEMENT_TEXT = "The supplement records the evaluation seed as 314159."


@pytest.fixture
def pdf_case(tmp_path):
    workspace = tmp_path / "本地 审稿"
    workspace.mkdir()
    data_dir = workspace / "审稿 数据"
    hook_dir = workspace / "hooks"
    hook_dir.mkdir()
    (hook_dir / "sitecustomize.py").write_text(
        "import sys\n"
        "def offline(event, args):\n"
        "    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:\n"
        "        raise RuntimeError(f'Unexpected network access: {event}')\n"
        "sys.addaudithook(offline)\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(hook_dir), env.get("PYTHONPATH", "")) if path
    )
    env["ABSTRACT_REVIEW_DATA_DIR"] = str(data_dir)

    def run_python(*args):
        return subprocess.run(
            [sys.executable, *map(str, args)],
            cwd=workspace,
            env=env,
            capture_output=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )

    paper = workspace / "本地 论文.pdf"
    supplement = workspace / "补充 材料.pdf"
    generated = run_python(GENERATOR, paper, "--supplement", supplement)
    assert generated.returncode == 0, generated.stderr

    def cli(*args, success=True):
        result = run_python("-m", "abstract_review", *args)
        if success:
            assert result.returncode == 0, result.stderr
            return json.loads(result.stdout)
        assert result.returncode != 0
        assert "Traceback" not in result.stderr
        return result

    return cli, paper, supplement, data_dir


def export(cli, started):
    return cli(
        "review", "export", started["run_id"],
        "--version-id", started["report_version_id"], "--format", "json",
    )


def assert_blob(data_dir, item, expected_bytes):
    relative_path = Path(item["blob_path"])
    assert not relative_path.is_absolute()
    stored = (data_dir / relative_path).resolve()
    assert stored.is_relative_to(data_dir.resolve())
    assert stored.read_bytes() == expected_bytes
    assert item["sha256"] == hashlib.sha256(expected_bytes).hexdigest()
    return stored


def assert_crop(pdf, evidence, text):
    locator = evidence["locator"]
    page = pdf[locator["page_index"]]
    rect = pymupdf.Rect(locator["bbox"])
    assert not rect.is_empty
    assert rect in page.rect
    assert text in " ".join(page.get_textbox(rect).split())


def test_pdf_review_reports_original_body_and_table_coordinates(pdf_case):
    cli, paper, _, data_dir = pdf_case
    started = cli("review", "start", "--pdf", paper)
    assert started["run_status"] == "report_ready"
    assert cli("review", "status", started["run_id"]) == started
    report = export(cli, started)

    assert report["mode"] == "pdf"
    assert report["original_abstract"]["text"] == PDF_ABSTRACT
    assert report["reviewed_abstract"]["version_id"] == started["report_version_id"]
    assert report["findings"] == []
    assert report["abstract_review"]["status"] == "pending"

    main, = report["documents"]
    assert (main["role"], main["name"]) == ("main", paper.name)
    assert [(page["page_index"], page["page_number"], page["status"]) for page in main["pages"]] == [
        (0, 1, "parsed"), (1, 2, "parsed"),
    ]
    stored = assert_blob(data_dir, main, paper.read_bytes())
    body, = [
        item for item in report["evidence"]
        if item["kind"] == "body_text" and BODY_TEXT in item["text"]
    ]
    cells = {
        (item["locator"]["row_index"], item["locator"]["column_index"]): item
        for item in report["evidence"] if item["kind"] == "table_cell"
    }
    assert {position: item["text"] for position, item in cells.items()} == {
        (0, 0): "Dataset", (0, 1): "Baseline", (0, 2): "Final",
        (1, 0): "A", (1, 1): "80%", (1, 2): "88%",
        (2, 0): "B", (2, 1): "70%", (2, 2): "79%",
    }
    result = cells[1, 2]
    assert body["doc_id"] == result["doc_id"] == main["doc_id"]
    assert body["locator"]["page_index"] == 0
    assert "Introduction" in body["locator"]["section"]
    assert result["locator"]["page_index"] == 1
    assert result["locator"]["bbox"] == pytest.approx([268, 148, 370, 174], abs=1)
    with pymupdf.open(paper) as original, pymupdf.open(stored) as retained:
        assert "88%" in original[0].get_text()
        for pdf in (original, retained):
            assert_crop(pdf, body, BODY_TEXT)
            assert_crop(pdf, result, "88%")


def test_pdf_review_keeps_two_abstracts_and_only_adds_confirmed_supplement(pdf_case):
    cli, paper, supplement, data_dir = pdf_case
    pasted = paper.parent / "另贴 摘要.txt"
    pasted_text = "这是作者提供的摘要。\r\n待核查准确率。"
    pasted.write_bytes(pasted_text.encode("utf-8"))

    default = export(cli, cli("review", "start", "--pdf", paper, "--abstract-file", pasted))
    assert default["reviewed_abstract"]["text"] == PDF_ABSTRACT
    assert [item["role"] for item in default["documents"]] == ["main"]
    assert not any(SUPPLEMENT_TEXT in item["text"] for item in default["evidence"])

    selected = cli(
        "review", "start", "--pdf", paper, "--supplement", supplement,
        "--abstract-file", pasted, "--abstract-source", "pasted",
    )
    report = export(cli, selected)
    by_kind = {item["kind"]: item for item in report["abstract_versions"]}
    assert set(by_kind) == {"pdf_original", "pasted_original"}
    assert by_kind["pdf_original"]["text"] == PDF_ABSTRACT
    assert by_kind["pasted_original"]["text"] == pasted_text
    assert report["original_abstract"]["text"] == pasted_text
    assert report["reviewed_abstract"]["text"] == pasted_text
    assert selected["report_version_id"] == by_kind["pasted_original"]["version_id"]
    assert default["report_version_id"] != selected["report_version_id"]

    attached, = [item for item in report["documents"] if item["role"] == "supplement"]
    stored = assert_blob(data_dir, attached, supplement.read_bytes())
    source, = [item for item in report["evidence"] if SUPPLEMENT_TEXT in item["text"]]
    assert source["doc_id"] == attached["doc_id"]
    with pymupdf.open(stored) as original:
        assert_crop(original, source, SUPPLEMENT_TEXT)


def test_same_filename_with_new_content_keeps_initial_source(pdf_case):
    cli, paper, _, data_dir = pdf_case
    before = paper.read_bytes()
    initial = cli("review", "start", "--pdf", paper)
    original = export(cli, initial)
    with pymupdf.open(stream=before, filetype="pdf") as modified:
        modified[0].insert_text((64, 270), "This line is in the replacement only.", fontsize=11)
        after = modified.tobytes(no_new_id=True)
    paper.write_bytes(after)

    newer = export(cli, cli("review", "start", "--pdf", paper))
    original_doc, = original["documents"]
    newer_doc, = newer["documents"]
    assert newer_doc["name"] == original_doc["name"]
    assert newer_doc["sha256"] != original_doc["sha256"]
    assert newer_doc["doc_id"] != original_doc["doc_id"]
    assert_blob(data_dir, original_doc, before)
    assert_blob(data_dir, newer_doc, after)
    assert export(cli, initial) == original


def test_pdf_errors_report_invalid_source_without_attaching_it(pdf_case):
    cli, paper, _, _ = pdf_case
    invalid = paper.parent / "无效 文件.pdf"
    invalid.write_bytes(b"this is not a PDF")
    assert cli("review", "start", "--pdf", invalid, success=False).stderr

    started = cli("review", "start", "--pdf", paper, "--supplement", invalid)
    report = export(cli, started)
    assert [item["role"] for item in report["documents"]] == ["main"]
    assert any(
        diagnostic["code"] == "supplement_unreadable" and diagnostic["name"] == invalid.name
        for diagnostic in report["diagnostics"]
    )
