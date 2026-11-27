"""Exercise the local review workflow with JavaScript enabled."""

import os
import socket
import time
from contextlib import ExitStack
from threading import Thread

import pymupdf
import pytest
import uvicorn

from abstract_review.review import PdfReviewInput, ReviewService
from abstract_review.web import create_app
from test_web import _pdf_bytes


playwright = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.browser


@pytest.fixture
def workspace(tmp_path):
    with ExitStack() as sockets:
        # An ephemeral port can coincide with a port blocked by Chromium.
        for _ in range(10):
            listener = sockets.enter_context(socket.socket())
            listener.bind(("127.0.0.1", 0))
            host, port = listener.getsockname()
            if port >= 1024 and port not in {2049, 3659, 4045, 5060, 5061, 6000, 6566, 6697, 10080, *range(6665, 6670)}:
                break
        else:
            pytest.fail("Could not allocate a browser-accessible port on localhost.")
        server = uvicorn.Server(uvicorn.Config(create_app(tmp_path), log_level="error"))
        thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert server.started
            yield f"http://{host}:{port}", tmp_path
        finally:
            server.should_exit = True
            thread.join(timeout=5)


@pytest.fixture
def page():
    with playwright.sync_playwright() as session:
        executable = os.environ.get("ABSTRACTTRACE_BROWSER_EXECUTABLE")
        browser = session.chromium.launch(executable_path=executable)
        tab = browser.new_page(viewport={"width": 1280, "height": 720})
        try:
            yield tab
        finally:
            browser.close()


def upload(page, name="paper.pdf"):
    page.locator('[name="pdf"]').set_input_files({
        "name": name, "mimeType": "application/pdf", "buffer": _pdf_bytes(),
    })


def test_intake_preserves_inputs_on_error_and_ignores_hidden_paste(workspace, page):
    base_url, _ = workspace
    expect = playwright.expect
    page.goto(base_url)
    source = page.locator('[name="abstract_source"]')
    pasted = page.locator('[name="pasted_abstract"]')
    source.select_option("pasted")
    pasted.fill("A draft that should survive invalid upload.")
    page.get_by_text("补充材料与扩展核查").click()
    page.get_by_label("核查摘要遗漏").check()
    upload(page, "a" * 141 + ".pdf")
    page.get_by_role("button", name="开始审阅").click()
    expect(page.get_by_role("alert").first).to_contain_text("请重新选择")
    expect(source).to_have_value("pasted")
    expect(pasted).to_have_value("A draft that should survive invalid upload.")
    expect(page.get_by_label("核查摘要遗漏")).to_be_checked()

    pasted.fill(" \n ")
    source.select_option("pdf")
    expect(pasted).to_be_disabled()
    upload(page)
    page.get_by_role("button", name="开始审阅").click()
    page.get_by_role("link", name="查看最新报告").click()
    run_url = page.url.split("?")[0].split("#")[0]
    run_id = run_url.rsplit("/", 1)[-1]
    report = page.request.get(f"{base_url}/api/v1/reviews/{run_id}/export").json()
    assert [item["kind"] for item in report["abstract_versions"]] == ["pdf_original"]
    page.get_by_role("link", name="最近审阅").click()
    expect(page.get_by_role("link", name="paper.pdf")).to_be_visible()


def test_failed_edit_can_be_replaced_without_losing_draft(workspace, page):
    base_url, data = workspace
    expect = playwright.expect
    paper = data / "paper.pdf"
    paper.write_bytes(_pdf_bytes())
    service = ReviewService(data / "state.sqlite3")
    original = "The study tracks error rate. This sentence stays."
    ref = service.start(PdfReviewInput(pdf=paper, pasted_abstract=original, abstract_source="pasted", engine="offline"))
    run_id = ref["run_id"]
    with service.store.acquire(run_id) as token:
        report = service.store.read(run_id)
        report.update(run_status="failed", runtime_error="Synthetic verification failure")
        report["revisions"] = [{
            "revision_id": "new-rate-text", "status": "passed", "original_text": "The study tracks error rate.",
            "replacement": "The study reports the error rate.", "evidence_ids": [],
        }]
        service.store.save(report, token)

    page.goto(f"{base_url}/runs/{run_id}")
    page.get_by_role("link", name="重新编辑并复核").click()
    draft = page.get_by_label("确认提交文本")
    draft.fill("The study tracks error rate. My draft stays.")
    page.get_by_role("button", name="采用到编辑稿").click()
    updated = "The study reports the error rate. My draft stays."
    expect(draft).to_have_value(updated)
    page.get_by_role("button", name="保存并复核").click()
    expect(page.locator("#abstract")).to_contain_text(updated)
    expect(page.locator("#abstract ins").first).to_be_visible()


def test_pdf_switch_zoom_and_completion_preserve_preview(workspace, page):
    base_url, data = workspace
    expect = playwright.expect
    paper = data / "paper.pdf"
    paper.write_bytes(_pdf_bytes())
    supplement = data / "appendix.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((30, 50), "Appendix results")
        pdf.save(supplement)
    service = ReviewService(data / "state.sqlite3")
    ref = service.start(PdfReviewInput(pdf=paper, supplements=(supplement,), engine="offline"))
    run_id = ref["run_id"]
    with service.store.acquire(run_id) as token:
        report = service.store.read(run_id)
        report.update(run_status="running", stage="review")
        service.store.save(report, token)
    appendix = report["documents"][1]["doc_id"]
    completed = False

    def status(route):
        route.fulfill(json={
            "run_status": "report_ready" if completed else "running",
            "report_version_id": ref["report_version_id"], "stage": "review",
        })

    page.route(f"**/api/v1/reviews/{run_id}", status)
    page.goto(f"{base_url}/runs/{run_id}")
    docs = page.get_by_label("PDF 文档")
    docs.focus()
    docs.select_option(appendix)
    expect(page.locator(".pdf-heading")).to_contain_text("appendix.pdf")
    expect(docs).to_be_focused()
    page.get_by_label("页图缩放").select_option("200")
    page.get_by_role("link", name="查看版本与导出").click()
    completed = True
    expect(page.get_by_role("link", name="查看最新报告")).to_be_visible()
    expect(docs).to_have_value(appendix)
    expect(page.get_by_label("页图缩放")).to_have_value("200")
    assert page.url.endswith("#report-delivery")
    bounds = page.locator("#pdf-preview").bounding_box()
    assert bounds and bounds["height"] < 720


def test_page_anchor_does_not_reopen_a_claim_sharing_the_selected_evidence(workspace, page):
    base_url, data = workspace
    expect = playwright.expect
    paper = data / "paper.pdf"
    paper.write_bytes(_pdf_bytes())
    service = ReviewService(data / "state.sqlite3")
    ref = service.start(PdfReviewInput(pdf=paper, engine="offline"))
    run_id = ref["run_id"]
    with service.store.acquire(run_id) as token:
        report = service.store.read(run_id)
        anchor = next(item["evidence_id"] for item in report["evidence"] if item["locator"].get("bbox"))
        report["findings"] = [
            {"claim_id": claim_id, "text": f"Claim {claim_id}.", "review_status": "completed",
             "verdict": "supported", "reason": "Evidence available.", "evidence_ids": [anchor]}
            for claim_id in ("first", "second")
        ]
        service.store.save(report, token)
    page.goto(f"{base_url}/runs/{run_id}?evidence_id={anchor}")
    first = page.locator("#finding-first")
    expect(first).to_have_attribute("open", "")
    first.locator("summary").click()
    expect(first).not_to_have_attribute("open", "")
    page.get_by_role("link", name="处理修订").click()
    expect(page).to_have_url(f"{base_url}/runs/{run_id}?evidence_id={anchor}#revisions")
    expect(first).not_to_have_attribute("open", "")


def test_disjoint_provisional_edits_merge_within_one_sentence(workspace, page):
    base_url, data = workspace
    expect = playwright.expect
    paper = data / "paper.pdf"
    paper.write_bytes(_pdf_bytes())
    service = ReviewService(data / "state.sqlite3")
    original = "Error fell at every site, and all sites completed follow-up."
    ref = service.start(PdfReviewInput(pdf=paper, pasted_abstract=original, abstract_source="pasted", engine="offline"))
    run_id, version = ref["run_id"], ref["report_version_id"]
    with service.store.acquire(run_id) as token:
        report = service.store.read(run_id)
        clauses = [("first", "Error fell at every site", "Error fell at some sites"),
                   ("second", "all sites completed follow-up", "most sites completed follow-up")]
        report["findings"] = [
            {"claim_id": key, "text": text, "sentence_id": "s1", "start": original.index(text),
             "end": original.index(text) + len(text), "review_status": "pending", "evidence_ids": []}
            for key, text, _ in clauses
        ]
        report["provisional_revisions"] = {version: {
            key: {"replacement": replacement + ".", "inline_replacement": replacement,
                  "evidence_ids": [], "remaining_checks": []}
            for key, _, replacement in clauses
        }}
        service.store.save(report, token)
    page.goto(f"{base_url}/runs/{run_id}")
    page.get_by_role("link", name="去编辑器应用").first.click()
    buttons = page.get_by_role("button", name="一键应用")
    buttons.nth(0).click()
    buttons.nth(1).click()
    expect(page.get_by_label("确认提交文本")).to_have_value(
        "Error fell at some sites, and most sites completed follow-up."
    )
