import hashlib
import runpy
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from threading import Event

import pymupdf
import pytest
from fastapi.testclient import TestClient

from abstract_review.ingest import parse_pdf
from abstract_review.models import ModelUnavailable
from abstract_review.review import PdfReviewInput, ReviewService
from abstract_review.run_store import RunStore
from abstract_review.web import create_app


SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "d02" / "make_sample.py"


def _pdf_bytes():
    with pymupdf.open() as document:
        page = document.new_page(width=300, height=240)
        for line, y in [
            ("Abstract", 32), ("The study tracks error rate.", 49),
            ("Results", 76), ("Evidence paragraph reports the error rate.", 93),
        ]:
            page.insert_text((30, y), line)
        page.set_rotation(90)
        other = document.new_page(width=300, height=240)
        other.insert_text((30, 65), "Additional results.")
        return document.tobytes()


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path)) as browser:
        yield browser


def wait_for_run(client, run_id, status="report_ready"):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/reviews/{run_id}")
        assert response.status_code == 200, response.text
        state = response.json()
        if state["run_status"] == status:
            return state
        assert state["run_status"] in {"queued", "running"}, state
        time.sleep(0.02)
    pytest.fail(f"review {run_id} did not become {status}")


def test_pdf_upload_report_export_and_version_bound_image(tmp_path, client):
    pdf = _pdf_bytes()
    home = client.get("/")
    assert home.status_code == 200
    assert "导入论文" in home.text

    upload = client.post("/runs", files={"pdf": ("paper.pdf", pdf, "application/pdf")}, follow_redirects=False)
    assert upload.status_code == 303, upload.text
    browser_url = upload.headers["location"]
    run_id = browser_url.rsplit("/", 1)[-1]
    api = f"/api/v1/reviews/{run_id}"
    state = wait_for_run(client, run_id)
    version_id = state["report_version_id"]
    exported = client.get(f"{api}/export", params={"version_id": version_id, "format": "json"})
    assert exported.status_code == 200
    assert exported.headers["content-disposition"].endswith('abstract-review.json"')
    report = exported.json()
    assert report["reviewed_abstract"]["text"] == "The study tracks error rate."
    assert (tmp_path / "blobs" / f"{hashlib.sha256(pdf).hexdigest()}.pdf").is_file()

    evidence = next(item for item in report["evidence"] if item["text"].startswith("Evidence paragraph"))
    evidence_id, doc_id = evidence["evidence_id"], evidence["doc_id"]
    finding = client.get(f"{api}/evidence/{evidence_id}", params={"version_id": version_id})
    assert finding.status_code == 200 and finding.json() == evidence
    browser = client.get(browser_url, params={"evidence_id": evidence_id, "version_id": version_id})
    assert browser.status_code == 200, browser.text
    assert "下载 Markdown" in browser.text and "已高亮所选证据" in browser.text
    fragment = client.get(f"{browser_url}/preview", params={"evidence_id": evidence_id, "version_id": version_id})
    assert fragment.status_code == 200
    assert f'data-selected-evidence="{evidence_id}"' in fragment.text
    assert "data-focus-y=" in fragment.text and "已高亮所选证据" in fragment.text
    assert "<main" not in fragment.text
    markdown = client.get(f"{api}/export", params={"version_id": version_id, "format": "md"})
    assert markdown.status_code == 200
    assert markdown.headers["content-type"].startswith("text/markdown")
    assert "The study tracks error rate." in markdown.text

    page_url = f"/runs/{run_id}/pages/{doc_id}/0.png"
    plain = client.get(page_url, params={"version_id": version_id})
    marked = client.get(page_url, params={"version_id": version_id, "evidence_id": evidence_id})
    assert plain.status_code == marked.status_code == 200
    assert marked.headers["content-type"] == "image/png"
    assert marked.content.startswith(b"\x89PNG\r\n\x1a\n")
    with pymupdf.open(stream=pdf, filetype="pdf") as source:
        page = source[0]
        pixmap = pymupdf.Pixmap(marked.content)
        image_scale = pixmap.width / page.rect.width
        rotated_box = pymupdf.Rect(evidence["locator"]["bbox"]) * page.rotation_matrix * image_scale
        point = (int(rotated_box.x0 + 5), int(rotated_box.y0 + 5))
        assert pixmap.pixel(*point) != pymupdf.Pixmap(plain.content).pixel(*point)

    assert client.get(page_url, params={"version_id": "wrong"}).status_code == 404
    assert client.get(f"/runs/{run_id}/pages/not-in-run/0.png").status_code == 404
    assert client.get(f"/runs/{run_id}/pages/{doc_id}/99.png").status_code == 404
    assert client.get(f"/runs/{run_id}/pages/{doc_id}/1.png", params={"evidence_id": evidence_id}).status_code == 404


def test_version_bound_manual_edit_and_stale_form(client, monkeypatch):
    original = "Original evidence notes <em>the measured error rate</em>."
    result = client.post(
        "/api/v1/reviews", files={"pdf": ("paper.pdf", _pdf_bytes(), "application/pdf")},
        data={"abstract_source": "pasted", "pasted_abstract": original},
    )
    assert result.status_code == 202, result.text
    assert result.json()["run_status"] == "queued"
    run_id = result.json()["run_id"]
    base_version = wait_for_run(client, run_id)["report_version_id"]
    browser = client.get(f"/runs/{run_id}")
    assert "&lt;em&gt;the measured error rate&lt;/em&gt;" in browser.text
    assert "<em>the measured error rate</em>" not in browser.text

    endpoint = f"/api/v1/reviews/{run_id}/selection"
    updated = "The study tracks error rate on its test split."
    resuming, release = Event(), Event()
    initial_resume = ReviewService.resume

    def wait_in_worker(service, *args, **kwargs):
        resuming.set()
        assert release.wait(10)
        return initial_resume(service, *args, **kwargs)

    monkeypatch.setattr(ReviewService, "resume", wait_in_worker)
    submission = {"base_version_id": base_version, "submission_id": "version-one", "edited_text": updated}
    try:
        submitted = client.post(endpoint, data=submission)
        assert submitted.status_code == 202, submitted.text
        assert submitted.json()["run_status"] == "queued"
        assert resuming.wait(5)
        version = submitted.json()["report_version_id"]
        assert version != base_version
        export_url = f"/api/v1/reviews/{run_id}/export"
        selected = client.get(export_url, params={"version_id": version}).json()
        assert selected["reviewed_abstract"]["text"] == updated
        assert selected["abstract_review"]["status"] == "pending"
        assert client.get(export_url, params={"version_id": base_version}).json()["reviewed_abstract"]["text"] == original
        duplicate = client.post(endpoint, data=submission)
        assert duplicate.status_code == 202 and duplicate.json()["report_version_id"] == version
        history = client.get(f"/runs/{run_id}", params={"version_id": base_version})
        assert history.status_code == 200 and "历史摘要" in history.text and "返回最新版本" in history.text

        stale = client.post(endpoint, data={
            "base_version_id": base_version, "submission_id": "version-two", "edited_text": "Stale edit.",
        })
        assert stale.status_code == 409
        rejected = client.post(endpoint, headers={"Origin": "https://other.invalid"}, data={
            "base_version_id": version, "submission_id": "foreign", "edited_text": "Unrelated edit.",
        })
        assert rejected.status_code == 403
    finally:
        release.set()
    assert wait_for_run(client, run_id)["report_version_id"] == version


def test_revised_numeric_check_is_displayed_for_the_requested_version(tmp_path, client):
    source = tmp_path / "numeric.pdf"
    runpy.run_path(str(SAMPLE))["make_paper"](source)
    upload = client.post("/api/v1/reviews", files={"pdf": (source.name, source.read_bytes(), "application/pdf")},
                         data={"abstract_source": "pasted",
                               "pasted_abstract": "On dataset A, accuracy increased by 10 percentage points."})
    assert upload.status_code == 202, upload.text
    run_id = upload.json()["run_id"]
    candidate = wait_for_run(client, run_id)["report_version_id"]
    report_url = f"/runs/{run_id}"
    assert "复算一致" in client.get(report_url, params={"version_id": candidate}).text

    selection = client.post(f"/api/v1/reviews/{run_id}/selection", data={
        "base_version_id": candidate, "submission_id": "numeric-edit",
        "edited_text": "On dataset A, accuracy increased by 12 percentage points.",
    })
    assert selection.status_code == 202, selection.text
    edited = wait_for_run(client, run_id)["report_version_id"]
    assert edited != candidate
    current = client.get(report_url, params={"version_id": edited})
    assert current.status_code == 200 and "复算不一致" in current.text
    assert "复算不一致" not in client.get(report_url, params={"version_id": candidate}).text


def test_intake_returns_accepted_without_waiting_for_pdf_parsing(tmp_path, client, monkeypatch):
    parsing, release = Event(), Event()
    real_start = ReviewService.start

    def wait_in_worker(service, *args, **kwargs):
        parsing.set()
        assert release.wait(10)
        return real_start(service, *args, **kwargs)

    monkeypatch.setattr(ReviewService, "start", wait_in_worker)
    try:
        response = client.post("/api/v1/reviews", files={"pdf": ("paper.pdf", _pdf_bytes(), "application/pdf")})
        assert response.status_code == 202
        state = response.json()
        run_id = state["run_id"]
        assert state == {"run_id": run_id, "report_version_id": None, "run_status": "queued"}
        assert parsing.wait(5)
        assert client.get(response.headers["location"]).json()["run_status"] == "running"
        assert "首份报告" in client.get(f"/runs/{run_id}").text
        assert client.get(f"/api/v1/reviews/{run_id}/export").status_code == 409
        assert (tmp_path / "intake" / run_id / "main" / "paper.pdf").is_file()
    finally:
        release.set()
    assert wait_for_run(client, run_id)["report_version_id"]


@pytest.mark.parametrize("checkpointed", [False, True])
def test_restart_recovers_expired_queue_and_reuses_first_checkpoint(tmp_path, checkpointed):
    app = create_app(tmp_path)
    # Queue a real HTTP upload without a worker by omitting the lifespan context.
    with closing(TestClient(app)) as intake:
        response = intake.post("/api/v1/reviews", files={"pdf": ("paper.pdf", _pdf_bytes(), "application/pdf")})
        assert response.status_code == 202
    run_id = response.json()["run_id"]
    database = tmp_path / "state.sqlite3"
    service = ReviewService(database)
    previous = service.store.claim_next()
    assert previous is not None and previous.run_id == run_id
    if checkpointed:
        root = tmp_path / "intake" / run_id
        paper = root / previous.payload["pdf"]
        service.start(PdfReviewInput(paper, engine="offline"), run_id=run_id, lease_token=previous.token, execute=False)
        # The durable report and original PDF blob are sufficient for resumed work.
        paper.unlink()
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("UPDATE run_leases SET expires=0 WHERE run_id=?", (run_id,))
    with TestClient(create_app(tmp_path)) as restarted:
        result = wait_for_run(restarted, run_id)
        report = restarted.get(f"/api/v1/reviews/{run_id}/export").json()
        assert result["report_version_id"] == report["report_version_id"]
        assert report["reviewed_abstract"]["text"] == "The study tracks error rate."
    assert RunStore(database).claim_next() is None


def test_pause_and_continue_response_remain_nonblocking(client, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ModelUnavailable("No model configured for this test.")

    monkeypatch.setattr(ReviewService, "_responses", unavailable)
    response = client.post("/api/v1/reviews", files={"pdf": ("paper.pdf", _pdf_bytes(), "application/pdf")})
    run_id = response.json()["run_id"]
    paused = wait_for_run(client, run_id, "paused")
    monkeypatch.setattr(ReviewService, "_responses", lambda *args, **kwargs: None)
    resumed = client.post(f"/api/v1/reviews/{run_id}/resume", data={"request_id": "budget", "add_requests": 2})
    assert resumed.status_code == 202 and resumed.json()["run_status"] == "queued"
    assert wait_for_run(client, run_id)["report_version_id"] == paused["report_version_id"]
    report = client.get(f"/api/v1/reviews/{run_id}/export").json()
    assert report["budget_extensions"]["budget"] == [2, 0]


def test_bad_pdf_failure_is_visible_after_accepted_intake(client):
    response = client.post("/api/v1/reviews", files={"pdf": ("paper.pdf", b"not pdf", "application/pdf")})
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    state = wait_for_run(client, run_id, "failed")
    assert state["report_version_id"] is None and "invalid PDF" in state["runtime_error"]
    assert "invalid PDF" in client.get(f"/runs/{run_id}").text


def test_external_evidence_preview_is_limited_to_its_report_version(tmp_path, client):
    created = client.post("/api/v1/reviews", files={"pdf": ("paper.pdf", _pdf_bytes(), "application/pdf")})
    run_id = created.json()["run_id"]
    older = wait_for_run(client, run_id)["report_version_id"]
    api = f"/api/v1/reviews/{run_id}"
    selected = client.post(f"{api}/selection", data={
        "base_version_id": older, "submission_id": "external", "edited_text": "The source measured samples.",
    })
    assert selected.status_code == 202
    latest = wait_for_run(client, run_id)["report_version_id"]

    path = tmp_path / "source.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((40, 56), "The external source recorded 20 samples.")
        path.write_bytes(document.tobytes())
    source = parse_pdf(path)
    service = ReviewService(tmp_path / "state.sqlite3")
    sha256 = source.document["sha256"]
    doc_id = f"external-{sha256}"
    external_doc = {
        **source.document, "doc_id": doc_id, "role": "external", "blob_path": service._store_pdf(source.data, sha256),
    }
    evidence = {**source.evidence[0], "doc_id": doc_id}
    evidence_id = evidence["evidence_id"]
    with service.store.acquire(run_id) as token:
        current = service.store.read(run_id)
        current["external_checks"] = [{
            "attribution": {"text": "The publication measured samples."}, "support_status": "supported",
            "reason": "The source reports its sample count.", "document": external_doc, "evidence": [evidence],
        }]
        service.store.save(current, token)

    evidence_url = f"{api}/evidence/{evidence_id}"
    assert client.get(evidence_url, params={"version_id": latest}).json() == evidence
    assert client.get(evidence_url, params={"version_id": older}).status_code == 404
    preview = f"/runs/{run_id}/pages/{doc_id}/0.png"
    image = client.get(preview, params={"version_id": latest, "evidence_id": evidence_id})
    assert image.status_code == 200 and image.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert client.get(preview, params={"version_id": older}).status_code == 404

    browser = client.get(f"/runs/{run_id}", params={"version_id": latest, "evidence_id": evidence_id})
    assert browser.status_code == 200
    assert "The publication measured samples." in browser.text
    assert "已高亮所选证据" in browser.text
    assert client.get(f"/runs/{run_id}/preview", params={"version_id": older, "evidence_id": evidence_id}).status_code == 404
    assert evidence_id not in {item["evidence_id"] for item in service.export(run_id, latest)["evidence"]}
