import hashlib
import runpy
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pymupdf
import pytest
from fastapi.testclient import TestClient

from abstract_review.ingest import parse_pdf
from abstract_review.models import ModelUnavailable
from abstract_review.review import PdfReviewInput, ReviewService
from abstract_review.revisions import provisional_editor_options
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


def test_provisional_wording_keeps_the_pending_verdict(tmp_path, client, monkeypatch):
    upload = client.post("/api/v1/reviews", files={"pdf": ("paper.pdf", _pdf_bytes(), "application/pdf")})
    run_id = upload.json()["run_id"]
    version = wait_for_run(client, run_id)["report_version_id"]
    store = RunStore(tmp_path / "state.sqlite3")
    report = store.read(run_id)
    evidence_id = next(item["evidence_id"] for item in report["evidence"] if item["text"].startswith("Evidence paragraph"))
    report["configuration"]["engine"] = "openai"
    report["findings"] = [{"claim_id": "pending-claim", "text": report["original_abstract"]["text"], "sentence_id": "s1",
                           "verdict": None, "review_status": "pending", "evidence_ids": [evidence_id],
                           "reason": "The results excerpt does not cover all sites."}]
    store.save(report)

    calls = []

    def responses(self, report, *, time_limit):
        assert time_limit == 65

        def call(stage, material, schema):
            assert stage == "provisional_revision"
            assert [row["evidence_id"] for row in material["evidence"]] == ["E1"]
            calls.append(stage)
            return schema(replacement="The study reports the error rate.", reason="The source reports the measure.",
                          remaining_checks=["Verify the direction of change for each site."], evidence_ids=["E1"])

        return SimpleNamespace(call=call, events=[{"stage": "provisional_revision"}], elapsed_seconds=1)

    monkeypatch.setattr(ReviewService, "_responses", responses)
    report_url = f"/runs/{run_id}"
    endpoint = report_url + "/provisional-revision"
    headers = {"X-AbstractTrace-Fragment": "provisional"}
    payload = {"version_id": version, "claim_id": "pending-claim"}
    assert 'data-provisional-form' in client.get(report_url).text
    assert client.post(endpoint, data={**payload, "version_id": "stale"}, headers=headers).status_code == 409
    result = client.post(endpoint, data=payload, headers=headers)
    assert result.status_code == 200 and "The study reports the error rate." in result.text
    assert "Verify the direction of change for each site." in result.text
    assert evidence_id in result.text and "暂定修订 · 待核验" in result.text
    assert 'data-provisional-apply' in result.text
    assert 'data-replacement="The study reports the error rate."' in client.get(report_url).text
    assert client.post(endpoint, data=payload, headers=headers).status_code == 200
    assert calls == ["provisional_revision"]
    redirect = client.post(endpoint, data=payload, follow_redirects=False)
    assert redirect.status_code == 303
    assert 'id="finding-pending-claim" data-verdict="pending" open' in client.get(redirect.headers["location"]).text
    stored = store.read(run_id)
    assert stored["report_version_id"] == version
    assert stored["reviewed_abstract"] == report["reviewed_abstract"]
    assert stored["findings"] == report["findings"] and stored["revisions"] == report["revisions"]


def test_provisional_application_preserves_other_clauses():
    sentence = "Error fell at every site, and all sites completed follow-up."
    claim = "Error fell at every site"
    proposal = {"replacement": "Error fell at some sites.", "inline_replacement": "Error fell at some sites"}
    report = {"report_version_id": "v1", "original_abstract": {"text": sentence},
              "provisional_revisions": {"v1": {"claim": proposal}},
              "findings": [{"claim_id": "claim", "sentence_id": "s1", "text": claim, "verdict": "pending",
                            "start": 0, "end": len(claim)}]}
    [option] = provisional_editor_options(report)
    assert option["source"] == sentence
    assert option["replacement"] == "Error fell at some sites, and all sites completed follow-up."
    proposal.pop("inline_replacement")
    [legacy] = provisional_editor_options(report)
    assert legacy["replacement"] is None


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
    assert 'data-pdf-nav="next"' in fragment.text
    assert 'data-pdf-nav="previous"' not in fragment.text

    second_page = {"doc_id": doc_id, "page": 2, "version_id": version_id, "evidence_id": evidence_id}
    for path in (browser_url, f"{browser_url}/preview"):
        browsed = client.get(path, params=second_page)
        assert browsed.status_code == 200
        assert 'data-page-number="2"' in browsed.text
        assert 'data-pdf-nav="previous"' in browsed.text
        assert 'data-pdf-nav="next"' not in browsed.text
        assert 'data-pdf-nav="evidence"' in browsed.text
        assert f'/pages/{doc_id}/1.png?version_id={version_id}"' in browsed.text
        assert "已高亮所选证据" not in browsed.text
        assert client.get(path, params={**second_page, "page": 3}).status_code == 404
        assert client.get(path, params={**second_page, "doc_id": "not-in-version"}).status_code == 404

    returned = client.get(f"{browser_url}/preview", params={**second_page, "page": 1})
    assert 'data-page-number="1"' in returned.text
    assert "已高亮所选证据" in returned.text
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
        citation = current["evidence"][0]["evidence_id"]
        current["external_checks"] = [{
            "kind": "explicit_citation", "reference": {"provider": "doi", "identifier": "10.1234/example"},
            "citation_status": "exists", "access_status": "accessible",
            "attribution": {"text": "The publication measured samples."}, "support_status": "supported",
            "reason": "The source reports its sample count.", "document": external_doc,
            "citation_evidence_ids": [citation], "evidence_ids": [evidence_id],
            "evidence": [evidence, {**evidence, "evidence_id": "uncited", "text": "Unadopted candidate"}],
        }, {
            "kind": "novelty_search", "support_status": "not_checked", "attribution": {"text": "First method"},
            "reason": "Metadata is not a priority decision.", "query": "Novelty check search query",
            "candidates": [{"id": "https://openalex.org/W123", "display_name": "Candidate prior work",
                            "publication_date": "2026-01-01"}],
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
    assert "主论文中的引用位置" in browser.text
    assert "裁决采用的外部证据" in browser.text
    assert "其他候选段落（1）" in browser.text
    assert "Unadopted candidate" in browser.text
    assert "Novelty check search query" in browser.text
    assert "Candidate prior work" in browser.text
    assert "https://openalex.org/W123" in browser.text
    assert "已高亮所选证据" in browser.text
    assert client.get(f"/runs/{run_id}/preview", params={"version_id": older, "evidence_id": evidence_id}).status_code == 404
    assert evidence_id not in {item["evidence_id"] for item in service.export(run_id, latest)["evidence"]}


def test_original_evidence_is_paginated_and_searchable(tmp_path, client):
    response = client.post("/api/v1/reviews", files={"pdf": ("paper.pdf", _pdf_bytes(), "application/pdf")})
    run_id = response.json()["run_id"]
    version = wait_for_run(client, run_id)["report_version_id"]
    service = ReviewService(tmp_path / "state.sqlite3")
    with service.store.acquire(run_id) as token:
        report = service.store.read(run_id)
        template = report["evidence"][0]
        for number in range(65):
            report["evidence"].append({**template, "kind": "table_note", "evidence_id": f"fixture-{number}",
                                       "text": f"Pagination fixture row {number:02d}."})
        service.store.save(report, token)
    report_url = f"/runs/{run_id}"
    initial = client.get(report_url)
    assert initial.text.count('class="evidence-row"') == 30
    assert "Pagination fixture row 64." not in initial.text
    fragment = client.get(f"{report_url}/evidence-list", params={
        "version_id": version, "evidence_search": "Row 64", "evidence_kind": "table",
    })
    assert fragment.status_code == 200 and fragment.text.count('class="evidence-row"') == 1
    assert "Pagination fixture row 64." in fragment.text
    assert client.get(f"{report_url}/evidence-list", params={"version_id": "missing"}).status_code == 404


def test_recent_runs_includes_ready_and_failed_intake(client):
    assert "还没有本地审阅记录" in client.get("/runs").text
    valid = client.post("/api/v1/reviews", files={"pdf": ("paper.pdf", _pdf_bytes(), "application/pdf")})
    failed = client.post("/api/v1/reviews", files={"pdf": ("invalid.pdf", b"invalid", "application/pdf")})
    wait_for_run(client, valid.json()["run_id"])
    wait_for_run(client, failed.json()["run_id"], "failed")
    history = client.get("/runs").text
    assert "paper.pdf" in history and "下载 Markdown" in history
    assert "invalid.pdf" in history and "未完成" in history
