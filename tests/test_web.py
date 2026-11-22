import hashlib

import pymupdf
from fastapi.testclient import TestClient

from abstract_review.web import create_app


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


def test_pdf_upload_report_export_and_version_bound_image(tmp_path):
    client = TestClient(create_app(tmp_path))
    pdf = _pdf_bytes()
    home = client.get("/")
    assert home.status_code == 200
    assert "导入论文" in home.text

    upload = client.post("/runs", files={"pdf": ("paper.pdf", pdf, "application/pdf")}, follow_redirects=False)
    assert upload.status_code == 303, upload.text
    browser_url = upload.headers["location"]
    run_id = browser_url.rsplit("/", 1)[-1]
    api = f"/api/v1/reviews/{run_id}"
    state = client.get(api).json()
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


def test_version_bound_manual_edit_and_stale_form(tmp_path):
    client = TestClient(create_app(tmp_path))
    original = "Original evidence notes <em>the measured error rate</em>."
    result = client.post(
        "/api/v1/reviews", files={"pdf": ("paper.pdf", _pdf_bytes(), "application/pdf")},
        data={"abstract_source": "pasted", "pasted_abstract": original},
    )
    assert result.status_code == 202, result.text
    run_id, base_version = result.json()["run_id"], result.json()["report_version_id"]
    browser = client.get(f"/runs/{run_id}")
    assert "&lt;em&gt;the measured error rate&lt;/em&gt;" in browser.text
    assert "<em>the measured error rate</em>" not in browser.text

    endpoint = f"/api/v1/reviews/{run_id}/selection"
    updated = "The study tracks error rate on its test split."
    submitted = client.post(endpoint, data={
        "base_version_id": base_version, "submission_id": "version-one", "edited_text": updated,
    })
    assert submitted.status_code == 200, submitted.text
    version = submitted.json()["report_version_id"]
    assert version != base_version
    export_url = f"/api/v1/reviews/{run_id}/export"
    assert client.get(export_url, params={"version_id": version}).json()["reviewed_abstract"]["text"] == updated
    assert client.get(export_url, params={"version_id": base_version}).json()["reviewed_abstract"]["text"] == original
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
