"""Local HTML review workspace and version-scoped evidence views."""

import json
import re
import shutil
import sqlite3
from collections import Counter
from contextlib import asynccontextmanager
from math import isfinite
from pathlib import Path, PureWindowsPath
from threading import Event
from typing import Annotated
from uuid import uuid4

import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from jinja2 import Environment, PackageLoader, select_autoescape

from .ingest import MAX_PDF_BYTES
from .reporting import display_documents, display_evidence, evidence_path_tree, render_markdown
from .review import ReviewService
from .run_store import RunBusy
from .worker import LocalWorker


_TEMPLATES = Environment(loader=PackageLoader("abstract_review"), autoescape=select_autoescape())
_SHA256 = re.compile(r"[0-9a-f]{64}")
_STATUS_TEXT = {
    "completed": "已裁决", "pending": "待核验", "supported": "支持", "refuted": "反驳",
    "insufficient": "证据不足", "manual_conflict": "证据冲突", "manual_review": "需人工复核",
    "passed": "已通过", "returned": "已退回", "report_ready": "报告已就绪",
    "queued": "排队中", "paused": "已暂停", "failed": "未完成", "running": "正在处理",
    "not_checked": "未核查", "not_supported": "未获支持", "unclear": "尚无定论",
    "unverified": "未核实", "unavailable": "不可访问", "unreadable": "无法提取",
}


def _html(request: Request, *, status_code: int = 200, template_name: str = "review.html", **context) -> HTMLResponse:
    template = _TEMPLATES.get_template(template_name)
    html = template.render(request=request, statuses=_STATUS_TEXT, token=uuid4().hex, **context)
    return HTMLResponse(html, status_code=status_code, headers={"Cache-Control": "no-store"})


def _status(service: ReviewService, run_id: str) -> dict:
    try:
        return service.store.status(run_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


def _snapshot(service: ReviewService, run_id: str, version_id: str | None) -> tuple[dict, dict]:
    state = _status(service, run_id)
    if state["report_version_id"] is None:
        raise HTTPException(409, "当前任务还没有可查看的报告版本。")
    try:
        return state, service.export(run_id, version_id or state["report_version_id"])
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


def _block_foreign_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin is not None and origin != str(request.base_url).rstrip("/"):
        raise HTTPException(403, "Review changes require a request from this workspace.")


def _upload_pdf(upload: UploadFile, directory: Path) -> Path:
    name = PureWindowsPath(upload.filename or "").name
    if not name.lower().endswith(".pdf") or len(name) > 140:
        raise ValueError("请上传扩展名为 .pdf 的文件。")
    directory.mkdir()
    path = directory / name
    if path.parent != directory or not name:
        raise ValueError("无效的 PDF 文件名。")
    total = 0
    with path.open("xb") as target:
        while block := upload.file.read(1024 * 1024):
            total += len(block)
            if total > MAX_PDF_BYTES:
                raise ValueError(f"{name} 超过单个 PDF 的大小上限。")
            target.write(block)
    return path


def _blob_file(data_dir: Path, document: dict) -> Path:
    checksum = document.get("sha256")
    if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
        raise HTTPException(404, "PDF archive not found")
    relative_path = Path("blobs", f"{checksum}.pdf")
    if document.get("blob_path") != relative_path.as_posix():
        raise HTTPException(404, "PDF archive not found")
    root = data_dir.resolve()
    archive = (root / relative_path).resolve()
    if archive.parent != root / "blobs" or not archive.is_file():
        raise HTTPException(404, "PDF archive not found")
    return archive


def _box(locator: dict) -> pymupdf.Rect | None:
    coords = locator.get("bbox")
    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
        return None
    if not all(type(value) in (int, float) and isfinite(value) for value in coords):
        return None
    rect = pymupdf.Rect(coords)
    return None if rect.is_empty else rect


def _marks(selected: dict, page_evidence: list[dict]) -> list[tuple[pymupdf.Rect, bool]]:
    locator = selected["locator"]
    marks = []
    table, row, column = (locator.get(key) for key in ("table_index", "row_index", "column_index"))
    if selected.get("kind") == "table_cell" and all(type(value) is int for value in (table, row, column)):
        related_ids = {selected.get("row_header_evidence_id"), *selected.get("column_header_evidence_ids", ()),
                       *selected.get("note_evidence_ids", ())} - {None, selected["evidence_id"]}
        legacy_headers = {(0, column), (row, 0)} - {(row, column)} if "cell_role" not in selected else set()
        for item in page_evidence:
            loc = item["locator"]
            position = loc.get("row_index"), loc.get("column_index")
            linked = item["evidence_id"] in related_ids
            legacy = item.get("kind") == "table_cell" and position in legacy_headers
            if loc.get("table_index") == table and (linked or legacy):
                if rect := _box(loc):
                    marks.append((rect, False))
    if rect := _box(locator):
        marks.append((rect, True))
    return marks


def _render_page(archive: Path, index: int, marks: list[tuple[pymupdf.Rect, bool]]) -> bytes:
    try:
        with pymupdf.open(archive) as document:
            if not document.is_pdf or document.needs_pass or not 0 <= index < document.page_count:
                raise HTTPException(404, "PDF page not found")
            page = document[index]
            if page.rect.is_empty or page.rect.is_infinite:
                raise HTTPException(422, "PDF page cannot be rendered")
            visible = page.rect * page.derotation_matrix
            for mark, active in marks:
                mark &= visible
                if not mark.is_empty:
                    color = (0.08, 0.50, 0.33) if active else (0.15, 0.32, 0.57)
                    page.draw_rect(
                        mark, color=color, fill=color, width=1.4,
                        stroke_opacity=0.9, fill_opacity=0.22 if active else 0.13,
                    )
            scale = min(2, 1700 / max(page.rect.width, page.rect.height))
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False, annots=False)
            return pixmap.tobytes("png")
    except (pymupdf.mupdf.FzErrorBase, RuntimeError, ValueError) as exc:
        raise HTTPException(422, "PDF page cannot be rendered") from exc


def _selected_evidence(report: dict, evidence_id: str | None) -> dict | None:
    if evidence_id is None:
        return None
    for evidence in display_evidence(report):
        if evidence["evidence_id"] == evidence_id:
            return evidence
    raise HTTPException(404, "Evidence not found in this report version")


def _viewer(
    report: dict, selected: dict | None, doc_id: str | None, page: int | None,
) -> tuple[dict, int, str | None] | None:
    target_doc = doc_id
    if target_doc is None and selected:
        target_doc = selected.get("doc_id")
    document = next((item for item in display_documents(report) if item.get("pages")
                     and (target_doc is None or item["doc_id"] == target_doc)), None)
    locator = selected["locator"] if selected and selected.get("doc_id") == target_doc else None
    anchor = locator["page_index"] if locator and _box(locator) else None
    if document is None or (selected is not None and anchor is None and doc_id is None and page is None):
        if page is not None or doc_id is not None:
            raise HTTPException(404, "PDF document not found in this report version")
        return None
    indices = {row["page_index"] for row in document["pages"]}
    index = anchor if anchor is not None else min(indices)
    if page is not None:
        index = page - 1
    if index not in indices:
        raise HTTPException(404, "PDF page not found in this report version")
    return document, index, selected["evidence_id"] if selected and anchor == index else None


def _preview_focus(document: dict, index: int, selected: dict) -> tuple[float, float] | None:
    page = next((item for item in document.get("pages", ()) if item["page_index"] == index), None)
    if page is None or not page.get("width") or not page.get("height"):
        return None
    box = _box(selected["locator"])
    if box is None:
        return None
    x = min(1.0, max(0.0, (box.x0 + box.x1) / (2 * page["width"])))
    y = min(1.0, max(0.0, (box.y0 + box.y1) / (2 * page["height"])))
    return {90: (1 - y, x), 180: (1 - x, 1 - y), 270: (y, 1 - x)}.get(page.get("rotation"), (x, y))


def create_app(data_dir: Path) -> FastAPI:
    """Expose the PDF review workflow on the local web server."""
    data_dir = Path(data_dir).resolve()
    service = ReviewService(data_dir / "state.sqlite3")
    wake = Event()

    @asynccontextmanager
    async def lifespan(_app):
        worker = LocalWorker(service, data_dir, wake)
        worker.start()
        try:
            yield
        finally:
            worker.stop()

    app = FastAPI(title="AbstractTrace", lifespan=lifespan)

    def report_page(
        request: Request, run_id: str, version_id: str | None = None,
        evidence_id: str | None = None, *, doc_id: str | None = None, page: int | None = None,
        error: str | None = None, status_code: int = 200, preview_only: bool = False,
    ) -> HTMLResponse:
        intake = _status(service, run_id)
        if not preview_only and intake["report_version_id"] is None and version_id is None and evidence_id is None:
            return _html(request, state=intake, error=error, status_code=status_code)
        state, report = _snapshot(service, run_id, version_id)
        selected = _selected_evidence(report, evidence_id)
        version = report["report_version_id"]
        preview_url = None
        preview_document = None
        preview_page = None
        preview_focus = None
        preview_evidence = None
        page_numbers = []
        page_position = 0
        if preview := _viewer(report, selected, doc_id, page):
            preview_document, preview_page, preview_evidence = preview
            page_numbers = sorted({item["page_index"] + 1 for item in preview_document["pages"]})
            page_position = page_numbers.index(preview_page + 1)
            page_args = {"version_id": version}
            if preview_evidence is not None:
                page_args["evidence_id"] = preview_evidence
                preview_focus = _preview_focus(preview_document, preview_page, selected)
            preview_url = request.url_for(
                "review_page_image", run_id=run_id, doc_id=preview_document["doc_id"], page_index=preview_page,
            ).include_query_params(**page_args)
        evidence_by_id = {item["evidence_id"]: item for item in display_evidence(report)}
        preview_context = dict(
            run_id=run_id, version_id=version, selected=selected,
            evidence_by_id=evidence_by_id, preview_url=preview_url, preview_document=preview_document,
            preview_page=preview_page, preview_focus=preview_focus, preview_evidence=preview_evidence,
            page_numbers=page_numbers, page_position=page_position, requested_doc=doc_id, requested_page=page,
        )
        if preview_only:
            return _html(request, template_name="review_preview.html", **preview_context)
        from .revisions import provisional_editor_options

        evidence_graphs = {
            finding["claim_id"]: evidence_path_tree(paths, evidence_by_id)
            for finding in report.get("findings", ()) if (paths := finding.get("retrieval_paths"))
        }
        finding_counts = Counter(
            row.get("verdict") or row.get("review_status")
            for kind in ("compound_findings", "findings") for row in report.get(kind, ())
        )
        return _html(
            request, status_code=status_code, error=error, report=report, state=state,
            is_current=version == state["report_version_id"], evidence_graphs=evidence_graphs,
            finding_counts=finding_counts,
            provisional_options=provisional_editor_options(report),
            preview_endpoint=request.url_for("evidence_preview", run_id=run_id),
            **preview_context,
        )

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return _html(request)

    @app.post("/runs", name="start_review")
    @app.post("/api/v1/reviews", name="api_start_review", status_code=202)
    def start_review(
        request: Request, pdf: Annotated[UploadFile, File()],
        supplements: Annotated[list[UploadFile] | None, File()] = None,
        pasted_abstract: Annotated[str | None, Form()] = None,
        abstract_source: Annotated[str, Form()] = "pdf",
        engine: Annotated[str, Form()] = "offline",
        check_omissions: Annotated[bool, Form()] = False,
        check_external: Annotated[bool, Form()] = False,
    ) -> Response:
        _block_foreign_origin(request)
        is_html = request.url.path == "/runs"
        run_id = uuid4().hex
        root = data_dir / "intake" / run_id
        try:
            if abstract_source not in {"pdf", "pasted"}:
                raise ValueError("请选择有效的待审摘要来源。")
            if engine not in {"offline", "openai"}:
                raise ValueError("请选择可用的审阅方式。")
            if abstract_source == "pasted" and not (pasted_abstract or "").strip():
                raise ValueError("请填写待审摘要。")
            root.mkdir(parents=True)
            paper = _upload_pdf(pdf, root / "main")
            attachments = [
                _upload_pdf(upload, root / f"supplement-{i}")
                for i, upload in enumerate(supplements or []) if upload.filename
            ]
            state = service.store.queue_start(run_id, {
                "pdf": paper.relative_to(root).as_posix(),
                "supplements": [attachment.relative_to(root).as_posix() for attachment in attachments],
                "pasted_abstract": pasted_abstract or None, "abstract_source": abstract_source,
                "engine": engine, "check_omissions": check_omissions, "check_external": check_external,
            })
        except (ValueError, OSError, sqlite3.Error) as exc:
            shutil.rmtree(root, ignore_errors=True)
            if is_html:
                return _html(request, error=str(exc), status_code=400)
            raise HTTPException(400, str(exc)) from exc
        wake.set()
        if is_html:
            return RedirectResponse(request.url_for("view_run", run_id=state["run_id"]), status_code=303)
        return JSONResponse(state, status_code=202, headers={
            "Location": str(request.url_for("run_status", run_id=run_id)),
        })

    @app.get("/api/v1/reviews/{run_id}")
    def run_status(run_id: str) -> dict:
        return _status(service, run_id)

    @app.get("/runs/{run_id}", response_class=HTMLResponse, name="view_run")
    def view_run(request: Request, run_id: str, version_id: str | None = None, evidence_id: str | None = None,
                 doc_id: str | None = None, page: int | None = None):
        return report_page(request, run_id, version_id, evidence_id, doc_id=doc_id, page=page)

    @app.get("/runs/{run_id}/preview", response_class=HTMLResponse, name="evidence_preview")
    def evidence_preview(request: Request, run_id: str, version_id: str | None = None, evidence_id: str | None = None,
                         doc_id: str | None = None, page: int | None = None):
        return report_page(request, run_id, version_id, evidence_id, doc_id=doc_id, page=page, preview_only=True)

    @app.get("/api/v1/reviews/{run_id}/evidence/{evidence_id}")
    def review_evidence(run_id: str, evidence_id: str, version_id: str | None = None) -> dict:
        _, report = _snapshot(service, run_id, version_id)
        return _selected_evidence(report, evidence_id)

    @app.get("/api/v1/reviews/{run_id}/export", name="export_review")
    def export_review(run_id: str, version_id: str | None = None, format: str = "json") -> Response:
        if format not in {"json", "md"}:
            raise HTTPException(400, "Export format must be json or md")
        _, report = _snapshot(service, run_id, version_id)
        if format == "md":
            text, media_type = render_markdown(report), "text/markdown"
        else:
            text, media_type = json.dumps(report, ensure_ascii=False, indent=2), "application/json"
        return Response(text, media_type=media_type, headers={
            "Content-Disposition": f'attachment; filename="abstract-review.{format}"',
            "Cache-Control": "no-store",
        })

    @app.get("/runs/{run_id}/pages/{doc_id}/{page_index}.png", name="review_page_image")
    def review_page_image(
        run_id: str, doc_id: str, page_index: int, version_id: str | None = None, evidence_id: str | None = None,
    ) -> Response:
        _, report = _snapshot(service, run_id, version_id)
        document = next((doc for doc in display_documents(report) if doc["doc_id"] == doc_id), None)
        if document is None or not any(p["page_index"] == page_index for p in document.get("pages", [])):
            raise HTTPException(404, "PDF page not found in this report version")
        archive = _blob_file(data_dir, document)
        selected = _selected_evidence(report, evidence_id)
        marks = []
        if selected:
            if selected.get("doc_id") != doc_id or selected["locator"]["page_index"] != page_index:
                raise HTTPException(404, "Evidence anchor does not belong to this page")
            page_evidence = [
                e for e in display_evidence(report)
                if e.get("doc_id") == doc_id and e["locator"]["page_index"] == page_index
            ]
            marks = _marks(selected, page_evidence)
        return Response(_render_page(archive, page_index, marks), media_type="image/png", headers={
            "Cache-Control": "no-store",
        })

    @app.post("/runs/{run_id}/selection", name="select_revision")
    @app.post("/api/v1/reviews/{run_id}/selection", name="api_select_revision", status_code=202)
    def select_revision(
        request: Request, run_id: str, base_version_id: Annotated[str, Form()],
        submission_id: Annotated[str, Form()], revision_ids: Annotated[list[str] | None, Form()] = None,
        edited_text: Annotated[str | None, Form()] = None,
    ) -> Response:
        _block_foreign_origin(request)
        is_html = request.url.path.startswith("/runs/")
        current = _status(service, run_id)
        try:
            if current["report_version_id"] is None:
                raise RunBusy("请等待解析出首份报告后再提交修订。")
            state = service.submit_selection(
                run_id, base_version_id, submission_id, revision_ids=tuple(revision_ids or ()),
                edited_text=edited_text, enqueue=True,
            )
        except ValueError as exc:
            stale = _status(service, run_id)["report_version_id"] != base_version_id
            code = 409 if stale or isinstance(exc, RunBusy) else 400
            if is_html:
                return report_page(request, run_id, error=str(exc), status_code=code)
            raise HTTPException(code, str(exc)) from exc
        wake.set()
        if is_html:
            redirect = request.url_for("view_run", run_id=run_id).include_query_params(
                version_id=state["report_version_id"],
            )
            return RedirectResponse(redirect, status_code=303)
        return JSONResponse(state, status_code=202, headers={
            "Location": str(request.url_for("run_status", run_id=run_id)),
        })

    @app.post("/runs/{run_id}/provisional-revision", name="provisional_revision")
    def provisional_revision(
        request: Request, run_id: str, version_id: Annotated[str, Form()], claim_id: Annotated[str, Form()],
    ) -> Response:
        from .models import ModelUnavailable
        from .revisions import generate_provisional, provisional_editor_options

        _block_foreign_origin(request)
        _status(service, run_id)
        fragment = request.headers.get("x-abstracttrace-fragment") == "provisional"
        try:
            proposal = generate_provisional(service, run_id, version_id, claim_id)
        except (ValueError, ModelUnavailable) as exc:
            if isinstance(exc, ModelUnavailable):
                message, code = "暂定修订未能生成。请检查模型连接和用量，或稍后重试。", 502
            elif isinstance(exc, RunBusy):
                message, code = "报告正在处理或已更新。请打开最新报告后重试。", 409
            else:
                message, code = str(exc), 400
            if fragment:
                raise HTTPException(code, message) from exc
            return report_page(request, run_id, error=message, status_code=code)

        if fragment:
            report = service.export(run_id, version_id)
            return _html(
                request, template_name="review_provisional.html", run_id=run_id, version_id=version_id,
                finding_id=claim_id, proposal=proposal,
                refresh_provisional_options=True, provisional_options=provisional_editor_options(report),
                evidence_by_id={item["evidence_id"]: item for item in display_evidence(report)},
            )
        url = request.url_for("view_run", run_id=run_id).include_query_params(version_id=version_id, finding=claim_id)
        return RedirectResponse(url.replace(fragment=f"finding-{claim_id}"), status_code=303)

    @app.post("/runs/{run_id}/resume", name="resume_review")
    @app.post("/api/v1/reviews/{run_id}/resume", name="api_resume_review", status_code=202)
    def resume_review(
        request: Request, run_id: str, request_id: Annotated[str | None, Form()] = None,
        add_requests: Annotated[int, Form(ge=0)] = 0, add_seconds: Annotated[int, Form(ge=0)] = 0,
    ) -> Response:
        _block_foreign_origin(request)
        _status(service, run_id)
        try:
            state = service.store.queue_resume(run_id, request_id, add_requests, add_seconds)
        except ValueError as exc:
            code = 409 if isinstance(exc, RunBusy) else 400
            if request.url.path.startswith("/runs/"):
                return report_page(request, run_id, error=str(exc), status_code=code)
            raise HTTPException(code, str(exc)) from exc
        wake.set()
        if request.url.path.startswith("/runs/"):
            return RedirectResponse(request.url_for("view_run", run_id=run_id), status_code=303)
        return JSONResponse(state, status_code=202, headers={
            "Location": str(request.url_for("run_status", run_id=run_id)),
        })

    return app
