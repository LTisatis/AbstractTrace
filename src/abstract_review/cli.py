import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid UTF-8 JSON ({exc})") from exc


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def emit(payload: object) -> None:
    print(json.dumps(payload, indent=2))


def load_samples(path: Path):
    from .review import parse_sample

    manifest = read_json(path)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "samples"}
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or not isinstance(manifest["samples"], list)
        or not manifest["samples"]
    ):
        raise ValueError(f"{path}: expected schema_version 1 and a nonempty samples list")

    samples = {}
    for raw_sample in manifest["samples"]:
        sample = parse_sample(raw_sample)
        if sample.sample_id in samples:
            raise ValueError(f"{path}: duplicate sample_id {sample.sample_id!r}")
        samples[sample.sample_id] = sample
    return samples


def review_service(data_dir: Path):
    from .review import ReviewService

    return ReviewService(data_dir / "state.sqlite3")


def doctor(args: argparse.Namespace) -> None:
    args.data_dir.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.data_dir):
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
        finally:
            connection.close()
    result = {
        "status": "ok",
        "mode": "synthetic",
        "python_version": sys.version.split()[0],
        "sqlite_version": sqlite3.sqlite_version,
        "fts5": True,
        "pdf_parser": "pymupdf",
        "data_dir": str(args.data_dir.resolve()),
    }
    if args.online:
        from .models import ModelUnavailable, ResponsesModel, SmokeResult

        try:
            model = ResponsesModel(args.model, max_requests=2, max_seconds=50)
            output = model.call("smoke", {"phrase": "AbstractTrace ready"}, SmokeResult)
        except ModelUnavailable as exc:
            raise ValueError(str(exc)) from exc
        result["online"] = {"model": model.model, "structured_output": output.phrase}
    emit(result)


def start_review(args: argparse.Namespace) -> None:
    if args.pdf is not None or args.source is not None:
        from .review import PdfReviewInput

        if args.sample_id is not None:
            raise ValueError("--sample-id is for --fixture")
        pasted = args.abstract_file.read_bytes().decode("utf-8-sig") if args.abstract_file else None
        pdf = args.pdf
        resolved_source = None
        if args.source:
            from .references import resolve_pdf

            pdf = resolve_pdf(args.source, args.data_dir / "references")
            resolved_source = read_json(pdf.with_suffix(".json"))
        request = PdfReviewInput(
            pdf, tuple(args.supplement), pasted, args.abstract_source, args.engine, args.model,
            embed_model=args.embed_model, check_omissions=args.check_omissions,
            check_external=args.check_external, remote_source=resolved_source, ocr_language=args.ocr_language,
        )
        emit(review_service(args.data_dir).start(request))
        return
    if args.supplement or args.abstract_file or args.abstract_source != "pdf" or args.engine or args.embed_model or args.check_omissions or args.check_external or args.ocr_language:
        raise ValueError("PDF review options require --pdf or --source")
    samples = load_samples(args.fixture)
    sample_id = args.sample_id
    if sample_id is None:
        if len(samples) != 1:
            raise ValueError("--sample-id is required for a fixture with multiple samples")
        sample_id = next(iter(samples))
    if sample_id not in samples:
        raise ValueError(f"unknown sample ID: {sample_id!r}")
    emit(review_service(args.data_dir).start(samples[sample_id]))


def review_status(args: argparse.Namespace) -> None:
    emit(review_service(args.data_dir).status(args.run_id))


def export_review(args: argparse.Namespace) -> None:
    service = review_service(args.data_dir)
    version = args.version_id or service.status(args.run_id)["report_version_id"]
    report = service.export(args.run_id, version)
    if args.format == "md":
        from .reporting import render_markdown

        rendered = render_markdown(report)
        if args.output is None:
            print(rendered, end="")
        else:
            write_text(args.output, rendered)
    elif args.output is None:
        emit(report)
    else:
        write_json(args.output, report)


def resume_review(args: argparse.Namespace) -> None:
    emit(review_service(args.data_dir).resume(args.run_id, args.request_id, args.add_requests, args.add_seconds))


def submit_selection(args: argparse.Namespace) -> None:
    from uuid import uuid4

    text = args.text_file.read_bytes().decode("utf-8-sig") if args.text_file else None
    emit(review_service(args.data_dir).submit_selection(
        args.run_id, args.base_version_id, args.submission_id or uuid4().hex,
        tuple(args.revision_id or []), text,
    ))


def serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .web import create_app

    uvicorn.run(create_app(args.data_dir.resolve()), host="127.0.0.1", port=args.port)


def run_benchmark(args: argparse.Namespace) -> None:
    output_path = args.output_dir / "predictions.json"
    if output_path.resolve() == args.input_manifest.resolve():
        raise ValueError("predictions output must differ from the input manifest")

    service = review_service(args.data_dir)
    if args.engine:
        from .benchmark import run_pdf_samples

        checkpoints = args.output_dir / "sample-checkpoints.json"
        if checkpoints.resolve() == args.input_manifest.resolve():
            raise ValueError("checkpoint output must differ from the input manifest")
        recovered = read_json(checkpoints) if args.resume and checkpoints.is_file() else {}
        if not isinstance(recovered, dict):
            raise ValueError("invalid benchmark checkpoint file")

        def save_sample(sample_id: str, input_digest: str, rows: list[dict]) -> None:
            recovered[sample_id] = {"input_digest": input_digest, "predictions": rows}
            write_json(checkpoints, recovered)

        predictions = run_pdf_samples(service, args.input_manifest, args.engine, args.model,
                                      recovered=recovered, on_sample=save_sample)
    else:
        if args.resume:
            raise ValueError("--resume requires a PDF benchmark engine")
        predictions = []
        for sample in load_samples(args.input_manifest).values():
            run = service.start(sample)
            report = service.export(run["run_id"], run["report_version_id"])
            for finding in report["findings"]:
                predictions.append({
                    "sample_id": sample.sample_id,
                    "claim_id": finding["claim_id"],
                    "review_status": finding["review_status"],
                    "verdict": finding["verdict"],
                    "evidence_ids": finding["evidence_ids"],
                })
    write_json(output_path, {"schema_version": 1, "predictions": predictions})
    emit({"predictions_path": str(output_path), "prediction_count": len(predictions)})


def score_benchmark(args: argparse.Namespace) -> None:
    from .benchmark import summarize_classification
    from .scoring import score_predictions

    source_paths = {args.predictions.resolve(), args.gold_manifest.resolve()}
    if args.output.resolve() in source_paths:
        raise ValueError("score output must differ from both scoring input files")
    predictions, gold = read_json(args.predictions), read_json(args.gold_manifest)
    result = score_predictions(predictions, gold)
    result.update(summarize_classification(predictions, gold))
    write_json(args.output, result)
    emit(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="abstract-review", description="AbstractTrace paper evidence review")
    parser.add_argument(
        "--data-dir", type=Path, default=Path(os.environ.get("ABSTRACT_REVIEW_DATA_DIR") or ".review-data"),
        help="local state directory (default: .review-data or ABSTRACT_REVIEW_DATA_DIR)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    diagnostic = commands.add_parser("doctor", help="check local storage, SQLite FTS5 and optional online model")
    diagnostic.add_argument("--online", action="store_true")
    diagnostic.add_argument("--model", default=os.environ.get("ABSTRACT_REVIEW_MODEL", "gpt-6.2-sol"))
    diagnostic.set_defaults(handler=doctor)
    server = commands.add_parser("serve", help="launch the local report interface on loopback")
    server.add_argument("--port", type=int, default=8000)
    server.set_defaults(handler=serve)

    review = commands.add_parser("review", help="review local PDFs or synthetic fixtures")
    review_commands = review.add_subparsers(dest="review_command", required=True)
    start = review_commands.add_parser("start", help="start a review from a local PDF or synthetic fixture")
    source = start.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", type=Path)
    source.add_argument("--pdf", type=Path)
    source.add_argument("--source", help="resolve a DOI or arXiv identifier to open PDF")
    start.add_argument("--sample-id")
    start.add_argument("--supplement", type=Path, action="append", default=[], help="confirmed supplement PDF")
    start.add_argument("--abstract-file", type=Path, help="UTF-8 alternate abstract")
    start.add_argument("--abstract-source", choices=("pdf", "pasted"), default="pdf")
    start.add_argument("--engine", choices=("offline", "openai"), help="perform a full evidence review after parsing")
    start.add_argument("--model", default=os.environ.get("ABSTRACT_REVIEW_MODEL", "gpt-6.2-sol"))
    start.add_argument("--embed-model", type=Path, help="path to an installed local Sentence Transformers model")
    start.add_argument("--ocr-language", help="Tesseract language (for example eng or eng+chi_sim)")
    start.add_argument("--check-external", action="store_true")
    start.add_argument("--check-omissions", action="store_true")
    start.set_defaults(handler=start_review)

    status = review_commands.add_parser("status", help="read persisted run status")
    status.add_argument("run_id")
    status.set_defaults(handler=review_status)

    export = review_commands.add_parser("export", help="export a persisted version as JSON or Markdown")
    export.add_argument("run_id")
    export.add_argument("--version-id")
    export.add_argument("--format", choices=["json", "md"], default="json")
    export.add_argument("--output", type=Path)
    export.set_defaults(handler=export_review)

    resume = review_commands.add_parser("resume", help="resume a paused full review")
    resume.add_argument("run_id")
    resume.add_argument("--request-id")
    resume.add_argument("--add-requests", type=int, default=0)
    resume.add_argument("--add-seconds", type=int, default=0)
    resume.set_defaults(handler=resume_review)

    select = review_commands.add_parser("submit-selection", help="submit the actual abstract version for rechecking")
    select.add_argument("run_id")
    select.add_argument("--base-version-id", required=True)
    select.add_argument("--submission-id")
    edits = select.add_mutually_exclusive_group(required=True)
    edits.add_argument("--revision-id", action="append")
    edits.add_argument("--text-file", type=Path)
    edits.add_argument("--use-original", action="store_true")
    select.set_defaults(handler=submit_selection)

    bench = commands.add_parser("bench", help="produce predictions or score them against separate gold")
    bench_commands = bench.add_subparsers(dest="bench_command", required=True)
    run = bench_commands.add_parser("run", help="predict from an input-only manifest")
    run.add_argument("--input-manifest", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--engine", choices=("offline", "openai"), help="review a local PDF benchmark manifest")
    run.add_argument("--model", default=os.environ.get("ABSTRACT_REVIEW_MODEL", "gpt-6.2-sol"))
    run.add_argument("--resume", action="store_true", help="reuse matching persisted PDF sample checkpoints")
    run.set_defaults(handler=run_benchmark)

    score = bench_commands.add_parser("score", help="score predictions independently of review")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--gold-manifest", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(handler=score_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"abstract-review: {exc}", file=sys.stderr)
        return 1
    return 0
