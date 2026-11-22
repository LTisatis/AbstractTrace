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


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
    emit({
        "status": "ok",
        "mode": "synthetic",
        "python_version": sys.version.split()[0],
        "sqlite_version": sqlite3.sqlite_version,
        "fts5": True,
        "pdf_parser": "pymupdf",
        "data_dir": str(args.data_dir.resolve()),
    })


def start_review(args: argparse.Namespace) -> None:
    if args.pdf is not None:
        from .review import PdfReviewInput

        if args.sample_id is not None:
            raise ValueError("--sample-id is for --fixture")
        pasted = args.abstract_file.read_bytes().decode("utf-8-sig") if args.abstract_file else None
        request = PdfReviewInput(args.pdf, tuple(args.supplement), pasted, args.abstract_source)
        emit(review_service(args.data_dir).start(request))
        return
    if args.supplement or args.abstract_file or args.abstract_source != "pdf":
        raise ValueError("supplements and abstract options require --pdf")
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
    report = review_service(args.data_dir).export(args.run_id, args.version_id)
    if args.output is None:
        emit(report)
    else:
        write_json(args.output, report)


def run_benchmark(args: argparse.Namespace) -> None:
    samples = load_samples(args.input_manifest)
    output_path = args.output_dir / "predictions.json"
    if output_path.resolve() == args.input_manifest.resolve():
        raise ValueError("predictions output must differ from the input manifest")

    service = review_service(args.data_dir)
    predictions = []
    for sample in samples.values():
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
    from .scoring import score_predictions

    source_paths = {args.predictions.resolve(), args.gold_manifest.resolve()}
    if args.output.resolve() in source_paths:
        raise ValueError("score output must differ from both scoring input files")
    result = score_predictions(read_json(args.predictions), read_json(args.gold_manifest))
    write_json(args.output, result)
    emit(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="abstract-review", description="AbstractTrace paper evidence review")
    parser.add_argument(
        "--data-dir", type=Path, default=Path(os.environ.get("ABSTRACT_REVIEW_DATA_DIR") or ".review-data"),
        help="local state directory (default: .review-data or ABSTRACT_REVIEW_DATA_DIR)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="check Python, writable local storage and SQLite FTS5").set_defaults(
        handler=doctor
    )

    review = commands.add_parser("review", help="review local PDFs or synthetic fixtures")
    review_commands = review.add_subparsers(dest="review_command", required=True)
    start = review_commands.add_parser("start", help="start a review from a local PDF or synthetic fixture")
    source = start.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", type=Path)
    source.add_argument("--pdf", type=Path)
    start.add_argument("--sample-id")
    start.add_argument("--supplement", type=Path, action="append", default=[], help="confirmed supplement PDF")
    start.add_argument("--abstract-file", type=Path, help="UTF-8 alternate abstract")
    start.add_argument("--abstract-source", choices=("pdf", "pasted"), default="pdf")
    start.set_defaults(handler=start_review)

    status = review_commands.add_parser("status", help="read persisted run status")
    status.add_argument("run_id")
    status.set_defaults(handler=review_status)

    export = review_commands.add_parser("export", help="export a persisted JSON report")
    export.add_argument("run_id")
    export.add_argument("--version-id", required=True)
    export.add_argument("--format", choices=["json"], default="json")
    export.add_argument("--output", type=Path)
    export.set_defaults(handler=export_review)

    bench = commands.add_parser("bench", help="produce predictions or score them against separate gold")
    bench_commands = bench.add_subparsers(dest="bench_command", required=True)
    run = bench_commands.add_parser("run", help="predict from an input-only manifest")
    run.add_argument("--input-manifest", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
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
