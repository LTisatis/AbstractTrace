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
        from .models import ModelUnavailable, online_diagnostics

        try:
            result["online"] = online_diagnostics(args.model)
        except ModelUnavailable as exc:
            raise ValueError(str(exc)) from exc
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
            embed_model=args.embed_model, rerank_model=args.rerank_model, check_omissions=args.check_omissions,
            check_external=args.check_external, remote_source=resolved_source, ocr_language=args.ocr_language,
        )
        emit(review_service(args.data_dir).start(request))
        return
    if (args.supplement or args.abstract_file or args.abstract_source != "pdf" or args.engine or args.embed_model
            or args.rerank_model or args.check_omissions or args.check_external or args.ocr_language):
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
    import hashlib
    import time

    from .scoring import semantic_manifest_sha256

    started = time.perf_counter()
    results = []
    output_path = args.output_dir / "predictions.json"
    record_path = args.output_dir / "run.json"
    online_dataset = args.engine == "openai" and args.dataset in {"scifact", "scitabalign"}
    checkpoints = args.output_dir / ("sample-checkpoints.sqlite3" if online_dataset else "sample-checkpoints.json")
    files = {output_path.resolve(), record_path.resolve(), checkpoints.resolve(), args.input_manifest.resolve()}
    if len(files) != 4:
        raise ValueError("benchmark outputs and input manifest must use distinct files")
    manifest_bytes = args.input_manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{args.input_manifest}: invalid UTF-8 JSON ({exc})") from exc
    retrieval_options = {"embed_model": args.embed_model, "rerank_model": args.rerank_model}
    if any(retrieval_options.values()) and (args.dataset or not args.engine):
        raise ValueError("local retrieval models require a PDF benchmark engine without --dataset")
    batch_limits = {key: value for key, value in (("max_requests", args.max_requests), ("max_seconds", args.max_seconds))
                    if value is not None}
    if batch_limits and not online_dataset:
        raise ValueError("benchmark-wide limits are available for online SciFact and SciTabAlign")

    def finish(predictions: list[dict], configuration: dict, batch: dict | None = None) -> None:
        if hashlib.sha256(args.input_manifest.read_bytes()).hexdigest() != manifest_hash:
            raise ValueError("benchmark input manifest changed during review")
        document = {"schema_version": 1, "predictions": predictions}
        write_json(output_path, document)
        record = {"schema_version": 1, "input_manifest_sha256": manifest_hash,
                  "input_sample_ids": [row["sample_id"] for row in manifest["samples"]],
                  "predictions_sha256": semantic_manifest_sha256(document), "configuration": configuration,
                  "wall_seconds": round(time.perf_counter() - started, 6), "prediction_count": len(predictions),
                  "samples": results}
        summary = {"predictions_path": str(output_path), "prediction_count": len(predictions)}
        if batch is not None:
            record["batch"] = summary["batch"] = batch
        write_json(record_path, record)
        emit(summary)

    if args.dataset == "scifact":
        from .benchmark import benchmark_configuration
        from .scifact import run_scifact, scifact_checkpoint

        engine = args.engine or "offline"
        configuration = benchmark_configuration(engine, args.model)
        configuration["dataset"] = "scifact-cited-abstracts"
        if online_dataset:
            with scifact_checkpoint(checkpoints, manifest_hash, configuration, resume=args.resume, **batch_limits) as batch:
                predictions = run_scifact(manifest, engine, args.model, batch=batch)
                results.extend(batch.results)
                finish(predictions, configuration, batch.summary(len(manifest["samples"])))
            return
        if args.resume:
            raise ValueError("--resume is available for PDF benchmarks and online SciFact or SciTabAlign")
        predictions = run_scifact(manifest, engine, args.model)
    elif args.dataset == "scitabalign":
        from .benchmark import benchmark_configuration
        from .scitabalign import run_scitabalign

        engine = args.engine or "offline"
        configuration = benchmark_configuration(engine, args.model)
        configuration["dataset"] = "scitabalign-cells"
        if online_dataset:
            from .models import ModelUnavailable, ResponsesModel
            from .scifact import benchmark_checkpoint

            max_output_tokens = 2048
            configuration["max_output_tokens"] = max_output_tokens
            with benchmark_checkpoint(checkpoints, manifest_hash, configuration, dataset="SciTabAlign",
                                      resume=args.resume, include_request_usage=True, **batch_limits) as batch:
                try:
                    model = ResponsesModel(args.model, max_requests=batch.max_requests,
                                           max_seconds=batch.remaining_seconds, previous_attempts=batch.requests_used,
                                           before_call=batch.reserve, max_output_tokens=max_output_tokens)
                except ModelUnavailable as exc:
                    raise ValueError(str(exc)) from exc
                predictions = run_scitabalign(manifest, model=model, batch=batch)
                results.extend(batch.results)
                finish(predictions, configuration, batch.summary(len(manifest["samples"])))
            return
        if args.resume:
            raise ValueError("--resume is available for PDF benchmarks and online SciFact or SciTabAlign")
        predictions = run_scitabalign(manifest)
    elif args.engine:
        from .benchmark import benchmark_configuration, run_pdf_samples

        service = review_service(args.data_dir)
        configuration = benchmark_configuration(args.engine, args.model, **retrieval_options)
        recovered = read_json(checkpoints) if args.resume and checkpoints.is_file() else {}
        if not isinstance(recovered, dict):
            raise ValueError("invalid benchmark checkpoint file")

        def save_sample(sample_id: str, input_digest: str, rows: list[dict]) -> None:
            recovered[sample_id] = {"input_digest": input_digest, "predictions": rows, "result": results[-1]}
            write_json(checkpoints, recovered)

        def save_pause(sample_id: str, input_digest: str, run: dict) -> None:
            recovered[sample_id] = {"input_digest": input_digest, "pause": run}
            write_json(checkpoints, recovered)

        predictions = run_pdf_samples(service, args.input_manifest, args.engine, args.model,
                                      recovered=recovered, on_sample=save_sample, on_pause=save_pause,
                                      on_result=results.append,
                                      configuration=configuration, **retrieval_options)
    else:
        if args.resume:
            raise ValueError("--resume requires a PDF benchmark engine")
        service = review_service(args.data_dir)
        configuration = {"engine": "fixture"}
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
    finish(predictions, configuration)


def score_benchmark(args: argparse.Namespace) -> None:
    from .scoring import score_predictions

    source_paths = {args.predictions.resolve(), args.gold_manifest.resolve()}
    if args.run_record:
        if args.dataset not in (None, "scifact"):
            raise ValueError("--run-record is available for generic claim and SciFact scoring")
        source_paths.add(args.run_record.resolve())
    annotations = {}
    for name in ("semantic_matches", "verdict_projections"):
        path = getattr(args, name)
        if path is None:
            continue
        if args.dataset:
            raise ValueError(f"--{name.replace('_', '-')} is only available for generic claim scoring")
        annotations[name] = path
        source_paths.add(path.resolve())
    if args.output.resolve() in source_paths:
        raise ValueError("score output must differ from scoring input files")
    predictions, gold = read_json(args.predictions), read_json(args.gold_manifest)
    run_record = None
    if args.run_record:
        run_record = read_json(args.run_record)
        if not isinstance(run_record, dict):
            raise ValueError("run record must be a JSON object")
    if args.dataset == "scifact":
        from .scifact import score_scifact

        result = score_scifact(predictions, gold, run_record=run_record)
    elif args.dataset == "claimbench":
        from .claimbench import score_claimbench

        result = score_claimbench(predictions, gold)
    elif args.dataset == "scitabalign":
        from .scitabalign import score_scitabalign

        result = score_scitabalign(predictions, gold)
    else:
        reviews = {name: read_json(path) for name, path in annotations.items()}
        for name, content in reviews.items():
            if not isinstance(content, dict):
                raise ValueError(f"{name.replace('_', ' ')} must be a JSON object")
        result = score_predictions(predictions, gold, run_record=run_record, **reviews)
    write_json(args.output, result)
    emit(result)


def score_review_benchmark(args: argparse.Namespace) -> None:
    import hashlib

    from .review_quality import score_review_quality

    sources = {args.run_record.resolve(), args.gold_manifest.resolve()}
    if args.pricing:
        sources.add(args.pricing.resolve())
    if args.output.resolve() in sources:
        raise ValueError("review score output must differ from its input files")
    run_source = args.run_record.read_bytes()
    gold_source = args.gold_manifest.read_bytes()
    run_record = json.loads(run_source.decode("utf-8-sig"))
    configuration = run_record.get("configuration") if isinstance(run_record, dict) else None
    if (not isinstance(configuration, dict) or configuration.get("dataset") is not None
            or configuration.get("engine") not in ("offline", "openai") or not run_record.get("samples")):
        raise ValueError("score-reviews requires a PDF benchmark run record with samples")
    result = score_review_quality(run_record, json.loads(gold_source.decode("utf-8-sig")))
    result["input_sha256"] = {
        "run_record": hashlib.sha256(run_source).hexdigest(),
        "gold_manifest": hashlib.sha256(gold_source).hexdigest(),
    }
    if args.pricing:
        from .benchmark_costs import score_costs

        pricing_source = args.pricing.read_bytes()
        result["estimated_cost"] = score_costs(run_record, json.loads(pricing_source.decode("utf-8-sig")))
        result["input_sha256"]["pricing"] = hashlib.sha256(pricing_source).hexdigest()
    write_json(args.output, result)
    emit(result)


def prepare_scifact_benchmark(args: argparse.Namespace) -> None:
    from .scifact import prepare_scifact

    input_manifest = args.output_dir / "inputs.json"
    gold_manifest = args.output_dir / "gold.json"
    if len({path.resolve() for path in (input_manifest, gold_manifest, args.corpus, args.claims)}) != 4:
        raise ValueError("SciFact source and output files must be distinct")
    inputs, gold = prepare_scifact(args.corpus, args.claims, split=args.split, dataset_version=args.dataset_version)
    write_json(input_manifest, inputs)
    write_json(gold_manifest, gold)
    emit({"input_manifest": str(input_manifest), "gold_manifest": str(gold_manifest),
          "sample_count": len(inputs["samples"])})


def prepare_scitabalign_benchmark(args: argparse.Namespace) -> None:
    from .scitabalign import prepare_scitabalign

    input_manifest = args.output_dir / "inputs.json"
    gold_manifest = args.output_dir / "gold.json"
    if len({path.resolve() for path in (input_manifest, gold_manifest, args.annotations)}) != 3:
        raise ValueError("SciTabAlign source and output files must be distinct")
    inputs, gold = prepare_scitabalign(args.annotations, dataset_version=args.dataset_version)
    write_json(input_manifest, inputs)
    write_json(gold_manifest, gold)
    emit({"input_manifest": str(input_manifest), "gold_manifest": str(gold_manifest),
          "sample_count": len(inputs["samples"])})


def prepare_claimbench_benchmark(args: argparse.Namespace) -> None:
    from .claimbench import prepare_claimbench

    input_manifest = args.output_dir / "inputs.json"
    gold_manifest = args.output_dir / "gold.json"
    if len({path.resolve() for path in (input_manifest, gold_manifest, args.pdf, args.annotations)}) != 4:
        raise ValueError("CLAIM-BENCH source and output files must be distinct")
    inputs, gold = prepare_claimbench(
        args.pdf, args.annotations, output_dir=args.output_dir, paper_family_id=args.paper_family_id,
        split=args.split, dataset_version=args.dataset_version,
    )
    write_json(input_manifest, inputs)
    write_json(gold_manifest, gold)
    emit({"input_manifest": str(input_manifest), "gold_manifest": str(gold_manifest),
          "claim_count": len(gold["labels"])})


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
    start.add_argument("--rerank-model", type=Path, help="path to an installed local CrossEncoder")
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
    run.add_argument("--engine", choices=("offline", "openai"), help="benchmark engine for PDF, SciFact or SciTabAlign")
    run.add_argument("--dataset", choices=("scifact", "scitabalign"), help="select a public benchmark adapter")
    run.add_argument("--model", default=os.environ.get("ABSTRACT_REVIEW_MODEL", "gpt-6.2-sol"))
    run.add_argument("--embed-model", type=Path, help="path to an installed local Sentence Transformers model")
    run.add_argument("--rerank-model", type=Path, help="path to an installed local CrossEncoder")
    run.add_argument("--resume", action="store_true",
                     help="reuse completed checkpoints and resume matching paused PDF or online dataset runs")
    run.add_argument("--max-requests", type=int,
                     help="total online SciFact or SciTabAlign provider attempts, including resumes (default: 8)")
    run.add_argument("--max-seconds", type=int,
                     help="total online SciFact or SciTabAlign elapsed seconds, including resumes (default: 180)")
    run.set_defaults(handler=run_benchmark)

    score = bench_commands.add_parser("score", help="score predictions independently of review")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--gold-manifest", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--run-record", type=Path,
                       help="predictions-bound run.json for execution and pause diagnostics alongside scores")
    score.add_argument("--semantic-matches", type=Path, help="independent one-to-one proposition equivalence annotations")
    score.add_argument("--verdict-projections", type=Path, help="independent annotations of refuted conjuncts")
    score.add_argument("--dataset", choices=("scifact", "claimbench", "scitabalign"), help="select dataset-specific metrics")
    score.set_defaults(handler=score_benchmark)

    quality = bench_commands.add_parser("score-reviews", help="score persisted review artifacts against human checks")
    quality.add_argument("--run-record", type=Path, required=True)
    quality.add_argument("--gold-manifest", type=Path, required=True)
    quality.add_argument("--pricing", type=Path, help="versioned USD token rate table for cost estimates")
    quality.add_argument("--output", type=Path, required=True)
    quality.set_defaults(handler=score_review_benchmark)

    scifact = bench_commands.add_parser("prepare-scifact", help="adapt labeled SciFact release JSONL files")
    scifact.add_argument("--corpus", type=Path, required=True)
    scifact.add_argument("--claims", type=Path, required=True)
    scifact.add_argument("--split", choices=("train", "dev"), required=True)
    scifact.add_argument("--dataset-version", required=True)
    scifact.add_argument("--output-dir", type=Path, required=True)
    scifact.set_defaults(handler=prepare_scifact_benchmark)

    scitabalign = bench_commands.add_parser("prepare-scitabalign", help="adapt SciTabAlign release tables for cell retrieval")
    scitabalign.add_argument("--annotations", type=Path, required=True)
    scitabalign.add_argument("--dataset-version", required=True)
    scitabalign.add_argument("--output-dir", type=Path, required=True)
    scitabalign.set_defaults(handler=prepare_scitabalign_benchmark)

    claimbench = bench_commands.add_parser("prepare-claimbench", help="align local CLAIM-BENCH quotes to a PDF")
    claimbench.add_argument("--pdf", type=Path, required=True)
    claimbench.add_argument("--annotations", type=Path, required=True)
    claimbench.add_argument("--paper-family-id", required=True)
    claimbench.add_argument("--split", required=True)
    claimbench.add_argument("--dataset-version", required=True)
    claimbench.add_argument("--output-dir", type=Path, required=True)
    claimbench.set_defaults(handler=prepare_claimbench_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"abstract-review: {exc}", file=sys.stderr)
        return 1
    return 0
