"""Drive the local review and commit runnable checkpoints between stages."""

from .audit import audit_version
from .models import ModelUnavailable


def process(service, run_id: str, request_id: str | None, add_requests: int, add_seconds: int) -> dict:
    from .revisions import build_candidate, repair_candidate, verify_version

    if add_requests < 0 or add_seconds < 0 or ((add_requests or add_seconds) and not request_id):
        raise ValueError("positive budget extensions require a request ID")
    service.store.read(run_id)
    with service.store.acquire(run_id) as token:
        report = service.store.read(run_id)
        configuration = report.get("configuration")
        if configuration is None:
            raise ValueError("this review was started without an evaluation engine")
        if add_requests or add_seconds:
            extensions = report.setdefault("budget_extensions", {})
            payload = [add_requests, add_seconds]
            if request_id in extensions and extensions[request_id] != payload:
                raise ValueError("this request ID was already used for a different budget extension")
            if request_id not in extensions:
                configuration["max_requests"] += add_requests
                configuration["max_seconds"] += add_seconds
                extensions[request_id] = payload
        if report.get("run_status") == "report_ready":
            return service._run_ref(report)

        usage = report["usage"]
        previous_events = list(usage.get("model_events", []))
        previous_count = usage["external_requests"]
        model = None

        def checkpoint() -> None:
            if model is not None:
                usage["model_events"] = previous_events + model.events
                usage["external_requests"] = previous_count + len(model.events)
                usage["elapsed_seconds"] = round(model.elapsed_seconds, 3)
            service.store.save(report, token)

        report["run_status"] = "running"
        report.pop("runtime_error", None)
        checkpoint()
        try:
            model = service._responses(report, before_call=checkpoint)
            version = report["reviewed_abstract"]
            if version.get("kind") == "selected":
                report["abstract_review"] = verify_version(report, report["original_abstract"], version, model)
            else:
                _review_original(report, model, checkpoint, service.database.parent)
                if "candidate_proposal" not in report:
                    report["candidate_proposal"] = build_candidate(report, model)
                    checkpoint()
                candidate = report["candidate_proposal"]
                if candidate is not None:
                    review = verify_version(report, report["original_abstract"], candidate, model)
                    if review["status"] == "returned" and model is not None and not report.get("whole_repair_attempted"):
                        report["whole_repair_attempted"] = True
                        checkpoint()
                        repaired = repair_candidate(report, candidate, review, model)
                        if repaired is not None:
                            candidate = report["candidate_proposal"] = repaired
                            checkpoint()
                            review = verify_version(report, report["original_abstract"], candidate, model)
                    report["reviewed_abstract"] = candidate
                    report["report_version_id"] = candidate["version_id"]
                    report["abstract_versions"].append(candidate)
                    report["abstract_review"] = review
                else:
                    report["abstract_review"] = verify_version(report, report["original_abstract"], version, model)
                report.pop("candidate_proposal", None)
            report["run_status"] = "report_ready"
        except (ModelUnavailable, KeyboardInterrupt) as exc:
            report["run_status"] = "paused"
            report["runtime_error"] = str(exc) or "interrupted"
        except (OSError, ValueError) as exc:
            report["run_status"] = "failed"
            report["runtime_error"] = str(exc)
        finally:
            checkpoint()
        return service._run_ref(report)


def _review_original(report: dict, model, checkpoint, data_dir) -> None:
    configuration = report["configuration"]
    embedder = None
    if configuration["embed_model"]:
        from .evidence import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(configuration["embed_model"])
    audit_version(report, model, checkpoint, embedder=embedder, data_dir=data_dir)
    if any(finding.get("retryable") for finding in report["findings"]):
        raise ModelUnavailable("some original findings require the model to resume")
    if configuration["check_omissions"] or configuration["check_external"]:
        from .references import check_external, check_omissions

        if configuration["check_omissions"] and not report.get("omissions_checked"):
            report["omissions"] = check_omissions(report, model)
            report["omissions_checked"] = True
            checkpoint()
        if configuration["check_external"] and not report.get("references_checked"):
            report["external_checks"] = check_external(report, model=model, cache_dir=data_dir / "references")
            report["references_checked"] = True
            checkpoint()


def submit_selection(service, run_id: str, base_version_id: str, submission_id: str,
                     revision_ids: tuple[str, ...], edited_text: str | None) -> dict:
    from .revisions import apply_revisions, new_version

    if not submission_id.strip() or (edited_text is not None and revision_ids) or len(set(revision_ids)) != len(revision_ids):
        raise ValueError("selection requires a submission ID and exactly one unambiguous edit method")
    with service.store.acquire(run_id) as token:
        report = service.store.read(run_id)
        stored = report.setdefault("selections", {})
        payload = {"base": base_version_id, "revision_ids": list(revision_ids), "edited_text": edited_text}
        if submission_id in stored:
            if stored[submission_id]["payload"] != payload:
                raise ValueError("submission ID already refers to a different selection")
            return service._run_ref(service.store.read(run_id, stored[submission_id]["version_id"]))
        if report["report_version_id"] != base_version_id:
            raise ValueError("stale base version; refresh the current report before submitting")
        if report.get("configuration") is None:
            raise ValueError("run an evidence review before submitting a selection")
        by_id = {row["revision_id"]: row for row in report["revisions"] if row["status"] == "passed"}
        if any(revision_id not in by_id for revision_id in revision_ids):
            raise ValueError("selection includes a revision that has not passed verification")
        if edited_text is None:
            text = apply_revisions(report["original_abstract"]["text"], [by_id[rev] for rev in revision_ids])
        else:
            text = edited_text
        if not text.strip():
            raise ValueError("abstract text cannot be empty")
        version = new_version(text, "selected", base_version_id, list(revision_ids))
        stored[submission_id] = {"payload": payload, "version_id": version["version_id"]}
        report["abstract_versions"].append(version)
        report["reviewed_abstract"] = version
        report["report_version_id"] = version["version_id"]
        report["abstract_review"] = {"version_id": version["version_id"], "status": "pending"}
        report["run_status"] = "queued"
        service.store.save(report, token)
    return process(service, run_id, None, 0, 0)
