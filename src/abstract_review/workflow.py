"""Drive the local review and commit runnable checkpoints between stages."""

from contextlib import ExitStack, nullcontext

from .audit import audit_version, refresh_compound_findings, review_claim
from .evidence import EvidenceIndex, SentenceTransformerEmbedder
from .followups import FollowUps
from .models import ModelUnavailable
from .run_store import RunBusy


def process(service, run_id: str, request_id: str | None, add_requests: int, add_seconds: int,
            *, lease_token: str | None = None) -> dict:
    from .revisions import verify_version

    if add_requests < 0 or add_seconds < 0 or ((add_requests or add_seconds) and not request_id):
        raise ValueError("positive budget extensions require a request ID")
    service.store.read(run_id)
    lease = service.store.acquire(run_id) if lease_token is None else nullcontext(lease_token)
    with lease as token:
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
            followups = FollowUps(report, checkpoint)
            version = report["reviewed_abstract"]
            embedding_path = configuration.get("embed_model")
            embedder = SentenceTransformerEmbedder(embedding_path) if embedding_path and (
                version.get("kind") != "selected" or model is not None
            ) else None
            data_dir = service.database.parent
            if version.get("kind") == "selected":
                while True:
                    review = verify_version(report, report["original_abstract"], version, model, followups=followups)
                    if not _refresh_findings(report, model, followups, embedder, data_dir):
                        report["abstract_review"] = review
                        break
            else:
                _review_original(report, model, checkpoint, data_dir, followups, embedder, service._store_pdf)
                candidate, review = _review_candidate(report, model, followups, embedder, data_dir)
                if candidate is not None:
                    report["reviewed_abstract"] = candidate
                    report["report_version_id"] = candidate["version_id"]
                    report["abstract_versions"].append(candidate)
                report["abstract_review"] = review
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


def _review_original(report: dict, model, checkpoint, data_dir, followups: FollowUps, embedder, archive_pdf) -> None:
    configuration = report["configuration"]
    audit_version(report, model, checkpoint, embedder=embedder, data_dir=data_dir, reserve_followup=followups.reserve)
    if any(finding.get("retryable") for finding in report["findings"]):
        raise ModelUnavailable("some original findings require the model to resume")
    if configuration["check_omissions"] or configuration["check_external"]:
        from .references import check_external, check_omissions

        if configuration["check_omissions"] and not report.get("omissions_checked"):
            report["omissions"] = check_omissions(report, model)
            report["omissions_checked"] = True
            checkpoint()
        if configuration["check_external"] and not report.get("references_checked"):
            report["external_checks"] = check_external(report, model=model, cache_dir=data_dir / "references",
                                                        archive_pdf=archive_pdf)
            report["references_checked"] = True
            checkpoint()


def _refresh_findings(report: dict, model, followups: FollowUps, embedder, data_dir) -> bool:
    if model is None:
        return False
    targets = [(number, row, followups.rounds.get(row["claim_id"], {}))
               for number, row in enumerate(report["findings"])]
    targets = [(number, row, ledger) for number, row, ledger in targets
               if ledger.get("stage") in {"verify_sentence", "verify_abstract"}
               and ledger.get("evidence_ids") and "original_recheck" not in ledger]
    if not targets:
        return False
    index = EvidenceIndex(report["evidence"], embedder=embedder)
    changed = False
    try:
        for number, original, ledger in targets:
            claim_id = original["claim_id"]
            added = followups.material([claim_id], [])
            claim = next(row for row in report["claims"] if row["claim_id"] == claim_id)
            finding, calculation = review_claim(claim, index, model, documents=report["documents"],
                                                data_dir=data_dir, additional_evidence=added)
            if finding.get("retryable"):
                raise ModelUnavailable("rechecking an original finding requires the model to resume")
            changed_label = (finding["review_status"], finding["verdict"]) != (
                original["review_status"], original["verdict"]
            )
            directly_cites_new = bool(set(finding["evidence_ids"]).intersection(ledger["evidence_ids"]))
            ledger["original_recheck"] = {"status": finding["review_status"], "verdict": finding["verdict"],
                                          "adopted": changed_label and directly_cites_new}
            if ledger["original_recheck"]["adopted"]:
                changed = True
                report.setdefault("finding_history", []).append(original)
                report["findings"][number] = finding
                refresh_compound_findings(report)
                report["calculations"] = [row for row in report["calculations"] if row["claim_id"] != claim_id]
                if calculation is not None:
                    report["calculations"].append({**calculation, "claim_id": claim_id})
                report["unresolved_items"] = [row for row in report["unresolved_items"]
                                              if row.get("claim_id") != claim_id]
                if finding["review_status"] != "completed" or finding["verdict"] == "insufficient":
                    report["unresolved_items"].append({"claim_id": claim_id, "reason": finding["reason"]})
                for revision in report["revisions"]:
                    if revision["sentence_id"] == claim["sentence_id"] and revision["status"] == "passed":
                        revision.update(status="superseded", supersede_reason="New evidence changed the original finding.")
                report.pop("candidate_proposal", None)
            followups.checkpoint()
    finally:
        index.close()
    return changed


def _review_candidate(report: dict, model, followups: FollowUps, embedder, data_dir) -> tuple[dict | None, dict]:
    from .revisions import build_candidate, repair_candidate, verify_version

    original = report["original_abstract"]
    while True:
        if "candidate_proposal" not in report:
            report["candidate_proposal"] = build_candidate(report, model, followups=followups)
            followups.checkpoint()
        if _refresh_findings(report, model, followups, embedder, data_dir):
            continue
        candidate = report["candidate_proposal"]
        review = verify_version(report, original, candidate or original, model, followups=followups)
        if _refresh_findings(report, model, followups, embedder, data_dir):
            continue
        if candidate is None or review["status"] != "returned" or model is None or report.get("whole_repair_attempted"):
            return candidate, review
        repaired = repair_candidate(report, candidate, review, model, followups=followups)
        report["whole_repair_attempted"] = True
        if repaired is None:
            followups.checkpoint()
            return candidate, review
        report["candidate_proposal"] = repaired
        followups.checkpoint()


def submit_selection(service, run_id: str, base_version_id: str, submission_id: str,
                     revision_ids: tuple[str, ...], edited_text: str | None, *, enqueue: bool = False) -> dict:
    from .revisions import apply_revisions, new_version

    if not submission_id.strip() or (edited_text is not None and revision_ids) or len(set(revision_ids)) != len(revision_ids):
        raise ValueError("selection requires a submission ID and exactly one unambiguous edit method")
    payload = {"base": base_version_id, "revision_ids": list(revision_ids), "edited_text": edited_text}

    def existing_selection(report: dict) -> dict | None:
        entry = report.get("selections", {}).get(submission_id)
        if entry is None:
            return None
        if entry["payload"] != payload:
            raise ValueError("submission ID already refers to a different selection")
        return service._run_ref(service.store.read(run_id, entry["version_id"]))

    if enqueue and (existing := existing_selection(service.store.read(run_id))) is not None:
        return existing
    with ExitStack() as resources:
        try:
            token = resources.enter_context(service.store.acquire(run_id))
        except RunBusy:
            if enqueue and (existing := existing_selection(service.store.read(run_id))) is not None:
                return existing
            raise
        report = service.store.read(run_id)
        if (existing := existing_selection(report)) is not None:
            return existing
        stored = report.setdefault("selections", {})
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
        if enqueue:
            service.store.save(report, token, enqueue=("resume", {}))
            return service._run_ref(report)
        service.store.save(report, token)
    return process(service, run_id, None, 0, 0)
