"""Persist the one-round evidence budget shared by review and verification."""

from .evidence import EvidenceIndex


def _ignore_checkpoint() -> None:
    return None


class FollowUps:
    def __init__(self, report: dict, checkpoint=_ignore_checkpoint) -> None:
        self.report = report
        self.checkpoint = checkpoint
        self.rounds = report.setdefault("follow_up_rounds", {})
        for finding in report.get("findings", []):
            if finding.get("follow_up_performed"):
                self.rounds.setdefault(finding["claim_id"], {"stage": "judge"})

    def reserve(self, claim_id: str) -> bool:
        # audit_version checkpoints the round and its queries together.
        return self._reserve([claim_id], {"stage": "judge"})

    def _reserve(self, keys: list[str], record: dict) -> bool:
        if not keys or any(key in self.rounds for key in keys):
            return False
        for key in keys:
            self.rounds[key] = dict(record)
        return True

    def collect(self, keys: list[str], queries: list, prior: list[dict], stage: str) -> list[dict]:
        from .audit import normalize_targets, targeted_evidence

        documents = self.report.get("documents", ())
        requests = normalize_targets((
            request.model_dump(exclude_none=True) for request in queries
            if request.query.strip() and request.requirement.strip()
        ), documents)
        if not requests or any(not item.get("requirement") for item in requests):
            return []
        reservation = {"stage": stage, "queries": requests, "keys": keys,
                       "prior_ids": sorted({row["evidence_id"] for row in prior})}
        if self._reserve(keys, reservation):
            self.checkpoint()
        return self.material(keys, prior)

    def _complete(self, record: dict) -> None:
        from .audit import targeted_evidence

        index = EvidenceIndex(self.report["evidence"])
        try:
            result = targeted_evidence(index, record["queries"], documents=self.report.get("documents", ()),
                                       prior_ids=record["prior_ids"])
        finally:
            index.close()
        prior = set(record["prior_ids"])
        for key in record["keys"]:
            self.rounds[key].update(
                evidence_ids=[row["evidence_id"] for row in result["evidence"] if row["evidence_id"] not in prior],
                scopes=result["scopes"], paths=result["paths"],
            )
        self.checkpoint()

    def material(self, keys: list[str], prior: list[dict]) -> list[dict]:
        for key in keys:
            record = self.rounds.get(key, {})
            if "queries" in record and "evidence_ids" not in record:
                self._complete(record)
        seen = {row["evidence_id"] for row in prior}
        wanted = dict.fromkeys(eid for key in keys for eid in self.rounds.get(key, {}).get("evidence_ids", ()))
        available = {row["evidence_id"]: row for row in self.report["evidence"]}
        return [available[eid] for eid in wanted if eid not in seen and eid in available]
