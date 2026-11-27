from contextlib import closing

from abstract_review.audit import review_claim, targeted_evidence
from abstract_review.evidence import EvidenceIndex


def _body(eid, text, page, section="Introduction"):
    return {"evidence_id": eid, "doc_id": "paper", "kind": "body_text", "text": text,
            "locator": {"page_index": page, "section": section}}


def test_recognized_printed_heading_routes_to_page_without_claiming_exhaustion():
    printed = "3.1 Controlled ablation"
    documents = [{"doc_id": "paper", "pages": [
        {"page_index": 0, "status": "parsed"},
        {"page_index": 1, "status": "parsed", "source_headings": [printed]},
    ]}]
    evidence = [_body("elsewhere", "The same ablation was described elsewhere.", 0),
                _body("trial", "We compared the same task under both settings.", 1)]
    target = {"doc_id": "paper", "page_index": 1, "section": printed,
              "query": "the matched controlled ablation", "requirement": "same test conditions"}
    with closing(EvidenceIndex(evidence)) as index:
        valid = targeted_evidence(index, [target], documents=documents)
        unknown = targeted_evidence(index, [{**target, "section": "Wrong heading"}], documents=documents)
        wrong_table = targeted_evidence(index, [{**target, "table_label": "3"}], documents=documents)

    assert [row["evidence_id"] for row in valid["evidence"]] == ["trial"]
    scope, = valid["scopes"]
    assert scope["printed_heading"] == printed and scope["section"] is None and not scope["complete"]
    for unverified in (unknown, wrong_table):
        direct, = [scope for scope in unverified["scopes"] if scope["mode"] == "direct"]
        assert direct["matching_count"] == 0 and "printed_heading" not in direct


def test_grounded_proof_labels_are_navigation_options_within_the_document():
    rows = [_body("theorem", "Theorem 2. We show the upper bound.", 3, "Proofs"),
            _body("mention", "We refer the reader to Theorem 7.", 4, "Proofs"),
            _body("reference", "Theorem 9. One earlier proof is cited.", 5, "References"),
            _body("unreadable", "Corollary 3. A legacy OCR record.", 6, "Proofs")]
    document = {"doc_id": "paper", "role": "main", "pages": [
        {"page_index": n, "status": "unreadable" if n == 6 else "parsed"} for n in range(3, 7)
    ]}
    claim = {"claim_id": "p", "kind": "theoretical", "proposition": "The bound rules out every finite-state language.",
             "requirements": ["scope of the bound"], "support_query": "upper bound", "counter_query": "counterexample"}

    class Navigator:
        def call(self, stage, material, schema):
            assert stage == "judge"
            assert material["source_hints"] == [{"doc_id": "paper", "page_index": 3, "headings": ["Theorem 2"]}]
            assert "may span pages" in material["coverage_instructions"]
            return schema(verdict="pending", reason="No decision yet.", evidence_groups=[],
                          missing_fields=["scope of the bound"], follow_up_queries=[])

    with closing(EvidenceIndex(rows)) as index:
        finding, _ = review_claim(claim, index, Navigator(), documents=[document])
    assert finding["review_status"] == "pending"
