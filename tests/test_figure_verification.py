import hashlib
from base64 import b64decode

import pymupdf

from abstract_review.revisions import build_candidate, new_version, verify_version


class FiguresOnlyModel:
    def __init__(self, replacement):
        self.replacement = replacement
        self.calls = []

    def call(self, stage, material, schema):
        self.calls.append((stage, material))
        if stage == "revise":
            return schema.model_validate({
                "replacement": self.replacement, "evidence_ids": ["fig"], "semantic_diff": [{
                    "kind": "modified", "field": "direction", "before": "rose", "after": "fell",
                    "evidence_ids": ["fig"],
                }],
            })
        data = {"status": "passed", "reason": "The proposed statement follows the source.", "evidence_ids": ["fig"]}
        if stage == "verify_abstract":
            data["issues"] = []
        return schema.model_validate(data)


def _fixture(tmp_path, source="Measured error bars rose."):
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=200, height=150)
        page.draw_rect((30, 40, 80, 70), color=(0, 0, 0), fill=(0, 0.5, 0))
        page.set_rotation(90)
        data = pdf.tobytes()
    checksum = hashlib.sha256(data).hexdigest()
    archive = tmp_path / "blobs" / f"{checksum}.pdf"
    archive.parent.mkdir()
    archive.write_bytes(data)
    doc_id = f"main-{checksum}"
    original = new_version(source, "pdf_original", "", [])
    scoped = {"evidence_id": "fig", "doc_id": doc_id, "kind": "figure", "text": "Measured error bars",
              "locator": {"section": "Results", "page_index": 0, "bbox": [30, 40, 80, 70]}}
    other = {**scoped, "evidence_id": "unlinked", "text": "Isotope mineral spectrum",
             "locator": {**scoped["locator"], "bbox": [100, 100, 120, 120]}}
    report = {
        "original_abstract": original, "evidence": [scoped, other], "revisions": [], "verifications": [],
        "unresolved_items": [], "documents": [{"doc_id": doc_id, "sha256": checksum,
                                                "blob_path": f"blobs/{checksum}.pdf", "pages": [{"page_index": 0}]}],
        "findings": [{"claim_id": "claim", "sentence_id": "s1", "start": 0, "end": len(source), "text": source,
                      "proposition": source, "review_status": "completed", "verdict": "refuted",
                      "reason": "The source shows otherwise.", "evidence_ids": ["fig"]}],
    }
    return report, archive


def test_candidate_verification_shares_real_scoped_crops_with_whole_abstract(tmp_path):
    report, _ = _fixture(tmp_path)
    model = FiguresOnlyModel("Measured error bars fell.")
    candidate = build_candidate(report, model, data_dir=tmp_path)
    assert candidate is not None
    result = verify_version(report, report["original_abstract"], candidate, model, data_dir=tmp_path)
    assert result["status"] == "passed"

    (draft_stage, draft), (sentence_stage, sentence), (whole_stage, whole) = model.calls
    assert (draft_stage, sentence_stage, whole_stage) == ("revise", "verify_sentence", "verify_abstract")
    assert "figure_images" not in draft and "findings" not in sentence and "findings" not in whole
    assert sentence["figure_images"] == whole["figure_images"]
    for packet in (sentence, whole):
        assert [row["evidence_id"] for row in packet["evidence"]] == ["fig"]
        assert packet["figure_observations"] == [{"evidence_id": "fig", "status": "available", "numeric_precision": "approximate"}]
        image, = packet["figure_images"]
        prefix, pixels = image["image_url"].split(",", 1)
        assert prefix == "data:image/png;base64" and image["evidence_id"] == "fig"
        visual = pymupdf.Pixmap(b64decode(pixels))
        assert (visual.width, visual.height) == (100, 60)
        assert visual.pixel(50, 30)[1] > 90 and visual.pixel(50, 30)[0] == 0


def test_image_only_does_not_pass_exact_new_visual_number(tmp_path):
    report, _ = _fixture(tmp_path, "Measured error bars included 12 peaks.")
    original = report["original_abstract"]
    model = FiguresOnlyModel("Measured error bars included 14 peaks.")
    selected = new_version(model.replacement, "selected", original["version_id"], [])
    result = verify_version(report, original, selected, model, data_dir=tmp_path)
    sentence, = result["sentence_checks"]
    assert sentence["status"] == "manual_review" and "visual source" in sentence["reason"]
    assert result["status"] == "manual_review"
    for _, packet in model.calls:
        assert packet["figure_observations"][0]["numeric_precision"] == "approximate"


def test_missing_or_wrong_version_pdf_is_never_sent_as_a_figure(tmp_path):
    report, archive = _fixture(tmp_path)
    original = report["original_abstract"]
    model = FiguresOnlyModel("Measured error bars fell.")
    selected = new_version(model.replacement, "selected", original["version_id"], [])
    archive.write_bytes(b"a different archive version")
    result = verify_version(report, original, selected, model, data_dir=tmp_path)
    assert result["status"] == "manual_review"
    sentence, = result["sentence_checks"]
    assert sentence["status"] == "manual_review" and "unavailable" in sentence["reason"]
    for _, packet in model.calls:
        assert "figure_images" not in packet
        assert packet["figure_observations"] == [{"evidence_id": "fig", "status": "missing", "numeric_precision": None}]
