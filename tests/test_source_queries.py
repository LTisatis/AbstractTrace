from contextlib import closing

from abstract_review.evidence import EvidenceIndex
from abstract_review.source_queries import _probe, focused_source_context


def _lines(page, texts, *, doc="paper"):
    return [
        {"evidence_id": f"{doc}-{page}-{line}", "doc_id": doc, "kind": "body_text", "text": text,
         "locator": {"page_index": page, "section": "Introduction", "bbox": [10, line * 12, 200, line * 12 + 10]}}
        for line, text in enumerate(texts)
    ]


def test_removal_probe_finds_same_source_for_mirrored_claims():
    first = "The analysis finds removing the sensor guard leads to a signal failure."
    mirror = "The analysis finds removing the sensor guard prevents a signal failure."
    assert _probe(first) == _probe(mirror)
    rows = [*_lines(0, ["This detector had an optional sensor guard.",
                        "Removing it resulted in a signal failure.", "The experiment used the same detector."]),
            *_lines(1, ["Calibration logged signal failure.", "Humidity altered the detector.", "Results were stored."])]
    with closing(EvidenceIndex(rows)) as index:
        sources = [focused_source_context(index, claim, [{"doc_id": "paper", "role": "main"}])
                   for claim in (first, mirror)]
    assert sources[0] == sources[1]
    window, scope = sources[0]
    assert window.evidence_ids == ("paper-0-0", "paper-0-1", "paper-0-2")
    assert scope["mode"] == "ranked" and scope["focused_page_index"] == 0


def test_paired_numeric_probe_locates_comparison_without_converting_units():
    claim = "The two detectors improve on the baseline by 1.9 and 3.7 percentage points respectively."
    result = ["The study evaluated the instruments at one site.",
              "The improvements were 1.9% and 3.7% compared with the previous result.",
              "The protocol used the same threshold for each detector."]
    rows = [*_lines(0, ["The recorded measurements are 1.9% and 3.7%.", "These numbers are calibration.",
                        "Measurements are shown for illustration."]), *_lines(2, result)]
    with closing(EvidenceIndex(rows)) as index:
        window, scope = focused_source_context(index, claim, [{"doc_id": "paper", "role": "main"}])
        assert index.get(window.evidence_ids)[1]["text"] == result[1]
    assert scope["focus_anchor_id"] == "paper-2-1"
    assert window.evidence_ids == ("paper-2-0", "paper-2-1", "paper-2-2")


def test_probe_needs_a_main_document_and_matching_lexical_context():
    rows = _lines(0, ["The trial collected three audits.", "A signal failure disrupted diagnostics.",
                      "The test had two detector settings."])
    with closing(EvidenceIndex(rows)) as index:
        claim = "Removing the sensor guard prevents a signal failure."
        assert focused_source_context(index, claim, []) == (None, None)
        assert focused_source_context(index, claim, [{"doc_id": "paper", "role": "main"}]) == (None, None)


def test_upper_bound_probe_preserves_original_nearby_source_under_the_context_budget():
    claim = "The evaluation can contain up to 64k tokens."
    result = ["Two datasets were evaluated.", "Long sequences were prepared using one shared protocol.",
              "The varied lengths were 8k, 32k and 64 K tokens.", "Both datasets used this range.",
              "The distractors were randomly assigned to positions.", "Additional diagnostics follow below."]
    rows = [*_lines(0, ["A different protocol handled 64k words.", "The initial work has one dataset.",
                        "It lists a vocabulary size of 64,000 tokens."]),
            *_lines(1, result), *_lines(2, result, doc="supplement")]
    with closing(EvidenceIndex(rows)) as index:
        window, scope = focused_source_context(index, claim, [{"doc_id": "paper", "role": "main"}])
        assert window.text == "\n".join(result[1:5])
    assert scope["focus_anchor_id"] == "paper-1-2"
    assert window.evidence_ids == window.expanded_ids == tuple(f"paper-1-{line}" for line in range(1, 5))


def test_upper_bound_probe_requires_contiguous_number_and_supported_count_unit():
    rows = _lines(0, ["The batch contains 64k distinct rows for token analysis.",
                      "The abstract discusses tokens for both tasks.", "The protocol used random selection."])
    with closing(EvidenceIndex(rows)) as index:
        assert focused_source_context(index, "It handles up to 64k tokens.",
                                      [{"doc_id": "paper", "role": "main"}]) == (None, None)
    assert _probe("The intervention continues up to 64 hours.") is None
    assert _probe("The dataset contains at least 64k tokens.") is None
