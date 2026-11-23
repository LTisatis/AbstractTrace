from contextlib import closing

from abstract_review.evidence import EvidenceIndex


def _line(eid, text, y, *, x=30, width=165, document="paper", version="v1", page=0, section="Results"):
    return {"evidence_id": eid, "text": text, "doc_id": document, "doc_version_id": version, "kind": "body_text",
            "locator": {"page_index": page, "section": section, "bbox": [x, y, x + width, y + 10]}}


def _ids(result):
    return [row["evidence_id"] for row in result["evidence"]]


def test_followup_keeps_full_counterexample_without_crossing_source_boundaries():
    windows = [
        _line("opening", "With the low-temperature control enabled,", 120),
        _line("condition", "only the samples processed overnight", 132),
        _line("trigger", "showed a reversed conductivity trend", 144),
        _line("effect", "under the reference protocol described", 156),
        _line("closing", "in the supplementary measurement notes.", 168),
    ]
    other_rows = [
        _line("before", "The instrument was calibrated.", 108),
        _line("after", "Calibration data are available.", 180),
        _line("right", "With a different temperature setting", 156, x=330),
        _line("other-page", "A later adjustment was made.", 156, page=1),
        _line("other-section", "A different protocol was used.", 156, section="Methods"),
        _line("other-file", "An independent follow-up paper.", 156, document="appendix"),
        _line("other-version", "The initial version had different inputs.", 156, version="v0"),
    ]
    expected = [row["evidence_id"] for row in windows]
    with closing(EvidenceIndex([*reversed(windows), *other_rows])) as index:
        ranked = index.search("reversed conductivity trend", followup=True, limit=8)
        assert _ids(ranked) == expected
        assert ranked["evidence"] == windows
        assert index.evidence_units(reversed(expected)) == [tuple(expected)]
        assert index.body_passages(reversed(expected)) == [{"evidence_ids": expected}]
        assert index.body_passages(expected[:-1]) == []
        unscoped = index.search("reversed conductivity trend", limit=8)
        assert _ids(unscoped) == ["trigger"]


def test_fitting_a_second_passage_never_uses_some_of_its_required_lines():
    opening = [_line(f"a{i}", text, 100 + 12 * i) for i, text in enumerate((
        "Pulse anomaly was observed only", "when the baseline feed was active", "in one preselected subset",
        "of the original calibration", "series under test.",
    ))]
    second = [_line(f"b{i}", text, 220 + 12 * i) for i, text in enumerate((
        "Pulse measurements from the backups", "required access to the alternate feed", "and all calibration",
        "files from that night.",
    ))]
    independent = _line("added", "Other instrumentation was adequate.", 400)
    with closing(EvidenceIndex([*opening, *second, independent])) as index:
        expected = [row["evidence_id"] for row in opening]
        ranked = index.search("pulse anomaly", followup=True, limit=8)
        assert _ids(ranked) == expected
        assert index.body_passages(_ids(ranked)) == [{"evidence_ids": expected}]

        directly = index.inspect(section="Results", query="pulse anomaly", limit=8)
        assert _ids(directly) == [*expected, "added"]
        assert directly["scope"]["matching_count"] == 10
        assert directly["scope"]["complete"] is False


def test_oversized_passage_preserves_seed_and_never_advertises_partial_context():
    rows = [_line(f"a{i}", text, 100 + 12 * i) for i, text in enumerate((
        "When the protocol was adapted", "for the specified cohort", "and the initial test", "excluded all third-party inputs",
        "the seedword condition occurred", "only at the selected", "benchmark site operated", "at the expected", "voltage.",
    ))]
    with closing(EvidenceIndex(rows)) as index:
        ranked = index.search("seedword", followup=True, limit=8)
        assert _ids(ranked) == ["a4"]
        assert index.body_passages(_ids(ranked)) == []
        assert index.body_passages([row["evidence_id"] for row in rows]) == []


def test_passage_ambiguity_never_chooses_a_duplicate_layout_line():
    rows = [
        _line("opening", "This group includes", 100),
        _line("a", "the counterdriver measurements.", 112),
        _line("b", "a overlapping text layer.", 112),
    ]
    with closing(EvidenceIndex(rows)) as index:
        ranked = index.search("counterdriver", followup=True)
        assert _ids(ranked) == ["a"]
        assert index.body_passages(["opening", "a", "b"]) == []
