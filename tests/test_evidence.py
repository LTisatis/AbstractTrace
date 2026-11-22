import sys
from types import SimpleNamespace

import pytest

from abstract_review.evidence import EvidenceIndex, SentenceTransformerEmbedder


def node(eid, text, *, doc="paper", version="", section="Results", page=0, bbox=None, cell=None, **fields):
    locator = {"section": section, "page_index": page, "page_number": page + 1, "bbox": bbox}
    if cell is not None:
        locator.update(zip(("table_index", "row_index", "column_index"), cell))
    return {
        "evidence_id": eid,
        "doc_id": doc,
        "doc_version_id": version,
        "kind": "body_text" if cell is None else "table_cell",
        "text": text,
        "locator": locator,
        **fields,
    }


def hit_ids(result):
    return [item["evidence_id"] for item in result["evidence"]]


def test_signed_units_and_cjk_queries_keep_original_anchors():
    negative = node("z-negative", "误差率下降 −０．３％", bbox=[10, 20, 150, 35])
    positive = node("a-positive", "误差率上升 +0.3%")
    unitless = node("b-unitless", "误差率的偏移量为 -0.3")
    unrelated = node("c-other", "Accuracy and computational cost")
    index = EvidenceIndex([positive, unitless, unrelated, negative])

    assert hit_ids(index.search("-0.3%")) == ["z-negative"]
    assert hit_ids(index.search("+0.3%")) == ["a-positive"]
    assert hit_ids(index.search("-0.3")) == ["b-unitless"]
    result = index.search("误差率下降", limit=1)
    assert result["evidence"] == [negative]
    assert hit_ids(index.search("computational cost")) == ["c-other"]


def test_cell_search_uses_its_headers_and_normalizes_percentage_point_units():
    first = node(
        "first", "8", cell=(0, 1, 1), row_header="Dataset A", column_header="Improvement (百分点)"
    )
    second = node("second", "8%", cell=(0, 2, 1), row_header="Dataset B", column_header="Accuracy (%)")
    index = EvidenceIndex([first, second])

    assert hit_ids(index.search("8 percentage points", expand=False)) == ["first"]
    assert hit_ids(index.search("8%", expand=False)) == ["second"]
    assert hit_ids(index.search("Dataset A Improvement", limit=1)) == ["first"]


def test_graph_path_uses_labeled_caption_to_reach_method_with_two_hops():
    value = node("value", "37.2", page=2, cell=(0, 1, 1), bbox=[60, 100, 110, 120], column_header="Gain (%)")
    header = node("header", "Gain (%)", page=2, cell=(0, 0, 1), bbox=[60, 86, 110, 100])
    caption = node("caption", "Table 2. Verified gains.", page=2, bbox=[10, 54, 130, 80])
    method = node("method", "Thresholds for Table 2 used the test split.", section="Methods", page=1)
    wrong_version = node(
        "wrong-version", "Table 2 is from a different version.", version="older", section="Methods", page=1
    )
    wrong_paper = node("wrong-paper", "Table 2 is from a different document.", doc="unrelated", section="Methods")
    index = EvidenceIndex([wrong_paper, value, wrong_version, method, header, caption])

    assert hit_ids(index.search("37.2%", expand=False)) == ["value"]
    result = index.search("37.2%", limit=3)
    assert hit_ids(result) == ["value", "caption", "method"]
    path, = result["paths"]
    assert path["evidence_ids"] == ["value", "caption", "method"]
    assert [edge["relation"] for edge in path["edges"]] == ["table_caption", "table_reference"]
    assert [edge["anchors"][0]["locator"] for edge in path["edges"]] == [
        caption["locator"], method["locator"]
    ]
    assert {edge["doc_id"] for edge in path["edges"]} == {"paper"}
    assert "value" in hit_ids(index.search("表 ２", expand=False))

    scoped = index.search("37.2%", limit=3, section="results")
    assert "method" not in hit_ids(scoped)
    assert scoped["scope"]["locations"] == [{"doc_id": "paper", "page_index": 2, "section": "Results"}]


def test_uncertain_caption_never_links_to_neighboring_tables():
    caption = node("caption", "Table 2. Ambiguous placement.", bbox=[0, 60, 220, 70])
    first = node("first", "12", cell=(0, 0, 0), bbox=[5, 80, 95, 115])
    second = node("second", "13", cell=(1, 0, 0), bbox=[105, 80, 210, 115])
    index = EvidenceIndex([caption, first, second])

    assert index.search("12")["paths"] == []
    assert hit_ids(index.search("12")) == ["first"]
    assert hit_ids(index.search("Table 2", expand=False)) == ["caption"]


def test_row_header_and_explicit_experiment_links_respect_document_version():
    result = node("r", "quasars in experiment E17", version="draft")
    method = node("m", "实验 E17 使用固定条件", section="方法", version="draft")
    other = node("bad", "experiment E17 used other conditions", section="Methods", version="final")
    header = node("row", "Dataset X", cell=(0, 1, 0), version="draft")
    cell = node("value", "72%", cell=(0, 1, 1), version="draft", row_header="Dataset X")
    index = EvidenceIndex([result, method, other, header, cell])

    assert hit_ids(index.search("quasars", limit=2)) == ["r", "m"]
    assert index.search("quasars", limit=2)["paths"][0]["edges"][0]["relation"] == "same_experiment"
    expanded = index.search("72%", limit=2)
    assert hit_ids(expanded) == ["value", "row"]
    assert expanded["paths"][0]["edges"][0]["relation"] == "row_header"


def test_scoped_lookup_preserves_order_and_returns_detached_evidence():
    first = node("a", "Visible source")
    second = node("b", "Second source")
    index = EvidenceIndex([first, second])

    first["text"] = "modified caller input"
    records = index.get(["b", "a"])
    assert [record["text"] for record in records] == ["Second source", "Visible source"]
    records[1]["locator"]["page_index"] = 50
    assert index.get(["a"])[0]["locator"]["page_index"] == 0
    with pytest.raises(ValueError, match="outside the current scope"):
        index.get(["foreign"])
    with pytest.raises(ValueError, match="duplicate evidence ID"):
        EvidenceIndex([first, first])


def test_local_embedding_can_retrieve_non_lexical_matches():
    class FakeEmbedder:
        def encode(self, texts):
            return [[3, 0] if text in {"solar cell", "photovoltaic"} else [0, 4] for text in texts]

    evidence = [node("right", "solar cell"), node("wrong", "dark sky")]
    assert hit_ids(EvidenceIndex(evidence).search("photovoltaic")) == []
    result = EvidenceIndex(evidence, embedder=FakeEmbedder()).search("photovoltaic")
    assert hit_ids(result) == ["right"]
    assert result["scope"]["channels"] == ["lexical", "exact", "semantic"]


def test_model_adapter_requires_an_explicit_local_snapshot(tmp_path, monkeypatch):
    loads = []

    def load_model(path, **kwargs):
        loads.append((path, kwargs))
        return SimpleNamespace(encode=lambda texts, **options: [[1.0, 0.0] for text in texts])

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=load_model))
    with pytest.raises(ValueError, match="local Sentence Transformers model snapshot"):
        SentenceTransformerEmbedder(tmp_path / "remote-model")
    assert loads == []
    (tmp_path / "modules.json").write_text("[]", encoding="utf-8")
    embedder = SentenceTransformerEmbedder(tmp_path)
    assert loads == [(str(tmp_path.resolve()), {"local_files_only": True, "trust_remote_code": False, "device": "cpu"})]
    assert list(embedder.encode(["paper"])) == [[1.0, 0.0]]
