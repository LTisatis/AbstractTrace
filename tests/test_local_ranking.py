import sys
from types import SimpleNamespace

import pytest

from abstract_review.local_ranking import LocalCrossEncoder, local_model_fingerprint


@pytest.fixture
def snapshot(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    return tmp_path


def test_local_cross_encoder_passes_ordered_pairs_and_disables_downloads(snapshot, monkeypatch):
    calls = []

    def predict(pairs, **options):
        calls.append((pairs, options))
        return [-4, 6]

    def load(path, **options):
        calls.append((path, options))
        return SimpleNamespace(predict=predict)

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=load))
    reranker = LocalCrossEncoder(snapshot)
    path, options = calls[0]
    assert path == str(snapshot.resolve())
    assert options["local_files_only"] is True
    assert options["trust_remote_code"] is False
    assert options["token"] is False
    assert options["device"] == "cpu"
    assert options["model_kwargs"] == {"use_safetensors": True}
    assert reranker.score("claim", []) == []
    assert reranker.score("claim", ["first", "second"]) == [-4.0, 6.0]
    assert calls[1] == (
        [("claim", "first"), ("claim", "second")],
        {"show_progress_bar": False, "convert_to_numpy": True},
    )


@pytest.mark.parametrize("scores", [[1], [float("nan"), 2], [[1, 2], [3, 4]]])
def test_local_cross_encoder_rejects_invalid_scores(snapshot, monkeypatch, scores):
    def load(*args, **kwargs):
        return SimpleNamespace(predict=lambda *args, **kwargs: scores)

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=load))
    with pytest.raises(ValueError, match="one .* per passage"):
        LocalCrossEncoder(snapshot).score("claim", ["first", "second"])


def test_local_cross_encoder_requires_safetensors_before_loading(snapshot, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    (snapshot / "model.safetensors").unlink()
    (snapshot / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(ValueError, match="missing model.safetensors"):
        LocalCrossEncoder(snapshot)


def test_model_fingerprint_binds_tokenizer_and_weights(snapshot):
    fingerprint = local_model_fingerprint(snapshot)
    (snapshot / "README.md").write_text("Model card", encoding="utf-8")
    assert local_model_fingerprint(snapshot) == fingerprint
    tokenizer = snapshot / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    with_tokenizer = local_model_fingerprint(snapshot)
    assert with_tokenizer != fingerprint
    (snapshot / "model.safetensors").write_bytes(b"updated weights")
    assert local_model_fingerprint(snapshot) != with_tokenizer
