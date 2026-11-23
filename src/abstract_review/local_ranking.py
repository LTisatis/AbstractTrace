"""Optional local inference for evidence retrieval."""

import math
from hashlib import file_digest, sha256
from pathlib import Path
from typing import Protocol


class PairRanker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


def validate_snapshot(model_path: Path | str, *, sentence_transformer: bool = False) -> Path:
    """Find a complete local snapshot with safetensors weights."""
    path = Path(model_path).resolve()
    if not path.is_dir():
        raise ValueError(f"expected a local Sentence Transformers model snapshot: {path}")
    if sentence_transformer and not (path / "modules.json").is_file():
        raise ValueError(f"missing Sentence Transformers modules.json in {path}")
    for filename in ("config.json", "model.safetensors"):
        if not (path / filename).is_file():
            raise ValueError(f"missing {filename} in local model snapshot: {path}")
    return path


def local_model_fingerprint(model_path: Path | str, *, sentence_transformer: bool = False) -> str:
    """Bind a run to the content of its local model and tokenizer."""
    path = validate_snapshot(model_path, sentence_transformer=sentence_transformer)
    fingerprint = sha256()
    suffixes = {".json", ".model", ".safetensors", ".spm", ".tiktoken", ".txt", ".vocab"}
    for artifact in sorted(path.rglob("*"), key=lambda entry: entry.as_posix()):
        name = artifact.relative_to(path)
        if any(part.startswith(".") for part in name.parts):
            continue
        if artifact.suffix in suffixes and artifact.is_file():
            fingerprint.update(name.as_posix().encode("utf-8"))
            fingerprint.update(b"\0")
            with artifact.open("rb") as stream:
                fingerprint.update(file_digest(stream, "sha256").digest())
    return fingerprint.hexdigest()


class LocalCrossEncoder:
    """Score passage relevance using a CPU model already present on disk."""

    def __init__(self, model_path: Path | str) -> None:
        path = validate_snapshot(model_path)
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ValueError("local reranking requires the embeddings extra") from exc
        self._model = CrossEncoder(
            str(path),
            local_files_only=True,
            trust_remote_code=False,
            device="cpu",
            token=False,
            model_kwargs={"use_safetensors": True},
            activation_fn=lambda logits: logits,
        )

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        raw_scores = self._model.predict(
            [(query, passage) for passage in passages],
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        try:
            scores = [float(value) for value in raw_scores]
        except (TypeError, ValueError) as exc:
            raise ValueError("local reranker must return one scalar per passage") from exc
        if len(scores) != len(passages) or not all(map(math.isfinite, scores)):
            raise ValueError("local reranker must return one finite score per passage")
        return scores
