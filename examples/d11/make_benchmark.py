"""Generate two synthetic PDF families and separate input and gold manifests."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pymupdf

from abstract_review.ingest import parse_pdf


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    verdict: str
    source: Literal["body", "delta"] | None = None


@dataclass(frozen=True)
class Paper:
    name: str
    split: str
    body: str
    baseline: str
    final: str
    claims: tuple[Claim, ...]


PAPERS = (
    Paper(
        name="Atlas", split="dev", body="The test split for Atlas was held fixed.", baseline="80%", final="88%",
        claims=(
            Claim("fixed-split", "The test split for Atlas was held fixed.", "supported", "body"),
            Claim("wrong-delta", "Accuracy increased by 12 percentage points on Atlas.", "refuted", "delta"),
            Claim("unknown-device", "Atlas experiments used a dedicated GPU cluster.", "insufficient"),
        ),
    ),
    Paper(
        name="Maple", split="test", body="The Maple evaluation kept the threshold fixed.", baseline="70%", final="75%",
        claims=(
            Claim("right-delta", "Accuracy increased by 5 percentage points on Maple.", "supported", "delta"),
            Claim("unknown-site", "Maple data were collected at a single hospital.", "insufficient"),
        ),
    ),
)


def write_paper(path: Path, paper: Paper) -> None:
    with pymupdf.open() as document:
        page = document.new_page(width=480, height=600)
        page.insert_text((64, 56), f"Synthetic accuracy evaluation for {paper.name}", fontsize=17)
        page.insert_text((64, 96), "Abstract", fontsize=13)
        page.insert_text((64, 116), "We compare the baseline and final accuracy.", fontsize=11)
        page.insert_text((64, 158), "1. Methods", fontsize=13)
        page.insert_text((64, 190), paper.body, fontsize=11)

        page = document.new_page(width=480, height=600)
        page.insert_text((64, 56), "2. Results", fontsize=13)
        page.insert_text((64, 97), "Table 1. Accuracy by dataset", fontsize=11)
        xs, ys = (64, 166, 268, 370), (122, 148, 174)
        for x in xs:
            page.draw_line((x, ys[0]), (x, ys[-1]), width=0.75)
        for y in ys:
            page.draw_line((xs[0], y), (xs[-1], y), width=0.75)
        for row, values in enumerate((("Dataset", "Baseline", "Final"), (paper.name, paper.baseline, paper.final))):
            for column, value in enumerate(values):
                page.insert_text((xs[column] + 10, ys[row] + 17), value, fontsize=11)
        path.write_bytes(document.tobytes(no_new_id=True))


def source_ids(path: Path, paper: Paper) -> dict[str, list[str]]:
    """Resolve planted source anchors to the current parser's stable evidence IDs."""
    parsed = parse_pdf(path)
    body, = [item for item in parsed.evidence if item["kind"] == "body_text"
             and item["locator"]["page_index"] == 0 and item["text"] == paper.body]
    delta = []
    for column, value in enumerate((paper.baseline, paper.final), start=1):
        cell, = [item for item in parsed.evidence if item["kind"] == "table_cell"
                 and item["locator"]["page_index"] == 1 and item["locator"]["table_index"] == 0
                 and item["locator"]["row_index"] == 1 and item["locator"]["column_index"] == column
                 and item["text"] == value]
        delta.append(cell["evidence_id"])
    return {"body": [body["evidence_id"]], "delta": delta}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def make_benchmark(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, labels = [], []
    for paper in PAPERS:
        family_id = paper.name.lower()
        sample_id = f"{family_id}-v1"
        path = output_dir / f"{family_id}.pdf"
        write_paper(path, paper)
        evidence_by_source = source_ids(path, paper)
        samples.append({"sample_id": sample_id, "paper_family_id": family_id, "split": paper.split,
                        "pdf": path.name, "claims": [{"claim_id": claim.claim_id, "text": claim.text} for claim in paper.claims]})
        labels.extend({"sample_id": sample_id, "claim_id": claim.claim_id, "verdict": claim.verdict, "split": paper.split,
                       "accepted_evidence_groups": [evidence_by_source[claim.source]] if claim.source else []}
                      for claim in paper.claims)
    write_json(output_dir / "inputs.json", {"schema_version": 1, "samples": samples})
    write_json(output_dir / "gold.json", {"schema_version": 1, "labels": labels})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path(".review-data/d11"))
    args = parser.parse_args()
    make_benchmark(args.output_dir)


if __name__ == "__main__":
    main()
