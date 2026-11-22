"""Create a small PDF with native text and a bordered results table."""

import argparse
from pathlib import Path

import pymupdf


WIDTH = 480
HEIGHT = 600
ABSTRACT = "Accuracy increased from 80% to 88% on dataset A."
BODY = "Dataset A uses a fixed test split for the primary evaluation."
SUPPLEMENT = "The supplement records the evaluation seed as 314159."
TABLE = (
    ("Dataset", "Baseline", "Final"),
    ("A", "80%", "88%"),
    ("B", "70%", "79%"),
)
GRID_X = (64, 166, 268, 370)
GRID_Y = (122, 148, 174, 200)


def save(document: pymupdf.Document, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(document.tobytes(no_new_id=True))


def make_paper(path: Path) -> None:
    with pymupdf.open() as document:
        page = document.new_page(width=WIDTH, height=HEIGHT)
        page.insert_text((64, 56), "A synthetic accuracy evaluation", fontsize=18)
        page.insert_text((64, 96), "Abstract", fontsize=13)
        page.insert_text((64, 116), ABSTRACT, fontsize=11)
        page.insert_text((64, 166), "1. Introduction", fontsize=13)
        page.insert_text((64, 190), BODY, fontsize=11)
        page.insert_text((64, 222), "An unrelated example also reports 88%.", fontsize=11)

        page = document.new_page(width=WIDTH, height=HEIGHT)
        page.insert_text((64, 58), "2. Results", fontsize=13)
        page.insert_text((64, 97), "Table 1. Accuracy by dataset", fontsize=11)
        for x in GRID_X:
            page.draw_line((x, GRID_Y[0]), (x, GRID_Y[-1]), width=0.75)
        for y in GRID_Y:
            page.draw_line((GRID_X[0], y), (GRID_X[-1], y), width=0.75)
        for row, values in enumerate(TABLE):
            for column, value in enumerate(values):
                page.insert_text((GRID_X[column] + 10, GRID_Y[row] + 17), value, fontsize=11)
        save(document, path)


def make_supplement(path: Path) -> None:
    with pymupdf.open() as document:
        page = document.new_page(width=WIDTH, height=HEIGHT)
        page.insert_text((64, 60), "Supplementary Methods", fontsize=14)
        page.insert_text((64, 90), SUPPLEMENT, fontsize=11)
        save(document, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="output PDF for the two-page paper")
    parser.add_argument("--supplement", type=Path, help="optional output PDF for the supplement")
    args = parser.parse_args()
    if args.supplement is not None and args.output.resolve() == args.supplement.resolve():
        parser.error("paper and supplement outputs must be different files")
    make_paper(args.output)
    if args.supplement is not None:
        make_supplement(args.supplement)


if __name__ == "__main__":
    main()
