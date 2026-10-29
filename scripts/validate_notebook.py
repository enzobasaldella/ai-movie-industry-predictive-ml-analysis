#!/usr/bin/env python3
"""Perform dependency-free structural validation of the analysis notebook."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "movie-analysis.ipynb"
RAW_ARCHIVE_PATH = ROOT / "IMDb Movies.zip"
PROCESSED_DATA_PATH = ROOT / "IMDb_Movies_Updated.csv"

REQUIRED_HEADINGS = {
    "# Movie Industry Analysis and Prediction",
    "## 3. Data loading and cleaning",
    "## 9. Logistic regression",
    "## 12. Neural-network baselines",
    "## 13. Neural-network evaluation",
}
RAW_COLUMNS = {
    "imdb_title_id",
    "original_title",
    "year",
    "avg_vote",
    "votes",
    "budget",
    "worlwide_gross_income",
    "metascore",
}
PROCESSED_COLUMNS = {
    "id",
    "title",
    "year",
    "imdb_score",
    "votes",
    "world_wide_boxoffice",
    "profit_margin",
}


def read_csv_header(stream: io.TextIOBase) -> set[str]:
    header = next(csv.reader(stream))
    return set(header)


def validate_notebook() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError("Expected notebook format 4.")

    headings: set[str] = set()
    code_cell_count = 0
    for index, cell in enumerate(notebook.get("cells", [])):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            headings.update(
                line.strip() for line in source.splitlines() if line.startswith("#")
            )
            continue
        if cell.get("cell_type") != "code":
            continue

        code_cell_count += 1
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise ValueError(f"Code cell {index} contains stored execution state.")
        compile(source, f"{NOTEBOOK_PATH.name}:cell-{index}", "exec")

    missing_headings = REQUIRED_HEADINGS - headings
    if missing_headings:
        raise ValueError(f"Missing notebook headings: {sorted(missing_headings)}")
    if code_cell_count == 0:
        raise ValueError("The notebook does not contain code cells.")


def validate_data_files() -> None:
    with zipfile.ZipFile(RAW_ARCHIVE_PATH) as archive:
        csv_members = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError("The raw archive must contain exactly one CSV file.")
        with archive.open(csv_members[0]) as binary_stream:
            with io.TextIOWrapper(binary_stream, encoding="utf-8-sig") as text_stream:
                raw_header = read_csv_header(text_stream)

    with PROCESSED_DATA_PATH.open(encoding="utf-8-sig", newline="") as stream:
        processed_header = read_csv_header(stream)

    if missing := RAW_COLUMNS - raw_header:
        raise ValueError(f"Raw dataset is missing columns: {sorted(missing)}")
    if missing := PROCESSED_COLUMNS - processed_header:
        raise ValueError(f"Processed dataset is missing columns: {sorted(missing)}")


def main() -> None:
    validate_notebook()
    validate_data_files()
    print("Notebook and dataset validation passed.")


if __name__ == "__main__":
    main()
