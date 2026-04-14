from __future__ import annotations

"""Merge raw CRMLS monthly files into separate unfiltered listing and sold datasets."""

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


FILENAME_PATTERN = re.compile(r"^CRMLS(?P<dataset_type>Listing|Sold)(?P<period>\d{6})\.csv$")
METADATA_COLUMNS = ("source_file", "file_period", "sort_date")
START_PERIOD = "202401"
SORT_COLUMNS = {
    "Listing": ("ListingContractDate", "ContractStatusChangeDate", "PurchaseContractDate", "CloseDate"),
    "Sold": ("CloseDate", "PurchaseContractDate", "ContractStatusChangeDate", "ListingContractDate"),
}
DEFAULT_OUTPUT_FILES = {
    "Listing": "CRMLSListingMasterUnfiltered.csv",
    "Sold": "CRMLSSoldMasterUnfiltered.csv",
}


@dataclass(frozen=True)
class SourceFile:
    path: Path
    dataset_type: str
    period: str


def normalize_headers(headers: list[str]) -> list[str]:
    """Make duplicate CSV headers unique so they can be loaded safely."""
    counts: dict[str, int] = defaultdict(int)
    normalized: list[str] = []

    for header in headers:
        base_name = (header or "column").strip() or "column"
        key = base_name.lower()
        counts[key] += 1

        if counts[key] == 1:
            normalized.append(base_name)
        else:
            normalized.append(f"{base_name}__dup{counts[key]}")

    return normalized


def read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read one CSV while preserving all columns, including duplicated headers."""
    with csv_path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.reader(handle)

        try:
            raw_headers = next(reader)
        except StopIteration:
            return [], []

        headers = normalize_headers(raw_headers)
        rows: list[dict[str, str]] = []

        for raw_row in reader:
            if not raw_row:
                continue

            if len(raw_row) < len(headers):
                raw_row = raw_row + [""] * (len(headers) - len(raw_row))
            elif len(raw_row) > len(headers):
                raw_row = raw_row[: len(headers)]

            rows.append({header: raw_row[index] for index, header in enumerate(headers)})

    return headers, rows


def derive_sort_date(row: dict[str, str], dataset_type: str, file_period: str) -> str:
    """Pick the best available date column for stable chronological sorting."""
    for column in SORT_COLUMNS[dataset_type]:
        value = (row.get(column) or "").strip()
        if not value:
            continue
        return value[:10]

    return f"{file_period[:4]}-{file_period[4:6]}-01"


def row_sort_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    """Sort by month first, then row-level date and listing identifiers."""
    return (
        row.get("file_period", ""),
        row.get("sort_date", ""),
        row.get("ListingKey", row.get("ListingKeyNumeric", "")),
        row.get("ListingId", ""),
    )


def merge_header_order(existing_headers: list[str], new_headers: list[str]) -> list[str]:
    """Keep metadata columns first, then preserve file column order."""
    ordered_headers: list[str] = []
    seen: set[str] = set()

    for header in METADATA_COLUMNS:
        if header not in seen:
            ordered_headers.append(header)
            seen.add(header)

    for header in existing_headers + new_headers:
        if header in seen:
            continue
        ordered_headers.append(header)
        seen.add(header)

    return ordered_headers


def collect_source_files(raw_path: Path, start_period: str, end_period: str) -> list[SourceFile]:
    """Collect raw listing and sold files between the requested periods."""
    sources: list[SourceFile] = []

    if not raw_path.exists():
        return sources

    for csv_path in sorted(raw_path.glob("*.csv")):
        match = FILENAME_PATTERN.match(csv_path.name)
        if not match:
            continue

        period = match.group("period")
        if period < start_period or period > end_period:
            continue

        sources.append(
            SourceFile(
                path=csv_path.resolve(),
                dataset_type=match.group("dataset_type"),
                period=period,
            )
        )

    return sorted(sources, key=lambda item: (item.dataset_type, item.period, item.path.name.lower()))


def latest_available_period(raw_path: Path, start_period: str) -> str | None:
    """Return the latest YYYYMM period available in raw/."""
    periods: list[str] = []

    for csv_path in raw_path.glob("*.csv"):
        match = FILENAME_PATTERN.match(csv_path.name)
        if not match:
            continue

        period = match.group("period")
        if period >= start_period:
            periods.append(period)

    return max(periods) if periods else None


def build_dataset_dataframe(dataset_type: str, sources: list[SourceFile]) -> pd.DataFrame:
    """Build one unfiltered dataframe for Listing or Sold."""
    dataset_headers: list[str] = []
    dataset_rows: list[dict[str, str]] = []

    for source in sources:
        if source.dataset_type != dataset_type:
            continue

        headers, rows = read_csv_rows(source.path)
        for header in headers:
            if header not in dataset_headers:
                dataset_headers.append(header)

        for row in rows:
            row["source_file"] = source.path.name
            row["file_period"] = source.period
            row["sort_date"] = derive_sort_date(row, dataset_type, source.period)
            dataset_rows.append(row)

    dataset_rows.sort(key=row_sort_key)
    ordered_columns = merge_header_order([], dataset_headers)
    dataframe = pd.DataFrame(dataset_rows)

    if dataframe.empty:
        return pd.DataFrame(columns=ordered_columns)

    final_columns = ordered_columns + [column for column in dataframe.columns if column not in ordered_columns]
    return dataframe.reindex(columns=final_columns)


def write_dataframe_csv(output_path: Path, dataframe: pd.DataFrame) -> None:
    """Write a dataframe to CSV while preserving empty strings."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.fillna("").to_csv(output_path, index=False, encoding="utf-8-sig")


def merge_raw_crmls_data_unfiltered(
    raw_dir: str | Path = "raw",
    output_dir: str | Path | None = "data",
    start_period: str = START_PERIOD,
    end_period: str | None = None,
    write_csv: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge raw monthly CRMLS files into separate unfiltered listing and sold datasets."""
    project_root = Path(__file__).resolve().parents[2]
    raw_path = Path(raw_dir)
    if not raw_path.is_absolute():
        raw_path = (project_root / raw_path).resolve()

    effective_end_period = end_period or latest_available_period(raw_path, start_period)
    if effective_end_period is None:
        empty_listing = pd.DataFrame(columns=list(METADATA_COLUMNS))
        empty_sold = pd.DataFrame(columns=list(METADATA_COLUMNS))
        return empty_listing, empty_sold

    sources = collect_source_files(raw_path, start_period=start_period, end_period=effective_end_period)
    listing_df = build_dataset_dataframe("Listing", sources)
    sold_df = build_dataset_dataframe("Sold", sources)

    if write_csv and output_dir is not None:
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = (project_root / output_path).resolve()

        write_dataframe_csv(output_path / DEFAULT_OUTPUT_FILES["Listing"], listing_df)
        write_dataframe_csv(output_path / DEFAULT_OUTPUT_FILES["Sold"], sold_df)

    return listing_df, sold_df
