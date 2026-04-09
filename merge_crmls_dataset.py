from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


FILENAME_PATTERN = re.compile(r"^CRMLS(?P<dataset_type>Listing|Sold)(?P<period>\d{6})\.csv$")
BOOTSTRAP_DIRS = ("raw", "new_data")
DEFAULT_UPDATE_DIRS = (".",)
METADATA_COLUMNS = ("source_file", "file_period", "sort_date")
START_PERIOD = "202401"
MASTER_FILES = {
    "Listing": Path("data/CRMLSListingMaster.csv"),
    "Sold": Path("data/CRMLSSoldMaster.csv"),
}
SORT_COLUMNS = {
    "Listing": ("ListingContractDate", "ContractStatusChangeDate", "PurchaseContractDate", "CloseDate"),
    "Sold": ("CloseDate", "PurchaseContractDate", "ContractStatusChangeDate", "ListingContractDate"),
}


@dataclass(frozen=True)
class SourceFile:
    path: Path
    dataset_type: str
    period: str
    priority: int


def normalize_headers(headers: list[str]) -> list[str]:
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


def resolve_dir(project_root: Path, directory: str) -> Path:
    path = Path(directory)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def most_recent_completed_period() -> str:
    today = datetime.today()
    year = today.year
    month = today.month - 1

    if month == 0:
        year -= 1
        month = 12

    return f"{year}{month:02d}"


def collect_source_files(project_root: Path, directories: tuple[str, ...], end_period: str) -> list[SourceFile]:
    selected: dict[tuple[str, str], SourceFile] = {}

    for priority, directory in enumerate(directories, start=1):
        scan_dir = resolve_dir(project_root, directory)
        if not scan_dir.exists():
            print(f"[skip] directory not found: {scan_dir}")
            continue

        for csv_path in sorted(scan_dir.glob("*.csv")):
            match = FILENAME_PATTERN.match(csv_path.name)
            if not match:
                continue

            dataset_type = match.group("dataset_type")
            period = match.group("period")
            if period < START_PERIOD or period > end_period:
                continue

            key = (dataset_type, period)
            source = SourceFile(
                path=csv_path.resolve(),
                dataset_type=dataset_type,
                period=period,
                priority=priority,
            )

            existing = selected.get(key)
            if existing is None or source.priority >= existing.priority:
                selected[key] = source

    return sorted(selected.values(), key=lambda item: (item.dataset_type, item.period, item.path.name.lower()))


def read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
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

            row = {header: raw_row[index] for index, header in enumerate(headers)}
            rows.append(row)

    return headers, rows


def read_master(master_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not master_path.exists():
        return [], []

    with master_path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [dict(row) for row in reader]

    return fieldnames, rows


def derive_sort_date(row: dict[str, str], dataset_type: str, file_period: str) -> str:
    for column in SORT_COLUMNS[dataset_type]:
        value = (row.get(column) or "").strip()
        if not value:
            continue

        date_text = value[:10]
        try:
            return datetime.strptime(date_text, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            continue

    return f"{file_period[:4]}-{file_period[4:6]}-01"


def merge_header_order(existing_headers: list[str], new_headers: list[str]) -> list[str]:
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


def row_sort_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("file_period", ""),
        row.get("sort_date", ""),
        row.get("ListingKey", row.get("ListingKeyNumeric", "")),
        row.get("ListingId", ""),
    )


def write_master(master_path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    master_path.parent.mkdir(parents=True, exist_ok=True)

    with master_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def build_incoming_rows(
    sources: list[SourceFile], dataset_type: str
) -> tuple[list[str], list[dict[str, str]], set[str], int]:
    incoming_headers: list[str] = []
    incoming_rows: list[dict[str, str]] = []
    incoming_periods: set[str] = set()
    source_row_count = 0

    for source in sources:
        if source.dataset_type != dataset_type:
            continue

        headers, rows = read_csv_rows(source.path)
        if not headers and not rows:
            print(f"[skip] empty file: {source.path.name}")
            continue

        for header in headers:
            if header not in incoming_headers:
                incoming_headers.append(header)

        source_row_count += len(rows)
        normalized_rows: list[dict[str, str]] = []
        for row in rows:
            row["source_file"] = source.path.name
            row["file_period"] = source.period
            row["sort_date"] = derive_sort_date(row, dataset_type, source.period)
            normalized_rows.append(row)

        normalized_rows.sort(key=row_sort_key)
        incoming_rows.extend(normalized_rows)
        incoming_periods.add(source.period)

        print(f"[load] {source.path.name}: {len(normalized_rows)} rows")

    return incoming_headers, incoming_rows, incoming_periods, source_row_count


def filter_residential_only(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if (row.get("PropertyType") or "").strip().lower() == "residential"]


def update_master_for_type(master_path: Path, dataset_type: str, sources: list[SourceFile]) -> None:
    incoming_headers, incoming_rows, incoming_periods, source_row_count = build_incoming_rows(sources, dataset_type)

    if not incoming_rows and not master_path.exists():
        print(f"[skip] no {dataset_type.lower()} source files found")
        return

    existing_headers, existing_rows = read_master(master_path)

    if incoming_periods:
        existing_rows = [row for row in existing_rows if row.get("file_period", "") not in incoming_periods]

    merged_rows = existing_rows + incoming_rows
    merged_rows.sort(key=row_sort_key)

    # Submission checkpoint: confirm row counts before and after concatenation.
    print(f"[count] {dataset_type} rows before concatenation: {len(existing_rows)} existing + {source_row_count} incoming")
    print(f"[count] {dataset_type} rows after concatenation: {len(merged_rows)}")

    # Submission checkpoint: confirm row counts before and after the Residential-only filter.
    print(f"[count] {dataset_type} rows before Residential filter: {len(merged_rows)}")
    merged_rows = filter_residential_only(merged_rows)
    print(f"[count] {dataset_type} rows after Residential filter: {len(merged_rows)}")

    output_headers = merge_header_order(existing_headers, incoming_headers)
    write_master(master_path, output_headers, merged_rows)

    print(
        f"[write] {master_path.name}: {len(merged_rows)} total rows "
        f"({len(incoming_rows)} rows refreshed from {len(incoming_periods)} month files)"
    )


def masters_exist(project_root: Path) -> bool:
    return all((project_root / master_file).exists() for master_file in MASTER_FILES.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize or update separate CRMLS listing/sold master datasets. "
            "First run bootstraps from raw + new_data. Later runs update from the project root."
        )
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory. Default: current directory.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a full rebuild from raw and new_data even if master files already exist.",
    )
    parser.add_argument(
        "--scan-dir",
        action="append",
        default=[],
        help=(
            "Extra directory to scan during update mode. "
            "Can be used multiple times if new monthly CSVs are not in the project root."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    end_period = most_recent_completed_period()

    bootstrap = args.rebuild or not masters_exist(project_root)
    if bootstrap:
        scan_dirs = BOOTSTRAP_DIRS
        print(f"[mode] bootstrap from raw + new_data ({START_PERIOD} to {end_period})")
    else:
        scan_dirs = DEFAULT_UPDATE_DIRS + tuple(args.scan_dir)
        print(f"[mode] update from: {', '.join(scan_dirs)} ({START_PERIOD} to {end_period})")

    sources = collect_source_files(project_root, scan_dirs, end_period=end_period)
    if not sources:
        print("[done] no matching CRMLS CSV files were found")
        return

    for dataset_type, master_file in MASTER_FILES.items():
        update_master_for_type(project_root / master_file, dataset_type, sources)


if __name__ == "__main__":
    main()
