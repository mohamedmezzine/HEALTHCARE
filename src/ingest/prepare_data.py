"""Load, clean, and write MIMIC-III tables for offline RL use."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

try:
    from .schema import (
        EXPECTED_COLUMNS,
        assert_valid_columns,
        coerce_datetime_columns,
        warn_deidentified_dob,
    )
except ImportError:  # Allows running as a script without package context
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.ingest.schema import (
        EXPECTED_COLUMNS,
        assert_valid_columns,
        coerce_datetime_columns,
        warn_deidentified_dob,
    )


RL_CORE_TABLES = {
    "ADMISSIONS.csv",
    "ICUSTAYS.csv",
    "PATIENTS.csv",
    "CHARTEVENTS.csv",
    "LABEVENTS.csv",
    "INPUTEVENTS_CV.csv",
    "INPUTEVENTS_MV.csv",
    "OUTPUTEVENTS.csv",
    "PRESCRIPTIONS.csv",
    "DIAGNOSES_ICD.csv",
    "PROCEDURES_ICD.csv",
}


def _list_csv_files(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source directory not found: {source_dir}. "
            "Pass --source with the folder containing CSVs."
        )
    return sorted([p for p in source_dir.iterdir() if p.suffix.lower() == ".csv"])


def _clean_dataframe(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Basic schema validation and ordering
    assert_valid_columns(df.columns, table_name)
    df = df[EXPECTED_COLUMNS[table_name]]

    # Trim whitespace and normalize empty strings to NA for object columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace({"": pd.NA})

    # Parse any date/time-like columns
    df = coerce_datetime_columns(df)
    warn_deidentified_dob(df, table_name)

    # Drop exact duplicates and fully empty rows
    df = df.drop_duplicates()
    df = df.dropna(how="all")

    return df


def _write_dataframe(
    df: pd.DataFrame, dest_dir: Path, table_name: str, fmt: str
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = table_name.rsplit(".", 1)[0]
    if fmt == "parquet":
        out_path = dest_dir / f"{stem}.parquet"
        df.to_parquet(out_path, index=False)
        return out_path
    if fmt == "csv":
        out_path = dest_dir / f"{stem}.csv"
        df.to_csv(out_path, index=False)
        return out_path
    raise ValueError(f"Unknown format: {fmt}")


def prepare_tables(source_dir: Path, dest_dir: Path, mode: str, fmt: str) -> list[Path]:
    csv_files = _list_csv_files(source_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")

    if mode == "rl_core":
        csv_files = [p for p in csv_files if p.name in RL_CORE_TABLES]

    outputs = []
    for path in csv_files:
        table_name = path.name
        df = pd.read_csv(path)
        df = _clean_dataframe(df, table_name)
        out_path = _write_dataframe(df, dest_dir, table_name, fmt)
        outputs.append(out_path)
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data", help="Directory with raw CSVs")
    parser.add_argument(
        "--dest", default="artifacts/cleaned", help="Output directory for cleaned data"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "rl_core"],
        default="all",
        help="Process all tables or only RL-relevant core tables",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output file format",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_dir = Path(args.source.strip())
    dest_dir = Path(args.dest.strip())

    try:
        outputs = prepare_tables(source_dir, dest_dir, args.mode, args.format)
    except ImportError as exc:
        if args.format == "parquet":
            raise ImportError(
                "Parquet output requires pyarrow or fastparquet. "
                "Install one or rerun with --format csv."
            ) from exc
        raise

    print(f"Wrote {len(outputs)} cleaned tables to {os.path.abspath(dest_dir)}")


if __name__ == "__main__":
    main()
