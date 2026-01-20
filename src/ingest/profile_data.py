"""Generate data profiling report for cleaned MIMIC-III tables."""

import argparse
from pathlib import Path

import pandas as pd


def profile_table(path: Path, table_name: str) -> dict:
    """Generate profiling statistics for a single table."""
    df = pd.read_parquet(path)

    profile = {
        "table_name": table_name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
        "missing_stats": {},
        "dtypes": df.dtypes.value_counts().to_dict(),
    }

    # Missing value analysis
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            profile["missing_stats"][col] = {
                "count": int(missing_count),
                "percent": float(missing_count / len(df) * 100),
            }

    return profile


def profile_rl_core_tables(data_dir: Path) -> dict:
    """Profile the RL-relevant core tables."""
    rl_tables = [
        "ADMISSIONS",
        "ICUSTAYS",
        "PATIENTS",
        "CHARTEVENTS",
        "LABEVENTS",
        "INPUTEVENTS_CV",
        "INPUTEVENTS_MV",
        "OUTPUTEVENTS",
        "PRESCRIPTIONS",
        "DIAGNOSES_ICD",
        "PROCEDURES_ICD",
    ]

    profiles = {}
    for table_stem in rl_tables:
        parquet_path = data_dir / f"{table_stem}.parquet"
        if parquet_path.exists():
            profiles[table_stem] = profile_table(parquet_path, table_stem)
        else:
            profiles[table_stem] = {"error": f"File not found: {parquet_path}"}

    return profiles


def generate_summary_report(profiles: dict) -> str:
    """Generate human-readable summary report."""
    lines = ["=" * 80, "MIMIC-III Data Profiling Report", "=" * 80, ""]

    total_rows = sum(p.get("row_count", 0) for p in profiles.values())
    total_memory = sum(p.get("memory_mb", 0) for p in profiles.values())

    lines.append(f"Total tables: {len(profiles)}")
    lines.append(f"Total rows: {total_rows:,}")
    lines.append(f"Total memory: {total_memory:.2f} MB")
    lines.append("")

    for table_name, profile in profiles.items():
        if "error" in profile:
            lines.append(f"\n[{table_name}] ERROR: {profile['error']}")
            continue

        lines.append(f"\n[{table_name}]")
        lines.append(f"  Rows: {profile['row_count']:,}")
        lines.append(f"  Columns: {profile['column_count']}")
        lines.append(f"  Memory: {profile['memory_mb']:.2f} MB")

        if profile["missing_stats"]:
            lines.append(f"  Missing data in {len(profile['missing_stats'])} columns:")
            # Show top 5 columns with most missing
            sorted_missing = sorted(
                profile["missing_stats"].items(),
                key=lambda x: x[1]["percent"],
                reverse=True,
            )[:5]
            for col, stats in sorted_missing:
                lines.append(f"    - {col}: {stats['percent']:.1f}% ({stats['count']:,} rows)")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def compute_rl_readiness(data_dir: Path) -> dict:
    """Compute RL-specific readiness metrics."""
    icustays = pd.read_parquet(data_dir / "ICUSTAYS.parquet")
    admissions = pd.read_parquet(data_dir / "ADMISSIONS.parquet")
    chartevents = pd.read_parquet(data_dir / "CHARTEVENTS.parquet")

    # Count ICU stays with data
    icustay_ids = set(icustays["icustay_id"].unique())
    icustays_with_charts = set(chartevents["icustay_id"].dropna().unique())

    # Try loading input events
    try:
        input_cv = pd.read_parquet(data_dir / "INPUTEVENTS_CV.parquet")
        input_mv = pd.read_parquet(data_dir / "INPUTEVENTS_MV.parquet")
        icustays_with_inputs = set(input_cv["icustay_id"].dropna().unique()) | set(
            input_mv["icustay_id"].dropna().unique()
        )
    except FileNotFoundError:
        icustays_with_inputs = set()

    complete_episodes = icustay_ids & icustays_with_charts & icustays_with_inputs

    # Merge for outcomes
    icu_with_outcomes = icustays.merge(admissions, on="hadm_id", how="inner")
    mortality_rate = icu_with_outcomes["hospital_expire_flag"].mean()

    readiness = {
        "total_icustays": len(icustay_ids),
        "icustays_with_vitals": len(icustays_with_charts),
        "icustays_with_treatments": len(icustays_with_inputs),
        "complete_episodes": len(complete_episodes),
        "completeness_rate": len(complete_episodes) / len(icustay_ids) * 100,
        "mortality_rate": mortality_rate * 100,
        "avg_los_days": float(icustays["los"].mean()),
    }

    return readiness


def print_rl_readiness(readiness: dict):
    """Print RL readiness report."""
    print("\n" + "=" * 80)
    print("RL Pipeline Readiness Assessment")
    print("=" * 80)
    print(f"\nTotal ICU stays (episodes): {readiness['total_icustays']:,}")
    print(
        f"  With vitals/charts: {readiness['icustays_with_vitals']:,} "
        f"({readiness['icustays_with_vitals']/readiness['total_icustays']*100:.1f}%)"
    )
    print(
        f"  With treatments: {readiness['icustays_with_treatments']:,} "
        f"({readiness['icustays_with_treatments']/readiness['total_icustays']*100:.1f}%)"
    )
    print(
        f"  Complete (vitals + treatments): {readiness['complete_episodes']:,} "
        f"({readiness['completeness_rate']:.1f}%)"
    )
    print(f"\nOutcome distribution:")
    print(f"  Hospital mortality rate: {readiness['mortality_rate']:.1f}%")
    print(f"  Average ICU length of stay: {readiness['avg_los_days']:.2f} days")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="artifacts/cleaned",
        help="Directory with cleaned parquet files",
    )
    parser.add_argument(
        "--output", type=Path, help="Optional file to save report (default: print to stdout)"
    )
    args = parser.parse_args()

    print(f"Profiling data in: {args.data_dir.resolve()}")

    # Generate profiles
    profiles = profile_rl_core_tables(args.data_dir)
    report = generate_summary_report(profiles)

    # Compute RL readiness
    try:
        readiness = compute_rl_readiness(args.data_dir)
    except FileNotFoundError as e:
        print(f"\nWarning: Could not compute RL readiness: {e}")
        readiness = None

    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
            if readiness:
                f.write("\n\n")
                # Redirect print to file
                import sys
                from io import StringIO

                old_stdout = sys.stdout
                sys.stdout = StringIO()
                print_rl_readiness(readiness)
                f.write(sys.stdout.getvalue())
                sys.stdout = old_stdout
        print(f"\nReport saved to: {args.output.resolve()}")
    else:
        print(report)
        if readiness:
            print_rl_readiness(readiness)


if __name__ == "__main__":
    main()
