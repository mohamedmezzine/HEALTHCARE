import pandas as pd
import os
from pathlib import Path

# ---------------------------------------
# Configuration
# ---------------------------------------
DATA_DIR = Path(".")  # current directory
OUTPUT_FILE = "mimic_columns_report.txt"

# ---------------------------------------
# Helper function
# ---------------------------------------
def inspect_csv(path):
    try:
        df = pd.read_csv(path, nrows=50)
        return {
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
    except Exception as e:
        return {
            "error": str(e)
        }

# ---------------------------------------
# Main inspection loop
# ---------------------------------------
csv_files = sorted(DATA_DIR.glob("*.csv"))

report_lines = []
report_lines.append(f"Found {len(csv_files)} CSV files\n")
report_lines.append("=" * 80 + "\n")

for csv_path in csv_files:
    report_lines.append(f"FILE: {csv_path.name}\n")
    report_lines.append("-" * 80 + "\n")

    info = inspect_csv(csv_path)

    if "error" in info:
        report_lines.append(f"ERROR reading file: {info['error']}\n\n")
        continue

    report_lines.append(f"Number of columns: {len(info['columns'])}\n\n")
    report_lines.append("COLUMNS:\n")
    for col in info["columns"]:
        report_lines.append(f"  - {col}\n")

    report_lines.append("\nDTYPES:\n")
    for col, dtype in info["dtypes"].items():
        report_lines.append(f"  {col}: {dtype}\n")

    report_lines.append("\n" + "=" * 80 + "\n\n")

# ---------------------------------------
# Write report
# ---------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines(report_lines)

print(f"Inspection complete. Report written to {OUTPUT_FILE}")
