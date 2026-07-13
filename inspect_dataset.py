"""
Task 02 Dataset Inspection

This script inspects the structure, data types, missing values,
and sample records of the Task 02 Facebook political ads dataset.
"""

from pathlib import Path

import pandas as pd


DATA_FILE = Path("Data/task02_facebook_ads.csv")


def main() -> None:
    """Load and inspect the Task 02 dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE.resolve()}"
        )

    df = pd.read_csv(DATA_FILE)

    print("=" * 80)
    print("TASK 02 DATASET OVERVIEW")
    print("=" * 80)

    print(f"Dataset: {DATA_FILE.name}")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    print("\nCOLUMN NAMES")
    print("-" * 80)

    for index, column in enumerate(df.columns, start=1):
        print(f"{index}. {column}")

    print("\nDATA TYPES")
    print("-" * 80)
    print(df.dtypes)

    print("\nMISSING VALUES")
    print("-" * 80)

    missing_report = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percentage": df.isna().mean() * 100,
        }
    )

    print(
        missing_report
        .sort_values("missing_count", ascending=False)
        .to_string()
    )

    print("\nFIRST FIVE ROWS")
    print("-" * 80)
    print(df.head().to_string())


if __name__ == "__main__":
    main()