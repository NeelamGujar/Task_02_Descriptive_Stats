"""
Research Task 02
Descriptive Statistics and Grouped Analysis Using Pandas

This script analyzes the Task 02 Facebook political advertising
dataset using Pandas.
"""

from pathlib import Path

import pandas as pd


DATA_FILE = Path("Data/task02_facebook_ads.csv")


def shorten_text(value, maximum_length=60):
    """Shorten long text values for readable terminal output."""

    if pd.isna(value):
        return "N/A"

    text = str(value)

    if len(text) <= maximum_length:
        return text

    return text[: maximum_length - 3] + "..."


def load_dataset(file_path):
    """Load the dataset and validate that the file exists."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found.\nExpected location: {file_path.resolve()}"
        )

    return pd.read_csv(file_path)


def print_dataset_overview(df):
    """Print dataset dimensions and column information."""

    print("=" * 70)
    print("PANDAS DATASET OVERVIEW")
    print("=" * 70)

    print(f"Dataset: {DATA_FILE.name}")
    print(f"Total rows: {df.shape[0]:,}")
    print(f"Total columns: {df.shape[1]:,}")

    print("\nCOLUMN NAMES")

    for index, column in enumerate(df.columns, start=1):
        print(f"{index}. {column}")


def print_dataframe_information(df):
    """Display Pandas data types and DataFrame.info()."""

    print("\n" + "=" * 70)
    print("PANDAS DATA TYPES")
    print("=" * 70)

    print(df.dtypes)

    print("\n" + "=" * 70)
    print("DATAFRAME INFO")
    print("=" * 70)

    df.info()


def print_missing_value_report(df):
    """Print missing counts and percentages for each column."""

    missing_counts = df.isna().sum()
    missing_percentages = df.isna().mean() * 100

    print("\n" + "=" * 70)
    print("MISSING VALUE REPORT")
    print("=" * 70)

    print(
        f"{'Column':45}"
        f"{'Missing Count':>15}"
        f"{'Missing %':>10}"
    )

    print("-" * 70)

    for column in df.columns:
        print(
            f"{column:45}"
            f"{missing_counts[column]:>15,}"
            f"{missing_percentages[column]:>9.2f}%"
        )


def get_numeric_columns(df):
    """Return Pandas numeric columns."""

    return df.select_dtypes(include="number").columns.tolist()


def get_non_numeric_columns(df):
    """Return Pandas non-numeric columns."""

    numeric_columns = set(get_numeric_columns(df))

    return [
        column
        for column in df.columns
        if column not in numeric_columns
    ]


def print_numeric_statistics(df):
    """Print required statistics for all numeric columns."""

    numeric_columns = get_numeric_columns(df)

    print("\n" + "=" * 110)
    print("NUMERIC COLUMN STATISTICS")
    print("=" * 110)

    header = (
        f"{'Column':45}"
        f"{'Count':>12}"
        f"{'Mean':>10}"
        f"{'Min':>8}"
        f"{'Max':>8}"
        f"{'Std Dev':>12}"
        f"{'Median':>10}"
    )

    print(header)
    print("-" * 110)

    for column in numeric_columns:
        series = df[column]

        count = series.count()
        mean = series.mean()
        minimum = series.min()
        maximum = series.max()
        standard_deviation = series.std(ddof=1)
        median = series.median()

        print(
            f"{column:45}"
            f"{count:>12,}"
            f"{mean:>10.4f}"
            f"{minimum:>8.2f}"
            f"{maximum:>8.2f}"
            f"{standard_deviation:>12.4f}"
            f"{median:>10.2f}"
        )


def print_pandas_describe(df):
    """Print DataFrame.describe() for numeric and non-numeric columns."""

    print("\n" + "=" * 100)
    print("PANDAS DESCRIBE: NUMERIC COLUMNS")
    print("=" * 100)

    print(df.describe(include="number").to_string())

    print("\n" + "=" * 100)
    print("PANDAS DESCRIBE: NON-NUMERIC COLUMNS")
    print("=" * 100)

    print(df.describe(include=["object", "string"]).to_string())


def print_categorical_statistics(df):
    """Print statistics for non-numeric columns."""

    non_numeric_columns = get_non_numeric_columns(df)

    print("\n" + "=" * 110)
    print("CATEGORICAL AND DATE COLUMN STATISTICS")
    print("=" * 110)

    for column in non_numeric_columns:
        series = df[column].dropna().astype(str).str.strip()

        count = series.count()
        unique_count = series.nunique(dropna=True)
        value_counts = series.value_counts(dropna=True)

        print(f"\nCOLUMN: {column}")
        print("-" * 110)

        print(f"Non-null count: {count:,}")
        print(f"Unique values: {unique_count:,}")

        if value_counts.empty:
            print("Mode: N/A")
            print("Mode frequency: 0")
            print("Top 5 values:")
            print("  No non-missing values")
            continue

        mode_value = value_counts.index[0]
        mode_frequency = int(value_counts.iloc[0])

        print(f"Mode: {shorten_text(mode_value)}")
        print(f"Mode frequency: {mode_frequency:,}")
        print("Top 5 values:")

        for rank, (value, frequency) in enumerate(
            value_counts.head(5).items(),
            start=1
        ):
            percentage = frequency / count * 100 if count > 0 else 0.0

            print(
                f"  {rank}. {shorten_text(value)} "
                f"— {frequency:,} ({percentage:.2f}%)"
            )


def main():
    """Run the complete Pandas descriptive and grouped analysis."""

    df = load_dataset(DATA_FILE)

    print_dataset_overview(df)
    print_dataframe_information(df)
    print_missing_value_report(df)
    print_numeric_statistics(df)
    print_pandas_describe(df)
    print_categorical_statistics(df)

    numeric_columns = [
        "estimated_audience_size",
        "estimated_impressions",
        "estimated_spend",
    ]

    page_grouped = (
        df.groupby("page_id")[numeric_columns]
        .agg(["count", "mean", "min", "max", "std", "median"])
    )

    page_ad_grouped = (
        df.groupby(["page_id", "ad_id"])[numeric_columns]
        .agg(["count", "mean", "min", "max", "std", "median"])
    )

    output_dir = Path("Output")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    page_grouped.to_csv(
        output_dir / "pandas_by_page_id.csv"
    )

    page_ad_grouped.to_csv(
        output_dir / "pandas_by_page_id_ad_id.csv"
    )

    print("\n" + "=" * 100)
    print("PANDAS GROUPED ANALYSIS BY PAGE_ID")
    print("=" * 100)
    print(page_grouped.head(10).to_string())

    print("\n" + "=" * 100)
    print("PANDAS GROUPED ANALYSIS BY PAGE_ID + AD_ID")
    print("=" * 100)
    print(page_ad_grouped.head(10).to_string())

    print("\nGrouped files created:")
    print("Output/pandas_by_page_id.csv")
    print("Output/pandas_by_page_id_ad_id.csv")


if __name__ == "__main__":
    main()