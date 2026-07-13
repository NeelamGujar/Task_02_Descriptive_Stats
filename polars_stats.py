"""
Research Task 02
Descriptive Statistics and Grouped Analysis Using Polars

This script analyzes the Task 02 Facebook political advertising
dataset using Polars.
"""

from pathlib import Path

import polars as pl


DATA_FILE = Path("Data/task02_facebook_ads.csv")
OUTPUT_DIR = Path("Output")


def shorten_text(value, maximum_length=60):
    """Shorten long values for readable terminal output."""

    if value is None:
        return "N/A"

    text = str(value)

    if len(text) <= maximum_length:
        return text

    return text[: maximum_length - 3] + "..."


def load_dataset(file_path):
    """Load the CSV dataset using Polars."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found.\nExpected location: {file_path.resolve()}"
        )

    return pl.read_csv(
        file_path,
        infer_schema_length=10000,
        null_values=[
            "",
            "NA",
            "N/A",
            "null",
            "None",
            "nan",
            "missing",
        ],
    )


def get_numeric_columns(df):
    """Return all numeric columns."""

    numeric_types = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }

    return [
        column
        for column, dtype in df.schema.items()
        if dtype in numeric_types
    ]


def get_non_numeric_columns(df):
    """Return all non-numeric columns."""

    numeric_columns = set(get_numeric_columns(df))

    return [
        column
        for column in df.columns
        if column not in numeric_columns
    ]


def print_dataset_overview(df):
    """Print dataset dimensions and column names."""

    print("=" * 70)
    print("POLARS DATASET OVERVIEW")
    print("=" * 70)

    print(f"Dataset: {DATA_FILE.name}")
    print(f"Total rows: {df.height:,}")
    print(f"Total columns: {df.width:,}")

    print("\nCOLUMN NAMES")

    for index, column in enumerate(df.columns, start=1):
        print(f"{index}. {column}")


def print_schema(df):
    """Print Polars schema information."""

    print("\n" + "=" * 70)
    print("POLARS DATA TYPES")
    print("=" * 70)

    for column, dtype in df.schema.items():
        print(f"{column:50}{str(dtype):>20}")


def print_missing_value_report(df):
    """Print null counts and percentages."""

    print("\n" + "=" * 70)
    print("MISSING VALUE REPORT")
    print("=" * 70)

    print(
        f"{'Column':45}"
        f"{'Missing Count':>15}"
        f"{'Missing %':>10}"
    )

    print("-" * 70)

    total_rows = df.height

    for column in df.columns:
        missing_count = df[column].null_count()

        missing_percentage = (
            missing_count / total_rows * 100
            if total_rows > 0
            else 0.0
        )

        print(
            f"{column:45}"
            f"{missing_count:>15,}"
            f"{missing_percentage:>9.2f}%"
        )


def print_numeric_statistics(df):
    """Print descriptive statistics for numeric columns."""

    numeric_columns = get_numeric_columns(df)

    print("\n" + "=" * 135)
    print("NUMERIC COLUMN STATISTICS")
    print("=" * 135)

    header = (
        f"{'Column':45}"
        f"{'Count':>14}"
        f"{'Mean':>16}"
        f"{'Min':>14}"
        f"{'Max':>14}"
        f"{'Std Dev':>16}"
        f"{'Median':>14}"
    )

    print(header)
    print("-" * 135)

    for column in numeric_columns:
        stats = df.select(
            pl.col(column).count().alias("count"),
            pl.col(column).mean().alias("mean"),
            pl.col(column).min().alias("minimum"),
            pl.col(column).max().alias("maximum"),
            pl.col(column).std(ddof=1).alias("standard_deviation"),
            pl.col(column).median().alias("median"),
        ).row(0, named=True)

        std_value = stats["standard_deviation"]

        std_display = (
            f"{std_value:,.4f}"
            if std_value is not None
            else "N/A"
        )

        print(
            f"{column:45}"
            f"{stats['count']:>14,}"
            f"{stats['mean']:>16,.4f}"
            f"{stats['minimum']:>14,.2f}"
            f"{stats['maximum']:>14,.2f}"
            f"{std_display:>16}"
            f"{stats['median']:>14,.2f}"
        )


def print_polars_describe(df):
    """Print Polars describe output."""

    print("\n" + "=" * 100)
    print("POLARS DESCRIBE")
    print("=" * 100)

    print(df.describe())


def print_categorical_statistics(df):
    """Print descriptive statistics for non-numeric columns."""

    non_numeric_columns = get_non_numeric_columns(df)

    print("\n" + "=" * 110)
    print("CATEGORICAL AND DATE COLUMN STATISTICS")
    print("=" * 110)

    for column in non_numeric_columns:
        non_null_series = (
        df[column]
        .drop_nulls()
        .cast(pl.String)
        .str.strip_chars()
        )

        count = len(non_null_series)
        unique_count = non_null_series.n_unique()

        value_counts = (
            non_null_series
            .value_counts(sort=True)
            .head(5)
        )

        print(f"\nCOLUMN: {column}")
        print("-" * 110)

        print(f"Non-null count: {count:,}")
        print(f"Unique values: {unique_count:,}")

        if value_counts.height == 0:
            print("Mode: N/A")
            print("Mode frequency: 0")
            print("Top 5 values:")
            print("  No non-missing values")
            continue

        value_column = column
        frequency_column = "count"

        mode_value = value_counts[value_column][0]
        mode_frequency = value_counts[frequency_column][0]

        print(f"Mode: {shorten_text(mode_value)}")
        print(f"Mode frequency: {mode_frequency:,}")
        print("Top 5 values:")

        for rank, row in enumerate(
            value_counts.iter_rows(named=True),
            start=1
        ):
            value = row[value_column]
            frequency = row[frequency_column]

            percentage = (
                frequency / count * 100
                if count > 0
                else 0.0
            )

            print(
                f"  {rank}. {shorten_text(value)} "
                f"— {frequency:,} ({percentage:.2f}%)"
            )


def create_grouped_analysis(df):
    """Create grouped summaries by page and by page/ad."""

    numeric_columns = [
        "estimated_audience_size",
        "estimated_impressions",
        "estimated_spend",
    ]

    aggregation_expressions = []

    for column in numeric_columns:
        aggregation_expressions.extend(
            [
                pl.col(column).count().alias(f"{column}_count"),
                pl.col(column).mean().alias(f"{column}_mean"),
                pl.col(column).min().alias(f"{column}_min"),
                pl.col(column).max().alias(f"{column}_max"),
                pl.col(column).std(ddof=1).alias(f"{column}_std"),
                pl.col(column).median().alias(f"{column}_median"),
            ]
        )

    page_grouped = (
        df.group_by("page_id")
        .agg(
            pl.len().alias("row_count"),
            *aggregation_expressions,
        )
        .sort("page_id")
    )

    page_ad_grouped = (
        df.group_by(["page_id", "ad_id"])
        .agg(
            pl.len().alias("row_count"),
            *aggregation_expressions,
        )
        .sort(["page_id", "ad_id"])
    )

    return page_grouped, page_ad_grouped


def save_grouped_results(page_grouped, page_ad_grouped):
    """Save grouped Polars results."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    page_grouped.write_csv(
        OUTPUT_DIR / "polars_by_page_id.csv"
    )

    page_ad_grouped.write_csv(
        OUTPUT_DIR / "polars_by_page_id_ad_id.csv"
    )


def print_grouped_previews(page_grouped, page_ad_grouped):
    """Print previews of grouped analysis."""

    print("\n" + "=" * 100)
    print("POLARS GROUPED ANALYSIS BY PAGE_ID")
    print("=" * 100)

    print(page_grouped.head(10))

    print("\n" + "=" * 100)
    print("POLARS GROUPED ANALYSIS BY PAGE_ID + AD_ID")
    print("=" * 100)

    print(page_ad_grouped.head(10))


def main():
    """Run the complete Polars analysis."""

    df = load_dataset(DATA_FILE)

    print_dataset_overview(df)
    print_schema(df)
    print_missing_value_report(df)
    print_numeric_statistics(df)
    print_polars_describe(df)
    print_categorical_statistics(df)

    page_grouped, page_ad_grouped = create_grouped_analysis(df)

    save_grouped_results(
        page_grouped=page_grouped,
        page_ad_grouped=page_ad_grouped,
    )

    print_grouped_previews(
        page_grouped=page_grouped,
        page_ad_grouped=page_ad_grouped,
    )

    print("\nGrouped files created:")
    print("Output/polars_by_page_id.csv")
    print("Output/polars_by_page_id_ad_id.csv")


if __name__ == "__main__":
    main()