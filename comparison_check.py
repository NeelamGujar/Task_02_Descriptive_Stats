"""
Research Task 02
Comparison Check for Pure Python, Pandas, and Polars

This script verifies that grouped numerical results from all three
approaches match within a small floating-point tolerance.
"""

from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("Output")

PURE_PAGE_FILE = OUTPUT_DIR / "pure_python_by_page_id.csv"
PANDAS_PAGE_FILE = OUTPUT_DIR / "pandas_by_page_id.csv"
POLARS_PAGE_FILE = OUTPUT_DIR / "polars_by_page_id.csv"

PURE_PAGE_AD_FILE = OUTPUT_DIR / "pure_python_by_page_id_ad_id.csv"
PANDAS_PAGE_AD_FILE = OUTPUT_DIR / "pandas_by_page_id_ad_id.csv"
POLARS_PAGE_AD_FILE = OUTPUT_DIR / "polars_by_page_id_ad_id.csv"

TOLERANCE = 1e-6

NUMERIC_COLUMNS = [
    "estimated_audience_size",
    "estimated_impressions",
    "estimated_spend",
]

STATISTICS = [
    "count",
    "mean",
    "min",
    "max",
    "std",
    "median",
]


def validate_files_exist(file_paths):
    """Raise an error when any required output file is missing."""

    missing_files = [
        file_path
        for file_path in file_paths
        if not file_path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(
            str(file_path)
            for file_path in missing_files
        )

        raise FileNotFoundError(
            f"Required comparison files are missing:\n{missing_text}"
        )

def flatten_pandas_grouped_file(file_path, grouping_columns):
    """
    Load a Pandas grouped CSV with two header rows, restore the
    grouping columns, and remove the extra index-name metadata row.
    """

    df = pd.read_csv(
        file_path,
        header=[0, 1]
    )

    flattened_columns = []

    for top_level, second_level in df.columns:
        top_level = str(top_level).strip()
        second_level = str(second_level).strip()

        if (
            top_level.startswith("Unnamed")
            or second_level.startswith("Unnamed")
        ):
            flattened_columns.append("temporary_group_column")
        else:
            flattened_columns.append(
                f"{top_level}_{second_level}"
            )

    # Restore the grouped index column names.
    for index, grouping_column in enumerate(grouping_columns):
        flattened_columns[index] = grouping_column

    df.columns = flattened_columns

    # Remove the extra Pandas index-name row written to the CSV.
    metadata_mask = pd.Series(True, index=df.index)

    for grouping_column in grouping_columns:
        metadata_mask &= (
            df[grouping_column].astype(str).str.strip()
            == grouping_column
        )

    df = df.loc[~metadata_mask].copy()

    return df

def load_standard_grouped_file(file_path):
    """Load a normal single-header grouped CSV."""

    return pd.read_csv(file_path)


def standardize_dataframe(df, grouping_columns):
    """Sort rows and ensure numeric comparison columns are numeric."""

    for column in df.columns:
        if column not in grouping_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return (
        df.sort_values(grouping_columns)
        .reset_index(drop=True)
    )


def compare_row_counts(
    pure_df,
    pandas_df,
    polars_df,
    label,
):
    """Compare the total number of grouped rows."""

    counts = {
        "Pure Python": len(pure_df),
        "Pandas": len(pandas_df),
        "Polars": len(polars_df),
    }

    print(f"\n{label} GROUP COUNTS")
    print("-" * 70)

    for approach, count in counts.items():
        print(f"{approach:15}: {count:,}")

    if len(set(counts.values())) == 1:
        print("Result: PASS — group counts match")
        return True

    print("Result: FAIL — group counts do not match")
    return False


def compare_group_keys(
    pure_df,
    pandas_df,
    polars_df,
    grouping_columns,
    label,
):
    """Verify that all approaches contain the same group keys."""

    pure_keys = pure_df[grouping_columns].astype(str)
    pandas_keys = pandas_df[grouping_columns].astype(str)
    polars_keys = polars_df[grouping_columns].astype(str)

    pure_matches_pandas = pure_keys.equals(pandas_keys)
    pure_matches_polars = pure_keys.equals(polars_keys)

    print(f"\n{label} GROUP KEYS")
    print("-" * 70)

    if pure_matches_pandas and pure_matches_polars:
        print("Result: PASS — all group keys match")
        return True

    print("Result: FAIL — group keys differ")
    return False


def values_match(left, right):
    """Compare values while treating matching missing values as equal."""

    if pd.isna(left) and pd.isna(right):
        return True

    if pd.isna(left) or pd.isna(right):
        return False

    return abs(float(left) - float(right)) <= TOLERANCE


def compare_numeric_results(
    pure_df,
    pandas_df,
    polars_df,
    label,
):
    """Compare grouped numeric statistics across all three approaches."""

    mismatch_count = 0
    checked_count = 0

    print(f"\n{label} NUMERIC STATISTICS")
    print("-" * 70)

    for numeric_column in NUMERIC_COLUMNS:
        for statistic in STATISTICS:
            column = f"{numeric_column}_{statistic}"

            if column not in pure_df.columns:
                print(f"Missing from Pure Python: {column}")
                mismatch_count += 1
                continue

            if column not in pandas_df.columns:
                print(f"Missing from Pandas: {column}")
                mismatch_count += 1
                continue

            if column not in polars_df.columns:
                print(f"Missing from Polars: {column}")
                mismatch_count += 1
                continue

            for row_index in range(len(pure_df)):
                pure_value = pure_df.at[row_index, column]
                pandas_value = pandas_df.at[row_index, column]
                polars_value = polars_df.at[row_index, column]

                checked_count += 1

                if not values_match(pure_value, pandas_value):
                    mismatch_count += 1

                    if mismatch_count <= 10:
                        print(
                            f"Mismatch in {column}, row {row_index}: "
                            f"Pure Python={pure_value}, "
                            f"Pandas={pandas_value}"
                        )

                if not values_match(pure_value, polars_value):
                    mismatch_count += 1

                    if mismatch_count <= 10:
                        print(
                            f"Mismatch in {column}, row {row_index}: "
                            f"Pure Python={pure_value}, "
                            f"Polars={polars_value}"
                        )

    print(f"Values checked: {checked_count:,}")

    if mismatch_count == 0:
        print("Result: PASS — all numeric statistics match")
        return True

    print(f"Result: FAIL — {mismatch_count:,} mismatches found")
    return False


def compare_grouped_level(
    pure_file,
    pandas_file,
    polars_file,
    grouping_columns,
    label,
):
    """Run all checks for one grouping level."""

    pure_df = load_standard_grouped_file(pure_file)

    pandas_df = flatten_pandas_grouped_file(
        pandas_file,
        grouping_columns
    )

    polars_df = load_standard_grouped_file(polars_file)

    pure_df = standardize_dataframe(
        pure_df,
        grouping_columns
    )

    pandas_df = standardize_dataframe(
        pandas_df,
        grouping_columns
    )

    polars_df = standardize_dataframe(
        polars_df,
        grouping_columns
    )

    row_count_pass = compare_row_counts(
        pure_df,
        pandas_df,
        polars_df,
        label
    )

    key_pass = compare_group_keys(
        pure_df,
        pandas_df,
        polars_df,
        grouping_columns,
        label
    )

    numeric_pass = compare_numeric_results(
        pure_df,
        pandas_df,
        polars_df,
        label
    )

    return row_count_pass and key_pass and numeric_pass


def main():
    """Run grouped result comparisons."""

    required_files = [
        PURE_PAGE_FILE,
        PANDAS_PAGE_FILE,
        POLARS_PAGE_FILE,
        PURE_PAGE_AD_FILE,
        PANDAS_PAGE_AD_FILE,
        POLARS_PAGE_AD_FILE,
    ]

    validate_files_exist(required_files)

    print("=" * 80)
    print("TASK 02 GROUPED RESULT COMPARISON")
    print("=" * 80)

    page_pass = compare_grouped_level(
        pure_file=PURE_PAGE_FILE,
        pandas_file=PANDAS_PAGE_FILE,
        polars_file=POLARS_PAGE_FILE,
        grouping_columns=["page_id"],
        label="PAGE_ID"
    )

    page_ad_pass = compare_grouped_level(
        pure_file=PURE_PAGE_AD_FILE,
        pandas_file=PANDAS_PAGE_AD_FILE,
        polars_file=POLARS_PAGE_AD_FILE,
        grouping_columns=["page_id", "ad_id"],
        label="PAGE_ID + AD_ID"
    )

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    if page_pass and page_ad_pass:
        print(
            "PASS — Pure Python, Pandas, and Polars grouped "
            "results match."
        )
    else:
        print(
            "FAIL — one or more grouped comparisons did not match."
        )


if __name__ == "__main__":
    main()