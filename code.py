"""Pandas DataFrame demo script

This small script shows common DataFrame operations:
- create a sample DataFrame
- read from CSV
- summarize basic stats
- filter rows
- add a computed/category column
- save results to CSV

Run as: python code.py
"""

from __future__ import annotations

import datetime
import os
from typing import Optional

import numpy as np
import pandas as pd


def create_sample_df(n: int = 12) -> pd.DataFrame:
    """Create a simple sample DataFrame for demonstrations.

    Columns:
    - id: integer
    - name: string
    - value: float
    - date: ISO date string
    """
    rng = np.random.default_rng(seed=42)
    df = pd.DataFrame(
        {
            "id": range(1, n + 1),
            "name": [f"item_{i}" for i in range(1, n + 1)],
            "value": np.round(rng.normal(loc=50, scale=15, size=n), 2),
            "date": [
                (datetime.date.today() - datetime.timedelta(days=int(x))).isoformat()
                for x in rng.integers(0, 365, size=n)
            ],
        }
    )
    # Introduce a couple of NaNs for demo
    if n >= 3:
        df.loc[1, "value"] = np.nan
        df.loc[2, "name"] = None
    return df


def read_csv_to_df(path: str) -> pd.DataFrame:
    """Read CSV at `path` into a DataFrame (simple wrapper)."""
    return pd.read_csv(path)


def summarize_df(df: pd.DataFrame) -> None:
    """Print quick summary info for a DataFrame."""
    print("===== DataFrame info =====")
    print(df.info())
    print("\n===== Head =====")
    print(df.head())
    print("\n===== Describe (numeric) =====")
    print(df.describe(include=[np.number]))
    print("\n===== Missing values per column =====")
    print(df.isna().sum())


def add_category_column(df: pd.DataFrame, col: str = "value") -> pd.DataFrame:
    """Add a categorical 'category' column based on `col` numeric ranges.

    Returns a new DataFrame (does not modify the original in-place).
    """
    out = df.copy()
    # Fill NaNs with a sentinel so we can categorize them separately
    out["_val_filled"] = out[col].fillna(-9999)
    out["category"] = pd.cut(
        out["_val_filled"],
        bins=[-10000, 30, 60, 1e9],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )
    out.loc[out["_val_filled"] == -9999, "category"] = "missing"
    out.drop(columns="_val_filled", inplace=True)
    return out


def filter_by_min_value(df: pd.DataFrame, min_value: float) -> pd.DataFrame:
    """Return rows where 'value' >= min_value (ignores NaNs)."""
    return df[df["value"] >= min_value].copy()


def save_df(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV (creates parent dirs if needed)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved DataFrame to: {path}")


def main(sample_output: Optional[str] = "sample_output.csv") -> None:
    """Demo runner for the pandas DataFrame examples."""
    print("Creating sample DataFrame...")
    df = create_sample_df(12)
    summarize_df(df)

    print("\nAdding category column based on 'value'...")
    df2 = add_category_column(df, "value")
    print(df2[["id", "name", "value", "category"]].head())

    print("\nFiltering rows with value >= 50...")
    filtered = filter_by_min_value(df2, 50)
    print(filtered.head())

    print("\nSaving filtered results...")
    save_df(filtered, sample_output)


if __name__ == "__main__":
    # Default run: produce sample_output.csv in current directory
    main("sample_output.csv")
