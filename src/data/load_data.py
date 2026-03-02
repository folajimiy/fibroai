from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import re

from src.utils.io import read_csv_flexible, infer_column

@dataclass
class DatasetSpec:
    text_col: str
    year_col: str | None = None
    date_col: str | None = None

YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")

def load_kaggle_csv(path: str | Path, text_col: str | None = None, year_col: str | None = None, date_col: str | None = None) -> tuple[pd.DataFrame, DatasetSpec]:
    df = read_csv_flexible(path)

    # Infer columns if not provided
    if text_col is None:
        text_col = infer_column(df, ["abstract", "Abstract", "ABSTRACT", "summary", "text"])
    if text_col is None:
        raise ValueError(f"Could not infer abstract/text column. Available columns: {list(df.columns)}")

    if year_col is None and date_col is None:
        year_col = infer_column(df, ["year", "Year", "pub_year", "publication_year"])
        if year_col is None:
            date_col = infer_column(df, ["date", "pubdate", "publication_date", "created", "year"])
    spec = DatasetSpec(text_col=text_col, year_col=year_col, date_col=date_col)

    # Normalize year
    if spec.year_col and spec.year_col in df.columns:
        df["year"] = pd.to_numeric(df[spec.year_col], errors="coerce")
    elif spec.date_col and spec.date_col in df.columns:
        years = df[spec.date_col].astype(str).str.extract(YEAR_RE)[0]
        df["year"] = pd.to_numeric(years, errors="coerce")
    else:
        df["year"] = pd.NA

    df["abstract_raw"] = df[spec.text_col].astype(str)
    df = df.dropna(subset=["abstract_raw"]).reset_index(drop=True)
    return df, spec
