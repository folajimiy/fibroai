from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.io import ensure_dir

def run(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.topics)
    # year may be missing in some rows
    df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # focus on top N topics excluding -1
    topic_counts = df[df["topic"] != -1]["topic"].value_counts().head(args.top_n)
    top_topics = topic_counts.index.tolist()

    trend = (
        df[df["topic"].isin(top_topics)]
        .groupby(["year","topic"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["year","topic"])
    )
    out_dir = ensure_dir(args.out_dir)
    trend.to_csv(Path(out_dir) / "topic_trends_by_year.csv", index=False)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--topics", required=True, help="outputs/tables/bertopic_doc_topics.csv")
    p.add_argument("--out_dir", default="outputs/tables")
    p.add_argument("--top_n", type=int, default=5)
    return p

if __name__ == "__main__":
    run(build_parser().parse_args())
