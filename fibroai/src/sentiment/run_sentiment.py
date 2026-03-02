from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from src.utils.io import ensure_dir

def run(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.topics)
    clf = pipeline("sentiment-analysis", model=args.model, device=args.device)

    labels = []
    scores = []
    texts = df["abstract_raw"].astype(str).tolist()
    for out in tqdm(clf(texts, batch_size=args.batch_size, truncation=True), total=len(texts)):
        labels.append(out["label"])
        scores.append(out["score"])
    df["sentiment_label"] = labels
    df["sentiment_score"] = scores

    out_dir = ensure_dir(args.out_dir)
    df.to_csv(Path(out_dir) / "doc_topics_with_sentiment.csv", index=False)

    # Aggregate by topic
    agg = df.groupby("topic")["sentiment_label"].value_counts(normalize=True).rename("proportion").reset_index()
    agg.to_csv(Path(out_dir) / "sentiment_by_topic.csv", index=False)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--topics", required=True, help="Path to bertopic_doc_topics.csv")
    p.add_argument("--out_dir", default="outputs/tables")
    p.add_argument("--model", default="distilbert-base-uncased-finetuned-sst-2-english")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0 for GPU")
    return p

if __name__ == "__main__":
    run(build_parser().parse_args())
