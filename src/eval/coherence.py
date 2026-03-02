from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora

from src.data.load_data import load_kaggle_csv
from src.preprocess.clean_text import preprocess_texts
from src.utils.io import ensure_dir

def coherence_for_topics(tokenized_texts: list[list[str]], topics_words: list[list[str]], dictionary: corpora.Dictionary) -> dict:
    # gensim supports: c_v, u_mass, c_uci, c_npmi
    metrics = {}
    for coh in ["c_v", "u_mass", "c_uci", "c_npmi"]:
        cm = CoherenceModel(topics=topics_words, texts=tokenized_texts, dictionary=dictionary, coherence=coh)
        metrics[coh] = cm.get_coherence()
    return metrics

def load_bertopic_topics(path: str | Path, topn_words: int = 10) -> list[list[str]]:
    df = pd.read_csv(path)
    topics = []
    for _, row in df.iterrows():
        words = [w.strip() for w in str(row["top_words"]).split(",")][:topn_words]
        topics.append(words)
    return topics

def load_lda_topics(path: str | Path, topn_words: int = 10) -> list[list[str]]:
    df = pd.read_csv(path)
    topics = []
    for _, row in df.iterrows():
        words = [w.strip() for w in str(row["top_words"]).split(",")][:topn_words]
        topics.append(words)
    return topics

def run(args: argparse.Namespace) -> None:
    df, _ = load_kaggle_csv(args.data, text_col=args.text_col, year_col=args.year_col, date_col=args.date_col)
    df["text_clean"] = preprocess_texts(df["abstract_raw"].tolist(), spacy_model=args.spacy_model)

    tokenized = [t.split() for t in df["text_clean"].tolist()]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    out_dir = ensure_dir(args.out_dir)
    rows = []

    # BERTopic topics
    bertop_words = load_bertopic_topics(Path(out_dir) / "bertopic_top_words.csv", topn_words=args.topn_words) if (Path(out_dir)/"bertopic_top_words.csv").exists() else None
    if bertop_words:
        m = coherence_for_topics(tokenized, bertop_words, dictionary)
        rows.append({"model":"BERTopic", **m})

    # LDA topics
    lda_words = load_lda_topics(Path(out_dir) / "lda_topics.csv", topn_words=args.topn_words) if (Path(out_dir)/"lda_topics.csv").exists() else None
    if lda_words:
        m = coherence_for_topics(tokenized, lda_words, dictionary)
        rows.append({"model":"LDA", **m})

    pd.DataFrame(rows).to_csv(Path(out_dir) / "coherence_summary.csv", index=False)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", default="outputs/tables")
    p.add_argument("--topn_words", type=int, default=10)
    p.add_argument("--spacy_model", default="en_core_web_sm")
    p.add_argument("--text_col", default=None)
    p.add_argument("--year_col", default=None)
    p.add_argument("--date_col", default=None)
    return p

if __name__ == "__main__":
    run(build_parser().parse_args())
