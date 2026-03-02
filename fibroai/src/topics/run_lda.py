from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from src.data.load_data import load_kaggle_csv
from src.preprocess.clean_text import preprocess_texts
from src.utils.io import ensure_dir

def train_lda(texts: list[str], num_topics: int, seed: int) -> tuple[LdaModel, list[list[str]], corpora.Dictionary, list]:
    tokenized = [t.split() for t in texts]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=seed,
        passes=10,
        chunksize=2000,
        alpha="auto",
        eta="auto",
    )
    return model, tokenized, dictionary, corpus

def grid_search_cv(texts: list[str], topic_range: range, seed: int) -> tuple[int, pd.DataFrame]:
    rows = []
    best_k, best_cv = None, -1e9
    for k in topic_range:
        model, tokenized, dictionary, corpus = train_lda(texts, k, seed)
        cm = CoherenceModel(model=model, texts=tokenized, dictionary=dictionary, coherence="c_v")
        cv = cm.get_coherence()
        rows.append({"num_topics": k, "c_v": cv})
        if cv > best_cv:
            best_cv, best_k = cv, k
    return best_k or topic_range.start, pd.DataFrame(rows)

def run(args: argparse.Namespace) -> None:
    df, _ = load_kaggle_csv(args.data, text_col=args.text_col, year_col=args.year_col, date_col=args.date_col)
    df["text_clean"] = preprocess_texts(df["abstract_raw"].tolist(), spacy_model=args.spacy_model)

    best_k, grid = grid_search_cv(df["text_clean"].tolist(), range(args.k_min, args.k_max + 1, args.k_step), args.seed)
    out_dir = ensure_dir(args.out_dir)
    grid.to_csv(Path(out_dir) / "lda_cv_grid.csv", index=False)

    model, tokenized, dictionary, corpus = train_lda(df["text_clean"].tolist(), best_k, args.seed)
    # Save topics
    topics = []
    for tid, words in model.show_topics(num_topics=best_k, num_words=args.topn_words, formatted=False):
        topics.append({"topic": tid, "top_words": ", ".join([w for w,_ in words])})
    pd.DataFrame(topics).to_csv(Path(out_dir) / "lda_topics.csv", index=False)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", default="outputs/tables")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k_min", type=int, default=10)
    p.add_argument("--k_max", type=int, default=80)
    p.add_argument("--k_step", type=int, default=5)
    p.add_argument("--topn_words", type=int, default=10)
    p.add_argument("--spacy_model", default="en_core_web_sm")
    p.add_argument("--text_col", default=None)
    p.add_argument("--year_col", default=None)
    p.add_argument("--date_col", default=None)
    return p

if __name__ == "__main__":
    run(build_parser().parse_args())
