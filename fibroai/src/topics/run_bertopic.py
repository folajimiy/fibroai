from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from bertopic import BERTopic
from umap import UMAP
import hdbscan

from src.data.load_data import load_kaggle_csv
from src.preprocess.clean_text import preprocess_texts
from src.embeddings.sbert_embed import embed_texts, save_embeddings, load_embeddings
from src.utils.io import ensure_dir

def run(args: argparse.Namespace) -> None:
    df, spec = load_kaggle_csv(args.data, text_col=args.text_col, year_col=args.year_col, date_col=args.date_col)

    # Preprocess
    df["text_clean"] = preprocess_texts(df["abstract_raw"].tolist(), spacy_model=args.spacy_model)

    # Embeddings (cached)
    emb_path = Path(args.embeddings_path)
    if args.use_cached_embeddings and emb_path.exists():
        embeddings = load_embeddings(emb_path)
    else:
        embeddings = embed_texts(df["abstract_raw"].tolist(), model_name=args.sbert_model, batch_size=args.batch_size, normalize=True)
        save_embeddings(embeddings, emb_path)

    umap_model = UMAP(
        n_neighbors=args.umap_n_neighbors,
        n_components=args.umap_n_components,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        metric=args.hdbscan_metric,
        cluster_selection_method=args.hdbscan_selection_method,
        prediction_data=True,
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(df["abstract_raw"].tolist(), embeddings)

    out_dir = ensure_dir(args.out_dir)
    # Save doc-topic assignments
    doc_topics = df[["abstract_raw","year"]].copy()
    doc_topics["topic"] = topics
    doc_topics.to_csv(out_dir / "bertopic_doc_topics.csv", index=False)

    # Save topic info
    info = topic_model.get_topic_info()
    info.to_csv(out_dir / "bertopic_topic_info.csv", index=False)

    # Save top words per topic
    rows = []
    for tid in info["Topic"].tolist():
        if tid == -1:
            continue
        words = topic_model.get_topic(tid) or []
        rows.append({"topic": tid, "top_words": ", ".join([w for w,_ in words[:args.topn_words]])})
    pd.DataFrame(rows).to_csv(out_dir / "bertopic_top_words.csv", index=False)

    # Save model (optional; can be large)
    if args.save_model:
        model_dir = ensure_dir(Path(args.model_dir))
        topic_model.save(model_dir / "bertopic_model", serialization="safetensors", save_ctfidf=True, save_embedding_model=args.save_embedding_model)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to Kaggle CSV")
    p.add_argument("--out_dir", default="outputs/tables")
    p.add_argument("--model_dir", default="outputs/models")
    p.add_argument("--embeddings_path", default="outputs/embeddings/sbert_embeddings.npy")
    p.add_argument("--use_cached_embeddings", action="store_true")
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--save_embedding_model", default=False, action="store_true")

    # column overrides
    p.add_argument("--text_col", default=None)
    p.add_argument("--year_col", default=None)
    p.add_argument("--date_col", default=None)

    # preprocessing
    p.add_argument("--spacy_model", default="en_core_web_sm")

    # embeddings
    p.add_argument("--sbert_model", default="all-MiniLM-L6-v2")
    p.add_argument("--batch_size", type=int, default=64)

    # UMAP/HDBSCAN defaults (tweak as needed)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--umap_n_neighbors", type=int, default=15)
    p.add_argument("--umap_n_components", type=int, default=5)
    p.add_argument("--umap_min_dist", type=float, default=0.0)
    p.add_argument("--umap_metric", default="cosine")

    p.add_argument("--hdbscan_min_cluster_size", type=int, default=15)
    p.add_argument("--hdbscan_min_samples", type=int, default=None)
    p.add_argument("--hdbscan_metric", default="euclidean")
    p.add_argument("--hdbscan_selection_method", default="eom")

    p.add_argument("--topn_words", type=int, default=10)
    return p

if __name__ == "__main__":
    run(build_parser().parse_args())
