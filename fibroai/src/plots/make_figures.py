from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    fig_dir = Path("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir = Path("outputs/tables")

    # Figure 1-style: top 5 topic counts (BERTopic, excluding -1)
    doc_topics_path = tbl_dir / "bertopic_doc_topics.csv"
    if doc_topics_path.exists():
        df = pd.read_csv(doc_topics_path)
        counts = df[df["topic"] != -1]["topic"].value_counts().head(5).sort_index()
        plt.figure()
        counts.plot(kind="bar")
        plt.xlabel("Topic ID")
        plt.ylabel("Abstract Count")
        plt.title("Top 5 BERTopic Topics by Abstract Count")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig1_top_topics.png", dpi=300)
        plt.close()

    # Figure 3-style: sentiment distribution by topic for top 5 topics
    sent_path = tbl_dir / "sentiment_by_topic.csv"
    if sent_path.exists():
        s = pd.read_csv(sent_path)
        # keep top topics by doc count
        if doc_topics_path.exists():
            top_topics = df[df["topic"] != -1]["topic"].value_counts().head(5).index.tolist()
            s = s[s["topic"].isin(top_topics)]
        pivot = s.pivot_table(index="topic", columns="sentiment_label", values="proportion", fill_value=0)
        plt.figure()
        pivot.plot(kind="bar")
        plt.xlabel("Topic ID")
        plt.ylabel("Proportion of Abstracts")
        plt.title("Sentiment Distribution by Topic")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig3_sentiment_by_topic.png", dpi=300)
        plt.close()

    # Figure 4-style: coherence comparison
    coh_path = tbl_dir / "coherence_summary.csv"
    if coh_path.exists():
        c = pd.read_csv(coh_path)
        if "c_v" in c.columns and len(c):
            plt.figure()
            plt.bar(c["model"], c["c_v"])
            plt.xlabel("Model")
            plt.ylabel("C_V Coherence")
            plt.title("Topic Coherence (C_V) Comparison")
            plt.tight_layout()
            plt.savefig(fig_dir / "fig4_coherence_cv.png", dpi=300)
            plt.close()

if __name__ == "__main__":
    main()
