from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

def run_cmd(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--text_col", default=None)
    p.add_argument("--year_col", default=None)
    p.add_argument("--date_col", default=None)
    args = p.parse_args()

    base_args = []
    if args.text_col: base_args += ["--text_col", args.text_col]
    if args.year_col: base_args += ["--year_col", args.year_col]
    if args.date_col: base_args += ["--date_col", args.date_col]

    run_cmd([sys.executable, "-m", "src.topics.run_bertopic", "--data", args.data] + base_args)
    run_cmd([sys.executable, "-m", "src.sentiment.run_sentiment", "--topics", "outputs/tables/bertopic_doc_topics.csv"])
    run_cmd([sys.executable, "-m", "src.topics.run_lda", "--data", args.data] + base_args)
    run_cmd([sys.executable, "-m", "src.eval.coherence", "--data", args.data] + base_args)
    run_cmd([sys.executable, "-m", "src.temporal.trends", "--topics", "outputs/tables/bertopic_doc_topics.csv"])
    run_cmd([sys.executable, "-m", "src.plots.make_figures"])

if __name__ == "__main__":
    main()
