# FibroAI — Transformer Topic & Sentiment Modeling for Fibromyalgia Literature

This repository reproduces the AIAS 2025 paper:

**Unveiling Fibromyalgia Research Frontiers: Transformer-Based Topic and Sentiment Modeling for Biomedical Meta-Analysis**  
Yetunde O. Folajimi, Leonidas Deligiannidis, Salem Othman, Ibukun Folajimi

It provides an end-to-end, reproducible pipeline to:
- preprocess **5,861 PubMed abstracts (1990–2020)** from the Kaggle dataset by Phaterpekar (2019)
- generate **Sentence-BERT** embeddings
- discover topics with **BERTopic** (UMAP + HDBSCAN + c-TF-IDF)
- benchmark against **LDA** and **CTM** (optional) using topic coherence metrics
- run **DistilBERT SST-2** sentiment analysis and aggregate by topic
- generate paper-style figures (topic frequency, temporal trends, sentiment by topic, coherence comparison)

---

## Quickstart (local)

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords punkt wordnet omw-1.4
```

### 2) Download the Kaggle dataset
See `data/README.md` for instructions. Place the CSV at:
```
data/raw/pubmed_fibromyalgia.csv
```

### 3) Run the full pipeline
```bash
python -m src.run_all --data data/raw/pubmed_fibromyalgia.csv
```

Outputs:
- Figures: `outputs/figures/`
- Tables/CSVs: `outputs/tables/`
- Embeddings cache: `outputs/embeddings/`

---

## Reproducing key outputs

### BERTopic topics + assignments
```bash
python -m src.topics.run_bertopic --data data/raw/pubmed_fibromyalgia.csv
```

### Sentiment by topic
```bash
python -m src.sentiment.run_sentiment --topics outputs/tables/bertopic_doc_topics.csv
```

### Coherence benchmarking (LDA + BERTopic; CTM optional)
```bash
python -m src.eval.coherence --data data/raw/pubmed_fibromyalgia.csv
```

### Create figures
```bash
python -m src.plots.make_figures
```

---

## Notes on reproducibility
- Defaults aim to match the paper, but clustering/topic counts can vary slightly with hardware and library versions.
- The **CTM** baseline is optional because it has heavier dependencies; see `src/topics/run_ctm.py`.

---

## Project structure
- `src/` : modular pipeline code
- `notebooks/` : your original Colab notebook (archived for transparency)
- `outputs/` : generated artifacts (not committed except `.gitkeep`)

---

## Citation
If you use this code, please cite the paper and this repository.

### BibTeX
<!-- ```bibtex
@inproceedings{folajimi2025fibroai,
  title={Unveiling Fibromyalgia Research Frontiers: Transformer-Based Topic and Sentiment Modeling for Biomedical Meta-Analysis},
  author={Folajimi, Yetunde O. and Deligiannidis, Leonidas and Othman, Salem and Folajimi, Ibukun},
  booktitle={AIAS 2025},
  year={2025}
}
``` -->

---

## License
