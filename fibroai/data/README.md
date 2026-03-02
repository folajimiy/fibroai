# Data (Kaggle)

This project reproduces the paper using the Kaggle dataset curated by **Tejas Phaterpekar (2019)**:
**PubMed fibromyalgia article abstracts dataset**.

Because Kaggle datasets may have redistribution restrictions, the raw CSV is **not** committed to this repository.

## Option A — Manual download (recommended)
1. Download from Kaggle (search: `pubmed-fibromyalgiaarticle-abstracts`)
2. Extract the CSV
3. Place it at:
```
data/raw/pubmed_fibromyalgia.csv
```

## Option B — Kaggle API (if you have it configured)
1. Install Kaggle CLI and authenticate (`~/.kaggle/kaggle.json`)
2. Run:
```bash
kaggle datasets download -d tphaterp/pubmed-fibromyalgiaarticle-abstracts -p data/raw --unzip
```
3. Rename/move the main CSV to:
```
data/raw/pubmed_fibromyalgia.csv
```

## Expected columns
The loader is flexible, but you should have at least:
- an abstract text column (e.g., `abstract`, `Abstract`, `ABSTRACT`)
- a publication year column OR a date column that includes the year

If the script can’t infer columns automatically, pass them explicitly:
```bash
python -m src.run_all --data data/raw/pubmed_fibromyalgia.csv --text_col Abstract --year_col Year
```
