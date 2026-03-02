from __future__ import annotations
import re
from typing import Iterable, List
import spacy
import nltk
from nltk.corpus import stopwords

def _ensure_nltk():
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.word_tokenize("test")
    except LookupError:
        nltk.download("punkt")

def get_nlp(model: str = "en_core_web_sm"):
    return spacy.load(model, disable=["ner", "parser"])

def preprocess_texts(texts: Iterable[str], spacy_model: str = "en_core_web_sm") -> List[str]:
    _ensure_nltk()
    nlp = get_nlp(spacy_model)
    sw = set(stopwords.words("english"))
    out = []
    for doc in nlp.pipe(texts, batch_size=64):
        toks = []
        for t in doc:
            if t.is_space or t.is_punct:
                continue
            lemma = t.lemma_.lower().strip()
            if not lemma or lemma in sw:
                continue
            if re.fullmatch(r"\d+", lemma):
                continue
            toks.append(lemma)
        out.append(" ".join(toks))
    return out
