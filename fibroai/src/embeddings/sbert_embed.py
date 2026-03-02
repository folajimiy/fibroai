from __future__ import annotations
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.io import ensure_dir

def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=normalize)
    return emb

def save_embeddings(emb: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, emb)

def load_embeddings(path: str | Path) -> np.ndarray:
    return np.load(path)
