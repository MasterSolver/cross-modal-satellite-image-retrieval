import faiss
import numpy as np
from pathlib import Path

def build_index(embeddings):
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("Embeddings must be a NumPy array.")

    if embeddings.ndim!=2:
        raise ValueError("Embeddings must be a 2D NumPy array of shape (N, D).")
    
    if embeddings.shape[0]==0:
        raise ValueError("Embeddings array must contain at least one embedding.")
    
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    embeddings = np.ascontiguousarray(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index

def save_index(index, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(path))

def load_index(path):
    path = Path(path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"FAISS index file not found: {path}"
        )
    
    index = faiss.read_index(str(path))

    return index
