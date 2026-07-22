import numpy as np
from pathlib import Path

from retrieval.faiss_index import build_index, save_index

# Sentinel-1
s1_embeddings = np.load(Path("outputs/embeddings/s1.npy"))
s1_index = build_index(s1_embeddings)
save_index(s1_index, Path("outputs/faiss_index/s1.index"))

# Sentinel-2
s2_embeddings = np.load(Path("outputs/embeddings/s2.npy"))
s2_index = build_index(s2_embeddings)
save_index(s2_index, Path("outputs/faiss_index/s2.index"))

print("✅ Both FAISS indices created successfully!")