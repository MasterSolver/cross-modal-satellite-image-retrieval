import numpy as np

from retrieval.faiss_index import build_index
from retrieval.search import search

embeddings = np.random.rand(100,768).astype(np.float32)

index = build_index(embeddings)

query = np.random.rand(768).astype(np.float32)

distances, indices = search(index, query, 5)

print(distances)
print(indices)