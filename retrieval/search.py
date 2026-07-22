import faiss
import numpy as np


def search(index, query_embedding, k):
    if not isinstance(index, faiss.Index):
        raise TypeError("Index is not a faiss index")
    if index.ntotal == 0:
        raise ValueError("Index should not be Empty")
    
    if not isinstance(query_embedding, np.ndarray):
        raise TypeError(
            "Query embedding must be a NumPy array."
        )
    
    if query_embedding.size == 0:
        raise ValueError(
            "Query embedding cannot be empty."
        )
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    elif query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
        raise ValueError(
            "Query embedding must have shape (D,) or (1, D)."
        )
    if query_embedding.dtype != np.float32:
        query_embedding = query_embedding.astype(np.float32)
    if query_embedding.shape[1] != index.d:
        raise ValueError(
            f"Query embedding dimension ({query_embedding.shape[1]}) "
            f"does not match index dimension ({index.d})."
        )
    
    query_embedding = np.ascontiguousarray(query_embedding)


    if not isinstance(k, int):
        raise TypeError("k must be an integer.")

    if k <= 0:
        raise ValueError("k must be greater than 0.")
    k = min(k, index.ntotal)

    distances, indices = index.search(query_embedding, k)
    return distances, indices