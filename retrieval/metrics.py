


def precision_at_k(retrieved, relevant):
    if len(retrieved) == 0:
        raise ValueError("Retrieved list cannot be empty.")
    if len(relevant) == 0:
        raise ValueError("Relevant list cannot be empty.")
    precision = len(set(retrieved) & set(relevant)) / len(retrieved)

    return precision

def recall_at_k(retrieved, relevant):
    if len(retrieved) == 0:
        raise ValueError("Retrieved list cannot be empty.")
    if len(relevant) == 0:
        raise ValueError("Relevant list cannot be empty.")
    recall = len(set(retrieved) & set(relevant)) / len(relevant)

    return recall

def f1_at_k(retrieved, relevant):
    precision = precision_at_k(retrieved, relevant)
    recall = recall_at_k(retrieved, relevant)

    if precision + recall == 0:
        return 0.0
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1