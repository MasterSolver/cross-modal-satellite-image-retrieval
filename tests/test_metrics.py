from retrieval.metrics import f1_at_k

def test_f1_at_k():
    retrieved = [1, 2, 3, 4, 5]
    relevant = [2, 4, 8]

    assert f1_at_k(retrieved, relevant) == 0.5

print("All tests passed!")