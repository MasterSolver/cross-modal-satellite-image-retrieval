from models.dinov2 import load_dinov2
from utils.embedding_generator import generate_embeddings
from utils.preprocessing import Modality, preprocess
from utils.dataset import BEN14KDataset
from pathlib import Path


if __name__ == "__main__":
    model, device = load_dinov2()

    dataset = BEN14KDataset(Path(r"data\BEN-14K\benv1_14k\benv1_14k_dataset_master_labels.csv"), Path(r"data\BEN-14K\benv1_14k\s1"), Path(r"data\BEN-14K\benv1_14k\s2"), preprocess)

    embeddings = generate_embeddings(dataset, Modality.SENTINEL_2, model, device)

    print(embeddings.shape)
    print(embeddings.dtype)