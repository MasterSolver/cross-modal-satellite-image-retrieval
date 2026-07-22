from utils.preprocessing import Modality
from torch.utils.data import DataLoader
from pathlib import Path
from models.feature_extractor import extract_embeddings
import torch
import numpy as np
from tqdm import tqdm




def generate_embeddings(dataset, modality, model, device):

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    all_embeddings = []

    all_s1_paths = []
    all_s2_paths = []

    for s1, s2, labels, s1_paths, s2_paths in tqdm(dataloader, desc=f"Generating {modality.upper()} embeddings"):
        if modality == Modality.SENTINEL_1:
            image_tensor = s1
        elif modality == Modality.SENTINEL_2:
            image_tensor = s2

        all_s1_paths.extend(s1_paths)
        all_s2_paths.extend(s2_paths)

        embeddings = extract_embeddings(image_tensor, model, device)
        all_embeddings.append(embeddings.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Final embedding tensor shape: {all_embeddings.shape}")
    all_embeddings = all_embeddings.numpy()
    print("Converted to NumPy.")

    if modality == Modality.SENTINEL_1:
        np.save(Path(r"outputs\embeddings\s1.npy"), all_embeddings)
        print("Embeddings saved successfully.")
    elif modality == Modality.SENTINEL_2:
        np.save(Path(r"outputs\embeddings\s2.npy"), all_embeddings)
        print("Embeddings saved successfully.")
    else:
        raise ValueError(f"Wrong modality: {modality}")
    np.save(
        "outputs/embeddings/s1_paths.npy",
        np.array(all_s1_paths, dtype=object)
    )

    np.save(
        "outputs/embeddings/s2_paths.npy",
        np.array(all_s2_paths, dtype=object)
    )

    return all_embeddings
        


        
