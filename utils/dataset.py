from torch.utils.data import Dataset
import pandas as pd
from .preprocessing import preprocess, Modality

class BEN14KDataset (Dataset):
    
    def __init__(self, csv_path, s1_root, s2_root, preprocess):
        self.data = pd.read_csv(csv_path)
        self.s1_root = s1_root
        self.s2_root = s2_root
        self.preprocessing = preprocess

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        s1_id = row["S1_ID"]
        s2_id = row["S2_ID"]
        labels = row["S1_Labels"]

        s1_path = self.s1_root / s1_id
        s2_path = self.s2_root / s2_id

        s1_tensor = self.preprocessing(s1_path, Modality.SENTINEL_1)
        s2_tensor = self.preprocessing(s2_path, Modality.SENTINEL_2)

        return s1_tensor, s2_tensor, labels, str(s1_path), str(s2_path)
