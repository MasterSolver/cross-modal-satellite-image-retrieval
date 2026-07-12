from pathlib import Path
from utils import dataset
from utils import preprocessing
from torch.utils.data import DataLoader

dataset = dataset.BEN14KDataset(Path(r"data\BEN-14K\benv1_14k\benv1_14k_dataset_master_labels.csv"), Path(r"data\BEN-14K\benv1_14k\s1"), Path(r"data\BEN-14K\benv1_14k\s2"), preprocessing.preprocess)

dataloader = DataLoader(
    dataset,
    batch_size = 16,
    shuffle = True
)

for s1, s2, labels in dataloader:
    print(s1.shape)
    print(s2.shape)
    print(labels)
    break