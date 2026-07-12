from pathlib import Path
from utils import dataset
from utils import preprocessing

dataset = dataset.BEN14KDataset(Path(r"data\BEN-14K\benv1_14k\benv1_14k_dataset_master_labels.csv"), Path(r"data\BEN-14K\benv1_14k\s1"), Path(r"data\BEN-14K\benv1_14k\s2"), preprocessing.preprocess)
s1_tensor, s2_tensor, labels = dataset[0]
print(len(dataset))
print(s1_tensor.shape)
print(s2_tensor.shape)
print(labels)