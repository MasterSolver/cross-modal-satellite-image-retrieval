from preprocessing import *
from pathlib import Path

image_path = Path(r"data\BEN-14K\benv1_14k\s2\S2A_MSIL2A_20170803T094031_28_15")

image = load_sample(image_path, Modality.SENTINEL_2)
print("Loaded shape:", image.shape)
print("Loaded dtype:", image.dtype)
print("Min:", image.min())
print("Max:", image.max())
print("Mean:", image.mean())
print("Std:", image.std())

# bands = select_bands(image, Modality.SENTINEL_2)
# print("Selected shape:", bands.shape)
# print("Selected dtype:", bands.dtype)

tensor = to_tensor(image)
print("Tensor shape:", tensor.shape)
print("Tensor dtype:", tensor.dtype)

resized = resize_tensor(tensor)
print("Resized shape:", resized.shape)
print("Resized dtype:", resized.dtype)

print("Before normalization:", tensor.min(), tensor.max())

normalized = normalize_tensor(resized, Modality.SENTINEL_2)

print("After normalization:", normalized.min(), normalized.max())