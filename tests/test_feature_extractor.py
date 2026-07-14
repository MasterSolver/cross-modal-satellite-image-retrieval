from models.feature_extractor import extract_embeddings
from utils.preprocessing import preprocess, Modality
from pathlib import Path
from models.dinov2 import load_dinov2

image_tensor = preprocess(Path(r"data\BEN-14K\benv1_14k\s2\S2A_MSIL2A_20170803T094031_26_20"), Modality.SENTINEL_2)

model, device = load_dinov2()

embeddings = extract_embeddings(image_tensor, model, device)

print(embeddings.shape)
print(embeddings.device)
print(embeddings.dtype)