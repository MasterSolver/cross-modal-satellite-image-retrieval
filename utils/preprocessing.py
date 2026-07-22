import rasterio
from pathlib import Path
import numpy as np
import torch
import torchvision

class Modality:
    SENTINEL_1 = "sentinel1"
    SENTINEL_2 = "sentinel2"

def load_image(image_path: Path) -> np.ndarray:
    """
    Load a GeoTIFF image and return it as a NumPy array.
    """
    
    if not image_path.exists():
        raise FileNotFoundError (f"Image not Found: {image_path}")

    with rasterio.open(image_path) as ds:

        image = ds.read(1)
        # bands = image.shape[0]
        return image
    
def load_sample(folder_path: Path, modality: str) -> np.ndarray:
    """
    find correct image file to load.
    """
    if modality == Modality.SENTINEL_2:
        redpath = next(folder_path.glob("*_B04.tif"))
        greenpath = next(folder_path.glob("*_B03.tif"))
        bluepath = next(folder_path.glob("*_B02.tif"))

        red_image = load_image(redpath)
        green_image = load_image(greenpath)
        blue_image = load_image(bluepath)

        image =  np.stack([red_image, green_image, blue_image])

    elif modality == Modality.SENTINEL_1:
        vvpath = next(folder_path.glob("*_VV.tif"))
        vhpath = next(folder_path.glob("*_VH.tif"))

        vvimage = load_image(vvpath)
        vhimage = load_image(vhpath)

        image = np.stack([vvimage, vhimage, vvimage])
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    return image

def to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Changing NumPy array to tensor
    """
    tensor_data = torch.from_numpy(image)
    tensor_data = tensor_data.float()
    return tensor_data

def resize_tensor(tensor_data: torch.Tensor) -> torch.Tensor:
    """
    resizing tensor to the target spatial dimensions.
    """
    resize = torchvision.transforms.Resize(size = (224,224), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
    resized_tensor = resize(tensor_data)
    return resized_tensor

def normalize_tensor(tensor: torch.Tensor, modality: str) -> torch.Tensor:
    """
    Normalizing the image make it visible to model
    """
    if modality == Modality.SENTINEL_2:
        tensor = tensor / 10000
    elif modality == Modality.SENTINEL_1:
        tensor = (tensor + 40) / 50
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    
    tensor = torch.clamp(tensor, min = 0, max = 1)
    return tensor

def preview(folder_path: Path, modality: str) -> np.ndarray:

    image = load_sample(folder_path, modality)

    image = image.astype(np.float32)

    if modality == Modality.SENTINEL_2:

        # Channels are [R, G, B]
        for i in range(3):
            band = image[i]

            p2 = np.percentile(band, 2)
            p98 = np.percentile(band, 98)

            band = np.clip(band, p2, p98)
            band = (band - p2) / (p98 - p2 + 1e-6)

            image[i] = band

    else:
        image = (image + 40) / 50
        image = np.clip(image, 0, 1)

    tensor = torch.from_numpy(image).float()
    tensor = resize_tensor(tensor)

    return tensor.permute(1, 2, 0).numpy()

def preprocess(folder_path: Path, modality: str) -> torch.Tensor:
    image = load_sample(folder_path, modality)
    tensor = to_tensor(image)
    resized = resize_tensor(tensor)
    result = normalize_tensor(resized, modality)
    return result