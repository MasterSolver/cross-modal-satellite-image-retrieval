import numpy as np
import rasterio
from pathlib import Path
from PIL import Image

from utils.preprocessing import Modality


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)

    image -= image.min()

    if image.max() > 0:
        image /= image.max()

    return image


def load_preview(folder_path: Path, modality: Modality):

    if modality == Modality.SENTINEL_2:

        red = rasterio.open(folder_path / "B04.tif").read(1)
        green = rasterio.open(folder_path / "B03.tif").read(1)
        blue = rasterio.open(folder_path / "B02.tif").read(1)

        rgb = np.dstack((red, green, blue))

    else:

        vv = rasterio.open(folder_path / "VV.tif").read(1)
        vh = rasterio.open(folder_path / "VH.tif").read(1)

        rgb = np.dstack((vv, vh, vv))

    rgb = normalize_image(rgb)

    rgb = Image.fromarray((rgb * 255).astype(np.uint8))
    rgb = rgb.resize((224, 224))

    return np.array(rgb)