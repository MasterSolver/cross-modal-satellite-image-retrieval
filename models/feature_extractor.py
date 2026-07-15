import torch


def extract_embeddings(image_tensor, model, device):
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    elif image_tensor.ndim != 4:
        raise ValueError("Expected 3 or 4-dimensional input...")
    image_tensor = image_tensor.to(device)



    with torch.inference_mode():
        outputs = model(image_tensor)

    embeddings = outputs.last_hidden_state[:, 0, :]

    return embeddings