import torch


def extract_embeddings(image_tensor, model, device):
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)



    with torch.inference_mode():
        outputs = model(image_tensor)

    embeddings = outputs.last_hidden_state[:, 0, :]

    return embeddings