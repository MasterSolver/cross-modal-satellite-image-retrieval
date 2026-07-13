from transformers import AutoModel
import torch


def load_dinov2():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model = model.to(device)
    model.eval()

    return model, device
