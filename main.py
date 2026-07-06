import torch


def check_device():
    print("=" * 50)
    print("Cross-Modal Satellite Image Retrieval")
    print("=" * 50)

    print(f"PyTorch Version : {torch.__version__}")

    if torch.cuda.is_available():
        print("Device          : GPU")
        print(f"GPU Name        : {torch.cuda.get_device_name(0)}")
    else:
        print("Device          : CPU")


if __name__ == "__main__":
    check_device()