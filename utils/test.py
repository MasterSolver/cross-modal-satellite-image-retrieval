from pathlib import Path

dataset_path = Path(r"data\BEN-14K\benv1_14k")

for item in dataset_path.iterdir():
    print(item.name)

    keywords = ["train", "val", "test", "split", "partition"]

for item in dataset_path.rglob("*"):
    if any(keyword in item.name.lower() for keyword in keywords):
        print(item)