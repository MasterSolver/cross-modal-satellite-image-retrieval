from pathlib import Path
import pandas as pd


def explore_dataset():
    print("=" * 60)
    print("BEN-14K Dataset Explorer")
    print("=" * 60)

    csv_path = Path(
        "data/BEN-14K/benv1_14k/benv1_14k_dataset_master_labels.csv"
    )

    if not csv_path.exists():
        print(f"CSV file not found:\n{csv_path}")
        return

    df = pd.read_csv(csv_path)

    print(f"\nDataset Shape : {df.shape}")
    print(f"Total Image Pairs : {len(df)}")

    print("\nColumns:")
    for column in df.columns:
        print(f" - {column}")

    print("\nFirst 5 Rows:")
    print(df.head())


if __name__ == "__main__":
    explore_dataset()