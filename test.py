import numpy as np
from pathlib import Path


DATA_DIR = Path("artifacts/data_transformed")


for split in ["train", "val", "test"]:

    split_dir = DATA_DIR / split
    files = sorted(split_dir.glob("*.npz"))

    print("\n" + "=" * 60)
    print(f"{split.upper()}")
    print("=" * 60)

    print("Number of batch files:", len(files))

    total_samples = 0
    all_labels = []

    for file in files:

        with np.load(file) as data:
            features = data["features"]
            labels = data["labels"]

        total_samples += len(features)
        all_labels.extend(labels.tolist())

    all_labels = np.array(all_labels)

    print("Total samples:", total_samples)
    print("Feature shape:", features.shape)
    print("Label shape:", all_labels.shape)

    print("Unique classes:", len(np.unique(all_labels)))
    print("Min label:", all_labels.min())
    print("Max label:", all_labels.max())

    unique, counts = np.unique(
        all_labels,
        return_counts=True
    )

    print("Minimum samples/class:", counts.min())
    print("Maximum samples/class:", counts.max())