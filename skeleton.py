import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "butterfly_data")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
CSV_PATH = os.path.join(DATA_DIR, "training_set.csv")

RANDOM_STATE = 42

# Loads the metadata from the csv file
def load_metadata():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} entries, {df['TARGET'].nunique()} classes")
    return df

# Loads the images 
def load_image(filename, size):
    path = os.path.join(TRAIN_IMG_DIR, filename)
    img = Image.open(path).convert("RGB").resize((size, size))
    return np.array(img)

# Load one image-label pair
def load_image_label_pair(df, index, size=224):
    row = df.iloc[index]
    img = load_image(row["file_name"], size)
    label = row["TARGET"]
    return img, label


def main():
    print("Loading the ELEC378 Final Project Dataset")

    df = load_metadata()
    print(df.head())
    print("Loading csv done")

    img, label = load_image_label_pair(df, 0)
    print(f"Sample pair — label: {label}, image shape: {img.shape}")

    X_train_files, X_val_files, y_train, y_val = train_test_split(
        df["file_name"].values,
        df["TARGET"].values,
        test_size=0.2,
        stratify=df["TARGET"].values,
        random_state=RANDOM_STATE,
    )
    print(f"Train: {len(X_train_files)}, Validation: {len(X_val_files)}")

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    img, label = load_image_label_pair(df, idx)
    print(f"Showing the {idx} th image: {label}")

    plt.imshow(img)
    plt.title(f"[{idx}] {label}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()