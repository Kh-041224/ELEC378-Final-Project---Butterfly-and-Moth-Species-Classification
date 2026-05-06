import skeleton
from skeleton import DATA_DIR, CSV_PATH, TRAIN_IMG_DIR
from sklearn.model_selection import train_test_split

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv(CSV_PATH)

# --------------------------------  
# CLass distributions
# ---------------------------------

def explore_class_distribution(df):
    print("----- CLASS DISTRIBUTION -----")
    num_samples = len(df)
    num_classes = df["TARGET"].nunique()

    print(f"Total samples: {num_samples}")
    print(f"Number of classes: {num_classes}")

    class_counts = df["TARGET"].value_counts().sort_values(ascending=False)
    print("\nTop 20 classes by count:")
    print(class_counts.head(20))

    print("\nSmallest 20 classes by count:")
    print(class_counts.tail(20))

    print(f"\nLargest class size: {class_counts.max()}")
    print(f"Smallest class size: {class_counts.min()}")
    print(f"Imbalance ratio (max/min): {class_counts.max() / class_counts.min():.2f}")

    plt.figure(figsize=(12, 5))
    class_counts.plot(kind="bar")
    plt.title("Number of Images per Class")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))

    bins = np.arange(class_counts.min(), class_counts.max() + 5, 5)

    plt.hist(class_counts.values, bins=bins, edgecolor='black', alpha=0.8)

    mean_val = class_counts.mean()
    median_val = class_counts.median()

    plt.axvline(mean_val, linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    plt.axvline(median_val, linestyle=':', linewidth=2, label=f'Median: {median_val:.1f}')

    plt.title("Distribution of Images per Class", fontsize=14)
    plt.xlabel("Number of Images per Class", fontsize=12)
    plt.ylabel("Number of Classes", fontsize=12)

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()  

def show_random_samples(df, img_dir, n=12, size=(12, 8)):
    sample_df = df.sample(n=min(n, len(df)), random_state=42)

    plt.figure(figsize=size)
    for i, (_, row) in enumerate(sample_df.iterrows()):
        img_path = os.path.join(img_dir, str(row["file_name"]).strip())
        img = Image.open(img_path).convert("RGB")

        plt.subplot(int(np.ceil(n / 4)), 4, i + 1)
        plt.imshow(img)
        plt.title(str(row["TARGET"]))
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# -------------------------
# intra-class variation
# -------------------------

def intra_class_variation(df, img_dir, resize=64, max_per_class=30):
    
    results = []

    # extracting images from each class
    for df_class in sorted(df["TARGET"].unique()):
        class_subset = df[df["TARGET"] == df_class]
    
        if len(df_class) > max_per_class:
            class_subset = class_subset.sample(max_per_class, random_state=42)
        
        image_vectors = []
        for _, row in class_subset.iterrows():
            img_path = os.path.join(img_dir, str(row["file_name"]).strip())
            try:
                img = Image.open(img_path).convert("RGB").resize((resize, resize))
                arr = np.asarray(img, dtype=np.float32) / 255.0
                image_vectors.append(arr.flatten())
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

        if len(image_vectors) < 2:
            continue

        X = np.stack(image_vectors, axis=0)
        mean_vec = X.mean(axis=0)
        dists = np.linalg.norm(X - mean_vec, axis=1) / np.sqrt(X.shape[1])

        results.append({
            "class": df_class,
            "num_images_used": len(X),
            "mean_distance_to_class_mean": dists.mean(),
            "std_distance": dists.std()
        })

    results_df = pd.DataFrame(results).sort_values(
        by="mean_distance_to_class_mean", ascending=False
    )

    print(results_df.head(20))

    plt.figure(figsize=(12, 5))
    plt.bar(results_df["class"].astype(str), results_df["mean_distance_to_class_mean"])
    plt.xticks(rotation=90)
    plt.title("Intra-Class Variation by Class")
    plt.xlabel("Class")
    plt.ylabel("Mean Distance to Class Mean")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(
    results_df["num_images_used"],
    results_df["mean_distance_to_class_mean"]
    )
    
    plt.xlabel("Number of Images")
    plt.ylabel("Intra-Class Variation")

    return results_df

# --------------------------------  
# FORMATTING CHECK
# ---------------------------------

        
def check_image_formatting(df, img_dir, max_images=None):
    print("----- IMAGE FORMATTING CHECK -----")

    records = []
    rows = df if max_images is None else df.iloc[:max_images]

    for _, row in rows.iterrows():
        fname = str(row["file_name"]).strip()
        img_path = os.path.join(img_dir, fname)

        try:
            with Image.open(img_path) as img:
                records.append({
                    "file_name": fname,
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "format": img.format
                })
        except Exception as e:
            records.append({
                "file_name": fname,
                "width": None,
                "height": None,
                "mode": None,
                "format": f"ERROR: {e}"
            })

    info_df = pd.DataFrame(records)

    print("Unique image sizes:")
    print(info_df[["width", "height"]].drop_duplicates().value_counts())

    print("\nUnique modes:")
    print(info_df["mode"].value_counts(dropna=False))

    print("\nUnique formats:")
    print(info_df["format"].value_counts(dropna=False))

    bad_rows = info_df[info_df["width"].isna()]
    print(f"\nUnreadable / problematic files: {len(bad_rows)}")
    if len(bad_rows) > 0:
        print(bad_rows.head(10))
    
    

# ------------------------------
# Image brightness and contrast consistency
# ------------------------------

def explore_image_statistics(df, img_dir, max_images=500):
    print("----- IMAGE STATISTICS -----")

    records = []

    rows = df.sample(min(max_images, len(df)), random_state=42)

    for _, row in rows.iterrows():
        fname = str(row["file_name"]).strip()
        img_path = os.path.join(img_dir, fname)

        try:
            img = Image.open(img_path).convert("L")
            arr = np.asarray(img, dtype=np.float32) / 255.0

            records.append({
                "file_name": fname,
                "class": row["TARGET"],
                "brightness": arr.mean(),
                "contrast": arr.std()
            })

        except Exception as e:
            print(f"Skipping {img_path}: {e}")

    stats_df = pd.DataFrame(records)

    print(f"Images analyzed: {len(stats_df)}")
    print("\nBrightness summary:")
    print(stats_df["brightness"].describe())

    print("\nContrast summary:")
    print(stats_df["contrast"].describe())

    # -----------------------------
    # histograms
    # -----------------------------
    plt.figure(figsize=(12, 4))

    # -------- Brightness --------
    plt.subplot(1, 2, 1)
    plt.hist(
        stats_df["brightness"],
        bins=30,
        edgecolor="black",
        alpha=0.8
    )

    mean_b = stats_df["brightness"].mean()
    median_b = stats_df["brightness"].median()

    plt.axvline(mean_b, linestyle="--", linewidth=2, label=f"Mean = {mean_b:.3f}")
    plt.axvline(median_b, linestyle=":", linewidth=2, label=f"Median = {median_b:.3f}")

    plt.title("Brightness Distribution")
    plt.xlabel("Mean Grayscale Intensity")
    plt.ylabel("Number of Images")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    # -------- Contrast --------
    plt.subplot(1, 2, 2)
    plt.hist(
        stats_df["contrast"],
        bins=30,
        edgecolor="black",
        alpha=0.8
    )

    mean_c = stats_df["contrast"].mean()
    median_c = stats_df["contrast"].median()

    plt.axvline(mean_c, linestyle="--", linewidth=2, label=f"Mean = {mean_c:.3f}")
    plt.axvline(median_c, linestyle=":", linewidth=2, label=f"Median = {median_c:.3f}")

    plt.title("Contrast Distribution")
    plt.xlabel("Grayscale Standard Deviation")
    plt.ylabel("Number of Images")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return stats_df


def main():
    print("Exploring class distribution...")
    explore_class_distribution(df)

    print("Showing random samples...")
    show_random_samples(df, TRAIN_IMG_DIR, n=12)

    format_df = check_image_formatting(df, TRAIN_IMG_DIR)
    variation_df = intra_class_variation(df, TRAIN_IMG_DIR, resize=64, max_per_class=100)
    explore_image_statistics(df, TRAIN_IMG_DIR, max_images=500)

if __name__ == "__main__":
    main()