# ELEC378-Final-Project---Butterfly-and-Moth-Species-Classification

---

## Repository Structure

```
.
├── README.md                                  ← You are here
├── KNN_Training_Gridsearch_Prediction.ipynb   ← KNN model (non-neural-network)
├── ResNet50_97_2_.ipynb                       ← ResNet50 model (neural network)                        
```

---

## Dataset

The dataset is hosted on the course Kaggle competition. It contains:

- **12,594 labeled training images** across 100 butterfly/moth species
- **1,000 unlabeled test images** for Kaggle submission
- **train.csv** with columns `file_name` and `TARGET`

Download the data from Kaggle and organize it as follows:

```
your_data_directory/
├── train.csv
├── train_images/
│   ├── train_000001.jpg
│   ├── train_000002.jpg
│   └── ...
└── test_images/
    ├── test_000001.jpg
    ├── test_000002.jpg
    └── ...
```

> **Note:** Kaggle may extract images into nested folders (e.g., `train_images/train_images/`). Make sure your path points to the directory that directly contains the `.jpg` files.

---

## Model 1: KNN (Non-Neural-Network)

### Overview

K-Nearest Neighbors with PCA-reduced raw pixel features. Exhaustive grid search over 216 hyperparameter configurations.

**Best configuration:** PCA components = 200, k = 20, metric = cosine, weights = distance  
**Validation accuracy:** 41.56%

### Environment

Runs on CPU (no GPU required). Tested on Python 3.11, Windows.

### Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn pillow
```

### How to Run

1. Open `KNN_Training_Gridsearch_Prediction.ipynb` in Jupyter Notebook or VS Code.

2. Update the four path constants in **Cell 1** to match your local data directory:
   ```python
   TRAIN_IMG_DIR = r'path/to/train_images'
   CSV_PATH      = r'path/to/train.csv'
   TEST_IMG_DIR  = r'path/to/test_images'
   SUBMISSION_PATH = r'path/to/submissions'
   ```

3. Run all cells in order. The notebook will:

   | Cell | Step | Time Estimate |
   |------|------|---------------|
   | 1 | Imports and config | Instant |
   | 2 | Load images, encode labels, 80/20 stratified split | ~30s |
   | 3 | StandardScaler + PCA (300 components, whitened) | ~20s |
   | 4 | Grid search: 9 PCA dims × 6 k-values × 4 metrics = 216 configs, each with 3-fold CV | ~15–30 min |
   | 5 | Visualization: PCA variance plot + grid search heatmaps | Instant |
   | 6 | Refit best model, predict test set, save submission CSV | ~1 min |

4. The submission CSV is saved to `SUBMISSION_PATH` with a timestamped filename:
   ```
   submission_knn_20260502_143012.csv
   ```

### Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `IMG_SIZE` | 128 | Images resized to 128×128 |
| `RANDOM_STATE` | 42 | Reproducibility seed |
| `MAX_PCA` | 300 | Components computed; grid search tests 2–300 |
| `n_components_grid` | [2, 5, 10, 20, 50, 100, 150, 200, 300] | PCA dims swept |
| `n_neighbors_grid` | [2, 5, 10, 15, 20, 50] | k values swept |
| `metric_grid` | [Cosine, Euclidean, Manhattan, Chebyshev] | Distance metrics swept |
| `weights` | distance | Inverse-distance weighted voting |

---

## Model 2: ResNet50 (Neural Network, Trained from Scratch)

### Overview

ResNet50 architecture with no pretrained weights, trained with SAM optimizer, aggressive data augmentation including RandAugment, MixUp, CutMix, and RandomErasing, cosine annealing warm restarts, and test-time augmentation (10 views).

**Best validation accuracy:** 94.28% (epoch ~140)  
**Kaggle score:** 97.8% public / 97.6% private

### Environment

**Requires a CUDA-capable GPU.** Training takes approximately 80 minutes for 160 epochs on A100, but timing varies depending on the GPU quality. 

### Dependencies

Google Colab comes with all dependencies pre-installed. If running locally:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow tqdm
```

PyTorch must be installed with CUDA support. See [pytorch.org](https://pytorch.org/get-started/locally/) for installation instructions matching your CUDA version.

### How to Run

#### Google Colab

1. Upload the dataset to your Google Drive under:
   ```
   My Drive/ELEC378FinalProject/
   ├── train.csv
   ├── train_images/
   └── test_images/
   ```

2. Open `ResNet50_97_2_.ipynb` in Google Colab.

3. Enable GPU: **Runtime → Change runtime type → T4/A100 GPU**.

4. Run all cells in order. The notebook will:
   - Mount Google Drive and set paths
   - Copy data to Colab local storage for faster I/O 
   - Compute dataset channel means and standard deviations 
   - Define augmentation transforms, model architecture, and SAM optimizer 
   - Train for 160 epochs with best checkpoint saving 
   - Run TTA inference on the test set and save submission CSV 


### Training Details

| Phase | Epochs | Learning Rate | Notes |
|-------|--------|---------------|-------|
| Warmup | 1–5 | 0.02 → 0.10 (linear) | Gradual ramp-up |
| Cycle 1 | 6–25 | Cosine 0.10 → 1e-5 | T₀ = 20 |
| Cycle 2 | 26–65 | Cosine 0.10 → 1e-5 | T₀ × 2 = 40 |
| Cycle 3 | 66–145 | Cosine 0.10 → 1e-5 | T₀ × 4 = 80 |
| Remaining | 146–160 | Cosine | Best checkpoint typically saved here |

### Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `IMG_SIZE` | 224 | Standard ResNet input |
| `BATCH_SIZE` | 64 | Adjust down for smaller GPUs |
| `EPOCHS` | 160 | With best-checkpoint saving |
| `RANDOM_STATE` | 100 | Seeds for Python, NumPy, PyTorch |
| Base optimizer | SGD + Nesterov (momentum=0.9) | Wrapped in SAM |
| SAM ρ | 0.05 | Perturbation radius |
| Weight decay | 5e-4 | L2 regularization |
| Label smoothing | 0.1 | In CrossEntropyLoss |
| Dropout | 0.3 | Before final linear layer |
| MixUp α | 0.4 | Beta(0.4, 0.4) distribution |
| CutMix α | 1.0 | Beta(1.0, 1.0) distribution |
| MixUp/CutMix prob | 0.5 | Per-batch probability |
| Gradient clip | 1.0 | Max gradient norm |
| TTA views | 10 | 1 center + 1 flipped + 8 random crops |

### Reproducing with a Saved Checkpoint

If you have the saved `best_model.pth` checkpoint and want to skip training:

1. Place `best_model.pth` in your working directory.
2. Run all cells up to and including the model definition cells (Cells 1–15).
3. Skip the training cell (Cell 17).
4. Run the inference cell (Cell 21), which loads `best_model.pth` and runs TTA prediction.

---

## Submission Format

Both notebooks output a CSV in the Kaggle-required format:

```csv
ID,TARGET
test_000001,MONARCH
test_000002,BLUE MORPHO
test_000003,ATLAS MOTH
...
```

- **ID**: Image filename without the `.jpg` extension
- **TARGET**: Predicted species name string

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError` on image paths | Check for nested folders (e.g., `train_images/train_images/`). Point paths to the directory containing `.jpg` files directly. |
| KNN grid search is very slow | Reduce `n_components_grid` to `[50, 100, 200]` or `n_neighbors_grid` to `[5, 10, 20]` for a faster sweep. |
| CUDA out of memory (ResNet50) | Reduce `BATCH_SIZE` from 64 to 32 or 16. If using T4 instead of A100, 32 is recommended. |
| `NaN` loss during ResNet50 training | The notebook has a built-in NaN guard that skips bad batches. If it occurs frequently, reduce the learning rate or SAM ρ. |
| Mixed-case label (`Iphiclus sister`) | This is expected. The LabelEncoder handles it correctly. Do not manually uppercase it. |
