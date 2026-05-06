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

## Model 1: KNN (Non-Neural-Network)

### Overview

K-Nearest Neighbors with PCA-reduced raw pixel features. Exhaustive grid search over 216 hyperparameter configurations.

**Best configuration:** PCA components = 200, k = 20, metric = cosine, weights = distance  
**Validation accuracy:** 41.56%

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
   | 2 | Load images, encode labels, 80/20 stratified split
   | 3 | StandardScaler + PCA (300 components, whitened)
   | 4 | Grid search: 9 PCA dims × 6 k-values × 4 metrics = 216 configs, each with 3-fold CV 
   | 5 | Visualization: PCA variance plot + grid search heatmaps
   | 6 | Refit best model, predict test set, save submission CSV 

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

## Model 2: ResNet50 (This is the best performing model)

### Overview

ResNet50 architecture with no pretrained weights, trained with SAM optimizer, data augmentation including RandAugment, MixUp, CutMix, and RandomErasing, cosine annealing warm restarts, and test-time augmentation (10 views).

**Best validation accuracy:** 94.28% (epoch ~140)  
**Kaggle score:** 97.8% public / 97.6% private


### Dependencies

Google Colab comes with all dependencies pre-installed. If running locally, they are all listed in the first cell. 

### How to Run

#### Google Colab

1. Upload the dataset to your Google Drive under:
   ```
   My Drive/ELEC378FinalProject/
   ├── train.csv
   ├── train_images/
   └── test_images/
   ```

2. Run all cells in order. The notebook will:
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


