# GTZAN Music Genre Classification

## Project Background

This project tackles the problem of **automatic music genre classification** using machine learning. Given a short audio clip, the goal is to predict its genre (e.g., blues, jazz, pop, rock) from a set of 10 categories. Rather than processing raw audio waveforms, the pipeline operates on pre-extracted acoustic features, making it efficient and well-suited for classical ML algorithms like SVM and LightGBM.

This is a 10-class classification problem with ~9,990 samples drawn from the GTZAN dataset — a standard benchmark in Music Information Retrieval (MIR).

---

## Environment Setup

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn lightgbm xgboost seaborn matplotlib joblib
```

| Library | Version (recommended) | Purpose |
|---|---|---|
| scikit-learn | ≥ 1.3 | SVM, preprocessing, evaluation |
| lightgbm | ≥ 4.0 | LightGBM classifier |
| xgboost | ≥ 2.0 | XGBoost classifier |
| pandas | ≥ 2.0 | Data loading and manipulation |
| numpy | ≥ 1.24 | Numerical operations |
| seaborn / matplotlib | latest | Visualization |
| joblib | ≥ 1.3 | Model serialization |

---

## Dataset

**Source:** [GTZAN Genre Collection](http://marsyas.info/downloads/datasets.html)

The dataset contains 1,000 audio clips per genre across 10 genres:
`blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`

**Feature file used:** `features_3_sec.csv` — each row represents a 3-second audio segment described by 58 acoustic features.

### Features include:
- Chroma STFT (mean & variance)
- RMS energy (mean & variance)
- Spectral centroid, bandwidth, rolloff (mean & variance)
- Zero crossing rate
- MFCCs 1–20 (mean & variance)

### Preprocessing steps:
1. Drop non-feature columns: `filename`, `length`
2. Encode labels with `LabelEncoder`
3. Split into train/test sets: 80% / 20%, stratified by genre (`random_state=42`)
4. Apply `StandardScaler` (fit on train, transform both splits) to normalize feature magnitudes

---

## Model Architecture

Three models are trained and compared, all wrapped in a `sklearn.Pipeline` with `VarianceThreshold(threshold=0.1)` for feature selection:

### 1. Support Vector Machine (SVM)
- Kernel: RBF (best found via grid search)
- Hyperparameter search: `C ∈ {1, 10, 50}`, `kernel ∈ {rbf, poly}`, `gamma ∈ {scale, auto}`
- 5-fold cross-validation via `GridSearchCV`
- Best params: `C=50, kernel=rbf, gamma=auto`

### 2. XGBoost
- Objective: `multi:softmax` (10 classes)
- Hyperparameter search: `n_estimators ∈ {100, 200}`, `max_depth ∈ {4, 6}`, `learning_rate=0.1`
- Tree method: `hist` for faster training

### 3. LightGBM
- Objective: `multiclass` (10 classes)
- Hyperparameter search: `n_estimators ∈ {100, 200}`, `num_leaves ∈ {31, 63}`, `learning_rate ∈ {0.05, 0.1}`
- Feature importance type: `gain`

---

## Usage

### Training

Run the notebook `GTZAN.ipynb` end-to-end, or execute the training cells individually. Trained models are saved automatically:

```
model/svm_model.pkl       # Best SVM pipeline
model/lgbm_model.joblib   # Best LightGBM pipeline
```

### Inference

```python
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model
model = joblib.load('model/lgbm_model.joblib')

# Prepare features (58-column DataFrame, same as training features)
X_new = pd.read_csv('your_features.csv')
X_new_scaled = scaler.transform(X_new)  # use the scaler fitted during training

# Predict
predictions = model.predict(X_new_scaled)
print(predictions)  # returns encoded genre labels
```

> Note: Save and reload the `StandardScaler` alongside the model to ensure consistent preprocessing at inference time.

---

## Experimental Results

| Model | Val Accuracy (CV) | Test Accuracy | Macro F1 |
|---|---|---|---|
| SVM (RBF, C=50) | 90.68% | 91.59% | 0.92 |
| XGBoost | — | — | — |
| LightGBM | — | — | — |

> Fill in the XGBoost and LightGBM results after running the full grid search. Confusion matrix plots are saved as `svm_confusion_matrix.png`.
