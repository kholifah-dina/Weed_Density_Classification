import os
import joblib
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from preprocessing import preprocess_image
from feature_extraction import extract_features, select_features

# Absolute paths relative to this file so they work regardless of working directory
_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_DIR, 'models')
MODELS_PATH = os.path.join(MODEL_DIR, 'weed_models.joblib')
METRICS_PATH = os.path.join(MODEL_DIR, 'weed_metrics.joblib')

FEATURE_NAMES = [
    'homogeneity_90deg', 'homogeneity_45deg', 'energy_45deg',      'energy_135deg',
    'energy_90deg',      'energy_0deg',        'homogeneity_0deg',  'homogeneity_135deg',
    'dissimilarity_90deg', 'dissimilarity_135deg', 'dissimilarity_45deg',
    'dissimilarity_0deg', 'G_std', 'G_mean',
    # Note: 'HuMoment_6' replaced by 'G_std' (Cohen d: 0.0001 → 1.162, p: 0.183 → <0.001)
]

# ─── Model factory ──────────────────────────────────────────────────────────
# class_weight='balanced' corrects for the Sedang majority bias (41 % vs 29 % Padat).
# GradientBoostingClassifier does not support class_weight; sample_weight is
# passed at fit-time inside _fit_and_evaluate instead.
_MODELS_CONFIG = {
    "Logistic Regression": lambda: LogisticRegression(
        solver='lbfgs', max_iter=300, random_state=42, class_weight='balanced'
    ),
    "SVM": lambda: SVC(
        kernel='rbf', C=5, gamma=0.01, probability=True, random_state=42,
        class_weight='balanced'
    ),
    "Decision Tree": lambda: DecisionTreeClassifier(
        max_depth=3, random_state=42, class_weight='balanced'
    ),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=500, random_state=42, class_weight='balanced'
    ),
    "Gradient Boosting": lambda: GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.1, random_state=42
        # sample_weight applied at fit-time — see _fit_and_evaluate
    ),
}


# ─── Internal helpers ────────────────────────────────────────────────────────

def _split_80_10_10(X, y):
    """
    Stratified 80 : 10 : 10 split.
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    # Step 1: 80% train, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Step 2: split 20% temp → 10% val + 10% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _compute_sample_weights(y):
    """
    Compute per-sample inverse-frequency weights so every class contributes
    equally during training regardless of class size.
    Equivalent to class_weight='balanced' for models that do not support it
    natively (e.g. GradientBoostingClassifier).
    """
    classes, counts = np.unique(y, return_counts=True)
    weight_map = {cls: len(y) / (len(classes) * cnt)
                  for cls, cnt in zip(classes, counts)}
    return np.array([weight_map[label] for label in y])


def _fit_and_evaluate(models_config, X_train_sc, X_val_sc, X_test_sc,
                      y_train, y_val, y_test, unique_labels):
    """
    Train every model in models_config and return
    (trained_models, evaluation_metrics, confusion_matrices).
    Metrics are computed on the test set; Val Accuracy is also recorded.

    GradientBoostingClassifier receives sample_weight at fit-time because it
    does not support the class_weight constructor parameter.
    """
    trained_models      = {}
    evaluation_metrics  = {}
    confusion_matrices  = {}

    # Pre-compute balanced sample weights for GBT
    sample_weights = _compute_sample_weights(y_train)

    for model_name, model_factory in models_config.items():
        model = model_factory()
        t0 = time.time()
        if model_name == "Gradient Boosting":
            model.fit(X_train_sc, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train_sc, y_train)
        exec_time = round(time.time() - t0, 4)

        y_pred_val  = model.predict(X_val_sc)
        y_pred_test = model.predict(X_test_sc)

        evaluation_metrics[model_name] = {
            "Accuracy":         round(accuracy_score(y_test, y_pred_test), 4),
            "Precision":        round(precision_score(y_test, y_pred_test, average='macro', zero_division=0), 4),
            "Recall":           round(recall_score(y_test, y_pred_test, average='macro',    zero_division=0), 4),
            "F1-Score":         round(f1_score(y_test, y_pred_test,      average='macro',   zero_division=0), 4),
            "Val Accuracy":     round(accuracy_score(y_val, y_pred_val), 4),
            "Execution Time (s)": exec_time,
        }
        confusion_matrices[model_name] = {
            "matrix": confusion_matrix(y_test, y_pred_test, labels=unique_labels),
            "labels": unique_labels,
        }
        trained_models[model_name] = model

    return trained_models, evaluation_metrics, confusion_matrices


# ─── Public API ─────────────────────────────────────────────────────────────

def train_models(dataset):
    """
    Train all models on an uploaded image dataset.

    Split strategy: 80% Train | 10% Validation | 10% Test  (stratified).
    Evaluation metrics are reported on the held-out Test split.

    Parameters
    ----------
    dataset : dict
        {'Renggang': [img_bytes, ...], 'Sedang': [...], 'Padat': [...]}

    Returns
    -------
    saved_metrics : dict
        Keys: 'metrics', 'confusion_matrices', 'features', 'split_info'.
    """
    X_all, y_all = [], []

    for class_name, img_bytes_list in dataset.items():
        for img_bytes in img_bytes_list:
            segmented = preprocess_image(image_bytes=img_bytes)
            if segmented is None:
                continue
            features = extract_features(segmented)
            selected = select_features(features)
            X_all.append(selected)
            y_all.append(class_name)

    X = np.array(X_all)
    y = np.array(y_all)

    X_train, X_val, X_test, y_train, y_val, y_test = _split_80_10_10(X, y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    unique_labels = sorted(set(y_all))
    trained_models, evaluation_metrics, confusion_matrices = _fit_and_evaluate(
        _MODELS_CONFIG, X_train_sc, X_val_sc, X_test_sc,
        y_train, y_val, y_test, unique_labels,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({'models': trained_models, 'scaler': scaler}, MODELS_PATH)

    saved_metrics = {
        "metrics":           evaluation_metrics,
        "confusion_matrices": confusion_matrices,
        "features":          FEATURE_NAMES,
        "split_info": {
            "total": len(y_all),
            "train": len(y_train),
            "val":   len(y_val),
            "test":  len(y_test),
        },
    }
    joblib.dump(saved_metrics, METRICS_PATH)
    return saved_metrics


def train_models_from_csv(csv_path):
    """
    Train all models from a **pre-extracted feature CSV** file.

    The CSV must contain the 39 raw feature columns + a ``Class`` column
    (same format as *Data_ekstraksi_Fitur_Gulma.csv*).
    Feature selection (top-14 by Information Gain) is applied automatically.

    Split: 80% Train | 10% Validation | 10% Test  (stratified).
    """
    import pandas as pd

    df   = pd.read_csv(csv_path)
    X_all = df.drop(columns=['Class']).to_numpy(dtype=float)
    y_all = df['Class'].to_numpy(dtype=str)
    X_sel = select_features(X_all)          # shape (n, 14)

    X_train, X_val, X_test, y_train, y_val, y_test = _split_80_10_10(X_sel, y_all)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    unique_labels = sorted(set(y_all))
    trained_models, evaluation_metrics, confusion_matrices = _fit_and_evaluate(
        _MODELS_CONFIG, X_train_sc, X_val_sc, X_test_sc,
        y_train, y_val, y_test, unique_labels,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({'models': trained_models, 'scaler': scaler}, MODELS_PATH)

    saved_metrics = {
        "metrics":           evaluation_metrics,
        "confusion_matrices": confusion_matrices,
        "features":          FEATURE_NAMES,
        "split_info": {
            "total": len(y_all),
            "train": len(y_train),
            "val":   len(y_val),
            "test":  len(y_test),
        },
    }
    joblib.dump(saved_metrics, METRICS_PATH)
    return saved_metrics


def load_pipeline():
    """Load the saved model pipeline (models + scaler)."""
    if not os.path.exists(MODELS_PATH):
        raise FileNotFoundError(
            "Model belum tersedia. Silakan lakukan Training terlebih dahulu di halaman Admin."
        )
    data = joblib.load(MODELS_PATH)
    # Validate expected structure: must be {'models': {...}, 'scaler': <StandardScaler>}
    if not isinstance(data, dict) or 'models' not in data or 'scaler' not in data:
        raise ValueError(
            "File model tidak kompatibel (versi lama tanpa StandardScaler). "
            "Silakan buka menu Admin → 🗂️ Training Model dan latih ulang "
            "model dengan data yang sama agar pipeline tersimpan dengan benar."
        )
    return data


def load_metrics():
    """Load saved evaluation metrics."""
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError("File metrik tidak ditemukan.")
    return joblib.load(METRICS_PATH)


def get_feature_values(image_bytes):
    """
    Extract the 14 selected features from a single image for **display** purposes.

    Returns
    -------
    np.ndarray of shape (14,), or None if the image cannot be decoded.
    """
    segmented = preprocess_image(image_bytes=image_bytes)
    if segmented is None:
        return None
    features = extract_features(segmented)
    return select_features(features)


def predict_with_model(image_bytes, model_name):
    """
    Run inference using **one specific model** from the saved pipeline.

    Returns
    -------
    prediction    : str   — class label ('Renggang' | 'Sedang' | 'Padat')
    feature_values: np.ndarray  — 14 selected features (unscaled)
    """
    pipeline = load_pipeline()
    trained_models = pipeline['models']
    scaler         = pipeline['scaler']

    if model_name not in trained_models:
        raise ValueError(f"Model '{model_name}' tidak ditemukan dalam pipeline yang tersimpan.")

    segmented = preprocess_image(image_bytes=image_bytes)
    features  = extract_features(segmented)
    selected  = select_features(features)
    X_single  = scaler.transform(np.array([selected]))

    prediction = trained_models[model_name].predict(X_single)[0]
    return prediction, selected


def test_inference(image_bytes):
    """
    Run inference on a single image using **all** saved models.

    Returns
    -------
    predictions    : dict          — {model_name: predicted_class_label}
    saved_metrics  : dict
    feature_values : np.ndarray   — 14 selected features (unscaled)
    """
    pipeline      = load_pipeline()
    saved_metrics = load_metrics()

    trained_models = pipeline['models']
    scaler         = pipeline['scaler']

    segmented = preprocess_image(image_bytes=image_bytes)
    features  = extract_features(segmented)
    selected  = select_features(features)
    X_single  = scaler.transform(np.array([selected]))

    predictions = {
        model_name: model.predict(X_single)[0]
        for model_name, model in trained_models.items()
    }

    return predictions, saved_metrics, selected
