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
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from preprocessing import preprocess_image
from feature_extraction import extract_features, extract_features_with_glcm


def _extract_for_pipeline(pipeline, segmented):
    """
    Ekstrak fitur dari gambar yang sudah di-preprocess sesuai dengan mode
    pipeline yang tersimpan:

    - Jika pipeline mempunyai 'selector' (dilatih dari CSV 39 fitur dengan GLCM):
        1. Ekstrak 39 fitur (GLCM + RGB + HSV + Hu)
        2. Terapkan selector (memilih 14 fitur terbaik)
    - Jika tidak ada 'selector' (dilatih dari CSV 19 fitur atau dari gambar upload):
        1. Ekstrak 19 fitur (RGB + HSV + Hu) saja

    Returns
    -------
    features_raw : np.ndarray  — fitur sebelum seleksi (19 atau 39)
    features_sel : np.ndarray  — fitur setelah seleksi (siap masuk scaler)
    n_raw        : int         — jumlah fitur asli
    uses_glcm    : bool        — True jika pipeline menggunakan GLCM+LAN
    """
    uses_glcm = 'selector' in pipeline
    if uses_glcm:
        features_raw = extract_features_with_glcm(segmented)  # (39,)
        features_sel = pipeline['selector'].transform(np.array([features_raw]))[0]  # (14,)
    else:
        features_raw = extract_features(segmented)            # (19,)
        features_sel = features_raw
    return features_raw, features_sel, len(features_raw), uses_glcm

# Absolute paths relative to this file so they work regardless of working directory
_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_DIR, 'models')
MODELS_PATH = os.path.join(MODEL_DIR, 'weed_models.joblib')
METRICS_PATH = os.path.join(MODEL_DIR, 'weed_metrics.joblib')

# ─── Konstanta jumlah fitur ──────────────────────────────────────────────────
# CSV dengan 39 fitur (20 GLCM + 6 RGB + 6 HSV + 7 HuMoment) → seleksi 14 terbaik (LAN/IG)
# CSV dengan 19 fitur (tanpa GLCM: 6 RGB + 6 HSV + 7 HuMoment) → pakai semua 19 fitur
N_FEATURES_WITH_GLCM    = 39   # jumlah fitur CSV dengan GLCM
N_FEATURES_WITHOUT_GLCM = 19   # jumlah fitur CSV tanpa GLCM
N_SELECT_BEST           = 14   # jumlah fitur yang dipilih LAN untuk CSV 39 fitur

# 19 fitur (tanpa GLCM) — untuk Mode A (upload gambar) dan CSV 19 fitur
FEATURE_NAMES = [
    'R_mean', 'G_mean', 'B_mean',
    'R_std',  'G_std',  'B_std',
    'H_mean', 'S_mean', 'V_mean',
    'H_std',  'S_std',  'V_std',
    'HuMoment_1', 'HuMoment_2', 'HuMoment_3',
    'HuMoment_4', 'HuMoment_5', 'HuMoment_6', 'HuMoment_7',
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
            X_all.append(features)
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

    Dua mode otomatis berdasarkan jumlah fitur:

    1. **CSV 39 fitur** (dengan GLCM — 20 GLCM + 6 RGB + 6 HSV + 7 HuMoment):
       - Terapkan Information Gain (LAN / mutual_info_classif) untuk memilih
         14 fitur terbaik.
       - Feature selector (SelectKBest) disimpan bersama scaler.

    2. **CSV 19 fitur** (tanpa GLCM — 6 RGB + 6 HSV + 7 HuMoment):
       - Semua 19 fitur digunakan langsung tanpa seleksi.

    Split: 80% Train | 10% Validation | 10% Test  (stratified).
    """
    import pandas as pd

    df    = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c != 'Class']
    X_all = df[feat_cols].to_numpy(dtype=float)
    y_all = df['Class'].to_numpy(dtype=str)

    n_feat = X_all.shape[1]
    use_feature_selection = (n_feat == N_FEATURES_WITH_GLCM)

    if use_feature_selection:
        # ── Mode: 39 fitur dengan GLCM → Seleksi 14 fitur terbaik (LAN/IG) ──
        # Fit selector pada SELURUH data sebelum split agar nama fitur konsisten;
        # namun untuk menghindari data leakage, selector di-fit ulang hanya pada
        # data train setelah split (standar ML yang benar).
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = _split_80_10_10(X_all, y_all)

        selector = SelectKBest(score_func=mutual_info_classif, k=N_SELECT_BEST)
        # Fit selector HANYA pada data training (tidak ada data leakage)
        selector.fit(X_train_raw, y_train)

        # Ambil nama fitur yang terpilih
        selected_mask  = selector.get_support()
        selected_names = [feat_cols[i] for i, m in enumerate(selected_mask) if m]

        # Transform semua split
        X_train_sel = selector.transform(X_train_raw)
        X_val_sel   = selector.transform(X_val_raw)
        X_test_sel  = selector.transform(X_test_raw)

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_sel)
        X_val_sc   = scaler.transform(X_val_sel)
        X_test_sc  = scaler.transform(X_test_sel)

        unique_labels = sorted(set(y_all))
        trained_models, evaluation_metrics, confusion_matrices = _fit_and_evaluate(
            _MODELS_CONFIG, X_train_sc, X_val_sc, X_test_sc,
            y_train, y_val, y_test, unique_labels,
        )

        os.makedirs(MODEL_DIR, exist_ok=True)
        # Simpan selector bersama pipeline agar bisa dipakai saat inference
        joblib.dump(
            {'models': trained_models, 'scaler': scaler, 'selector': selector},
            MODELS_PATH
        )

        saved_metrics = {
            "metrics":            evaluation_metrics,
            "confusion_matrices": confusion_matrices,
            "features":           selected_names,   # 14 fitur terpilih
            "n_features_input":   n_feat,
            "n_features_selected": N_SELECT_BEST,
            "feature_selection":  True,
            "split_info": {
                "total": len(y_all),
                "train": len(y_train),
                "val":   len(y_val),
                "test":  len(y_test),
            },
        }

    else:
        # ── Mode: 19 fitur tanpa GLCM → gunakan semua langsung ──────────────
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = _split_80_10_10(X_all, y_all)

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_raw)
        X_val_sc   = scaler.transform(X_val_raw)
        X_test_sc  = scaler.transform(X_test_raw)

        unique_labels = sorted(set(y_all))
        trained_models, evaluation_metrics, confusion_matrices = _fit_and_evaluate(
            _MODELS_CONFIG, X_train_sc, X_val_sc, X_test_sc,
            y_train, y_val, y_test, unique_labels,
        )

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump({'models': trained_models, 'scaler': scaler}, MODELS_PATH)

        saved_metrics = {
            "metrics":            evaluation_metrics,
            "confusion_matrices": confusion_matrices,
            "features":           FEATURE_NAMES,   # semua 19 fitur
            "n_features_input":   n_feat,
            "n_features_selected": n_feat,
            "feature_selection":  False,
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
    Ekstrak fitur dari satu gambar untuk ditampilkan di UI.
    Deteksi otomatis mode pipeline:
    - Pipeline dengan GLCM (selector): kembalikan 39 fitur asli (untuk tampilan)
    - Pipeline tanpa GLCM: kembalikan 19 fitur

    Returns
    -------
    features_raw : np.ndarray  — fitur asli (19 atau 39), atau None jika gagal
    feature_names_raw : list   — nama kolom fitur asli
    uses_glcm : bool           — True jika pipeline menggunakan GLCM
    """
    from feature_extraction import extract_features_with_glcm

    segmented = preprocess_image(image_bytes=image_bytes)
    if segmented is None:
        return None, FEATURE_NAMES, False

    # Cek apakah ada pipeline tersimpan dengan selector
    uses_glcm = False
    try:
        _pl = load_pipeline()
        uses_glcm = 'selector' in _pl
    except Exception:
        pass  # pipeline belum ada, pakai mode 19 fitur

    if uses_glcm:
        features_raw = extract_features_with_glcm(segmented)  # (39,)
        # Nama 39 fitur — urutan sesuai extract_features_with_glcm
        _glcm_names = [
            f"{prop}_{ang}"
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            for ang in ['0deg', '45deg', '90deg', '135deg']
        ]
        raw_names = _glcm_names + FEATURE_NAMES   # 20 GLCM + 19 = 39
        return features_raw, raw_names, True
    else:
        return extract_features(segmented), FEATURE_NAMES, False


def predict_with_model(image_bytes, model_name):
    """
    Run inference using **one specific model** from the saved pipeline.
    Secara otomatis mendeteksi mode pipeline (19 fitur vs 39 fitur+LAN).

    Returns
    -------
    prediction    : str          — class label ('Renggang' | 'Sedang' | 'Padat')
    features_raw  : np.ndarray   — fitur asli sebelum seleksi (19 atau 39, unscaled)
    """
    pipeline = load_pipeline()
    trained_models = pipeline['models']
    scaler         = pipeline['scaler']

    if model_name not in trained_models:
        raise ValueError(f"Model '{model_name}' tidak ditemukan dalam pipeline yang tersimpan.")

    segmented = preprocess_image(image_bytes=image_bytes)
    if segmented is None:
        raise ValueError("Gambar tidak dapat diproses.")

    features_raw, features_sel, _, _ = _extract_for_pipeline(pipeline, segmented)
    X_single = scaler.transform(np.array([features_sel]))

    prediction = trained_models[model_name].predict(X_single)[0]
    return prediction, features_raw


def test_inference(image_bytes):
    """
    Run inference pada satu gambar menggunakan **semua** model tersimpan.
    Secara otomatis mendeteksi mode pipeline (19 fitur vs 39 fitur+LAN).

    Returns
    -------
    predictions    : dict          — {model_name: predicted_class_label}
    saved_metrics  : dict
    features_raw   : np.ndarray   — fitur asli sebelum seleksi (19 atau 39, unscaled)
    """
    pipeline      = load_pipeline()
    saved_metrics = load_metrics()

    trained_models = pipeline['models']
    scaler         = pipeline['scaler']

    segmented = preprocess_image(image_bytes=image_bytes)
    if segmented is None:
        raise ValueError("Gambar tidak dapat diproses.")

    features_raw, features_sel, _, _ = _extract_for_pipeline(pipeline, segmented)
    X_single = scaler.transform(np.array([features_sel]))

    predictions = {
        model_name: model.predict(X_single)[0]
        for model_name, model in trained_models.items()
    }

    return predictions, saved_metrics, features_raw
