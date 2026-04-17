import os
import joblib
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from preprocessing import preprocess_image
from feature_extraction import extract_features, extract_features_with_glcm

# ─── Paths ───────────────────────────────────────────────────────────────────
_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_DIR, 'models')

# ─── Konstanta ────────────────────────────────────────────────────────────────
N_FEATURES_WITH_GLCM    = 39
N_FEATURES_WITHOUT_GLCM = 19
N_SELECT_BEST           = 14

# Nama 19 fitur (tanpa GLCM) — urutan sesuai extract_features()
FEATURE_NAMES_19 = [
    'R_mean', 'G_mean', 'B_mean',
    'R_std',  'G_std',  'B_std',
    'H_mean', 'S_mean', 'V_mean',
    'H_std',  'S_std',  'V_std',
    'HuMoment_1', 'HuMoment_2', 'HuMoment_3',
    'HuMoment_4', 'HuMoment_5', 'HuMoment_6', 'HuMoment_7',
]

# Nama 39 fitur — urutan sesuai extract_features_with_glcm()
FEATURE_NAMES_39 = (
    [f"{prop}_{ang}"
     for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
     for ang in ['0deg', '45deg', '90deg', '135deg']]
    + FEATURE_NAMES_19
)

# Mapping nama model → nama file pendek
MODEL_SHORT = {
    'Decision Tree':       'DT',
    'Logistic Regression': 'LR',
    'SVM':                 'SVM',
    'Random Forest':       'RF',
    'Gradient Boosting':   'GB',
}
SHORT_TO_FULL = {v: k for k, v in MODEL_SHORT.items()}

# ─── Model factory ────────────────────────────────────────────────────────────
def _make_model(model_name):
    if model_name == 'Decision Tree':
        return DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
    elif model_name == 'Logistic Regression':
        return LogisticRegression(solver='lbfgs', max_iter=300, random_state=42, class_weight='balanced')
    elif model_name == 'SVM':
        return SVC(kernel='rbf', C=5, gamma=0.01, probability=True, random_state=42, class_weight='balanced')
    elif model_name == 'Random Forest':
        return RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
    elif model_name == 'Gradient Boosting':
        return GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, random_state=42)
    raise ValueError(f"Unknown model: {model_name}")


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _split_80_10_10(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _compute_sample_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    weight_map = {cls: len(y) / (len(classes) * cnt)
                  for cls, cnt in zip(classes, counts)}
    return np.array([weight_map[label] for label in y])


# ─── Path helpers ─────────────────────────────────────────────────────────────

def get_model_path(model_name):
    short = MODEL_SHORT.get(model_name, model_name)
    return os.path.join(MODEL_DIR, f"{short}.joblib")


def get_available_models():
    """
    Return list of full model names yang sudah ada file .joblibnya.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    available = []
    for full_name, short in MODEL_SHORT.items():
        path = os.path.join(MODEL_DIR, f"{short}.joblib")
        if os.path.exists(path):
            available.append(full_name)
    return available


# ─── Feature preparation ──────────────────────────────────────────────────────

def prepare_features_from_images(image_bytes_dict, feature_mode='19'):
    """
    Ekstrak fitur dari dict gambar {'Renggang': [bytes,...], ...}.
    feature_mode: '19' atau '39'

    Returns: X_all (np.ndarray), y_all (np.ndarray)
    """
    X_all, y_all = [], []
    for class_name, img_list in image_bytes_dict.items():
        for img_bytes in img_list:
            segmented = preprocess_image(image_bytes=img_bytes)
            if segmented is None:
                continue
            if feature_mode == '39':
                feats = extract_features_with_glcm(segmented)
            else:
                feats = extract_features(segmented)
            X_all.append(feats)
            y_all.append(class_name)
    return np.array(X_all), np.array(y_all)


def apply_information_gain(X_train, y_train, X_val, X_test, feature_names):
    """
    Terapkan SelectKBest (Information Gain) pada training split.
    Returns: selector, X_train_sel, X_val_sel, X_test_sel, selected_names, scores_df
    """
    import pandas as pd

    selector = SelectKBest(score_func=mutual_info_classif, k=N_SELECT_BEST)
    selector.fit(X_train, y_train)

    X_train_sel = selector.transform(X_train)
    X_val_sel   = selector.transform(X_val)
    X_test_sel  = selector.transform(X_test)

    scores = selector.scores_
    support = selector.get_support()
    selected_names = [feature_names[i] for i, s in enumerate(support) if s]

    # DataFrame ranking semua fitur berdasarkan skor IG
    scores_df = pd.DataFrame({
        'Nama Fitur': feature_names,
        'IG Score':   scores,
        'Dipilih':    ['✅ Dipilih' if s else '❌ Tidak Dipilih' for s in support],
    }).sort_values('IG Score', ascending=False).reset_index(drop=True)
    scores_df.index += 1  # mulai dari 1

    return selector, X_train_sel, X_val_sel, X_test_sel, selected_names, scores_df


# ─── Training ─────────────────────────────────────────────────────────────────

def train_single_model(model_name, X_train_sc, X_val_sc, X_test_sc,
                       y_train, y_val, y_test):
    """
    Latih satu model. Return (fitted_model, metrics_dict, cm_dict).
    """
    model = _make_model(model_name)
    t0 = time.time()

    if model_name == 'Gradient Boosting':
        sw = _compute_sample_weights(y_train)
        model.fit(X_train_sc, y_train, sample_weight=sw)
    else:
        model.fit(X_train_sc, y_train)

    exec_time = round(time.time() - t0, 4)

    y_pred_val  = model.predict(X_val_sc)
    y_pred_test = model.predict(X_test_sc)
    # Gunakan union y_train + y_test agar semua kelas terwakili di confusion matrix
    # walau dataset kecil dan salah satu kelas tidak muncul di test split
    unique_labels = sorted(set(y_train) | set(y_test))

    metrics = {
        'Accuracy':           round(accuracy_score(y_test, y_pred_test), 4),
        'Precision':          round(precision_score(y_test, y_pred_test, average='macro', zero_division=0), 4),
        'Recall':             round(recall_score(y_test, y_pred_test, average='macro', zero_division=0), 4),
        'F1-Score':           round(f1_score(y_test, y_pred_test, average='macro', zero_division=0), 4),
        'Val Accuracy':       round(accuracy_score(y_val, y_pred_val), 4),
        'Execution Time (s)': exec_time,
    }
    cm = {
        'matrix': confusion_matrix(y_test, y_pred_test, labels=unique_labels),
        'labels': unique_labels,
    }
    return model, metrics, cm


# ─── Save / Load ──────────────────────────────────────────────────────────────

def save_model_bundle(model_name, model, scaler, selector,
                      feature_mode, features_used, metrics, cm, split_info):
    """
    Simpan satu model beserta semua metadata ke models/{SHORT}.joblib.
    selector = None jika mode 19 fitur (tanpa GLCM).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        'model':         model,
        'scaler':        scaler,
        'selector':      selector,        # None jika 19 fitur
        'feature_mode':  feature_mode,    # '19' atau '39'
        'features_used': features_used,   # nama fitur yang masuk ke model
        'n_features':    len(features_used),
        'metrics':       metrics,
        'confusion_matrix': cm,
        'split_info':    split_info,
    }
    joblib.dump(bundle, get_model_path(model_name))
    return bundle


def load_model_bundle(model_name):
    """
    Load bundle dari models/{SHORT}.joblib.
    Raise FileNotFoundError jika belum ada.
    """
    path = get_model_path(model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model '{model_name}' belum dilatih. "
            "Silakan latih model di menu Alur Pelatihan terlebih dahulu."
        )
    return joblib.load(path)


# ─── Inference ────────────────────────────────────────────────────────────────

def run_inference(bundle, image_bytes):
    """
    Jalankan prediksi pada satu gambar menggunakan bundle model.

    Returns
    -------
    prediction   : str   — 'Renggang' | 'Sedang' | 'Padat'
    features_raw : ndarray — fitur sebelum seleksi (19 atau 39)
    feat_names   : list    — nama fitur raw
    """
    segmented = preprocess_image(image_bytes=image_bytes)
    if segmented is None:
        raise ValueError("Gambar tidak dapat diproses.")

    feature_mode = bundle.get('feature_mode', '19')
    model        = bundle['model']
    scaler       = bundle['scaler']
    selector     = bundle.get('selector')

    if feature_mode == '39':
        features_raw = extract_features_with_glcm(segmented)
        feat_names   = FEATURE_NAMES_39
        features_sel = selector.transform(np.array([features_raw]))[0] if selector else features_raw
    else:
        features_raw = extract_features(segmented)
        feat_names   = FEATURE_NAMES_19
        features_sel = features_raw

    X_scaled   = scaler.transform(np.array([features_sel]))
    prediction = model.predict(X_scaled)[0]
    return prediction, features_raw, feat_names


# ─── CSV helpers (untuk menu Eksperimen) ──────────────────────────────────────

def load_csv_for_experiment(csv_path, feature_mode='39'):
    """
    Load CSV, apply IG jika 39 fitur.
    Returns X (setelah seleksi jika 39), y, selector (atau None), selected_names, scores_df
    """
    import pandas as pd

    df        = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c != 'Class']
    X_all     = df[feat_cols].to_numpy(dtype=float)
    y_all     = df['Class'].to_numpy(dtype=str)

    n_feat = X_all.shape[1]

    if feature_mode == '39' and n_feat == N_FEATURES_WITH_GLCM:
        X_train_raw, _, _, y_train, _, _ = _split_80_10_10(X_all, y_all)
        selector, X_all_sel, _, _, selected_names, scores_df = apply_information_gain(
            X_train_raw, y_train,
            X_all, X_all,   # dummy val/test — hanya butuh transform X_all
            FEATURE_NAMES_39
        )
        # transform seluruh data dengan selector
        X_all_sel = selector.transform(X_all)
        return X_all_sel, y_all, selector, selected_names, scores_df
    else:
        names_19 = FEATURE_NAMES_19[:n_feat]
        return X_all, y_all, None, names_19, None
