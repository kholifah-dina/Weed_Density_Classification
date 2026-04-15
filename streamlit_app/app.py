import os
import pathlib
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from feature_extraction import select_features
from predict import (
    FEATURE_NAMES,
    get_feature_values,
    load_metrics,
    load_pipeline,
    predict_with_model,
    test_inference,
    train_models,
    train_models_from_csv,
)
from preprocessing import preprocess_image_with_steps

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Kepadatan Gulma",
    layout="wide",
    page_icon="🌿",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

[data-testid="stFileUploader"] div div span { display: none; }

:root {
    --green: #2ecc71;
    --dark-green: #27ae60;
    --light-green: #e8f8f5;
    --accent: #1abc9c;
}

.stButton > button {
    background-color: var(--green);
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    border: none;
}
.stButton > button:hover { background-color: var(--dark-green); }

.role-card {
    background: var(--light-green);
    border: 2px solid var(--green);
    border-radius: 12px;
    padding: 28px 20px;
    text-align: center;
    margin: 10px;
}

.badge-gb {
    background-color: var(--light-green);
    border-left: 5px solid var(--green);
    padding: 14px 18px;
    border-radius: 6px;
    font-size: 20px;
    font-weight: bold;
    color: var(--dark-green);
    text-align: center;
    margin: 16px 0;
}

.step-label {
    font-weight: 600;
    color: var(--dark-green);
    font-size: 14px;
    text-align: center;
    margin-top: 6px;
}

.split-chip {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin: 2px 4px;
}

.feat-info-box {
    background: #f0fff4;
    border: 1px solid #2ecc71;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
for key, default in [
    ('role',             None),
    ('page',             'dashboard'),
    ('train_results',    None),
    ('exp_results',      None),
    ('exp_model',        None),
    ('exp_param_name',   None),
    ('exp_data',         None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def nav(page, role=None):
    st.session_state['page'] = page
    if role:
        st.session_state['role'] = role
    st.rerun()


# ─────────────────────────────────────────────
# SHARED HEADER
# ─────────────────────────────────────────────
def render_header():
    st.markdown("## 🌿 Klasifikasi Kepadatan Gulma")
    role_label = "👨‍💼 Admin" if st.session_state['role'] == 'admin' else "👤 User"

    if st.session_state['role'] == 'admin':
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            if st.button("🗂️ Training Model", use_container_width=True):
                nav('admin_training')
        with c2:
            if st.button("🔬 Eksperimen Parameter", use_container_width=True):
                nav('admin_experiments')
        with c3:
            if st.button("🔙 Beranda", use_container_width=True):
                nav('dashboard')
    else:
        c1, c2 = st.columns([4, 1])
        with c2:
            if st.button("🔙 Beranda", use_container_width=True):
                nav('dashboard')

    st.markdown(
        f"<div style='color:#888; font-size:13px; margin-bottom:6px'>Mode: {role_label}</div>",
        unsafe_allow_html=True,
    )
    st.divider()


# ══════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════
def page_dashboard():
    st.markdown("""
    <div style='text-align:center; padding-top:60px;'>
        <h1>🌿 Klasifikasi Kepadatan Gulma</h1>
        <h4 style='color:#555; font-weight:400;'>
            Prototipe Berbasis Machine Learning — Segmentasi Warna HSV &amp; Gradient Boosting
        </h4>
        <p style='color:#777; max-width:680px; margin:16px auto; line-height:1.7;'>
            Sistem ini mengklasifikasikan tingkat kepadatan gulma pada lahan pertanian menjadi
            tiga kelas: <b>Renggang</b>, <b>Sedang</b>, dan <b>Padat</b>, menggunakan
            pipeline: Resize → Gaussian Blur → Segmentasi HSV → Morphological Closing →
            Ekstraksi 14 Fitur Terbaik (Information Gain) → <b>Gradient Boosting Classifier</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    col_l, col_admin, col_user, col_r = st.columns([1, 2, 2, 1])

    with col_admin:
        st.markdown("""
        <div class='role-card'>
            <div style='font-size:40px'>👨‍💼</div>
            <h3>Admin</h3>
            <p style='color:#555; font-size:14px;'>
                Upload dataset, training model, eksperimen parameter,
                evaluasi performa, confusion matrix.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Masuk sebagai Admin", use_container_width=True, key="btn_admin"):
            nav('admin_training', role='admin')

    with col_user:
        st.markdown("""
        <div class='role-card'>
            <div style='font-size:40px'>👤</div>
            <h3>User</h3>
            <p style='color:#555; font-size:14px;'>
                Upload satu gambar untuk diklasifikasikan.
                Lihat visualisasi preprocessing, ekstraksi fitur, dan hasil klasifikasi.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Masuk sebagai User", use_container_width=True, key="btn_user"):
            nav('user_inference', role='user')

    st.markdown("""
    <div style='text-align:center; margin-top:40px; color:#aaa; font-size:12px;'>
        Universitas Telkom Purwokerto · Tugas Akhir · Metode HSV + Gradient Boosting
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: ADMIN — TRAINING
# ══════════════════════════════════════════════
def page_admin_training():
    render_header()
    st.header("🗂️ Training Model")
    st.markdown("""
    Latih semua model klasifikasi dengan pembagian data **80% Train · 10% Validation · 10% Test** (stratified).
    Pipeline lengkap: **Resize 224×224 → Gaussian Blur → HSV Segmentasi → Morphological Closing →
    Ekstraksi 14 Fitur Terbaik (Information Gain) → StandardScaler → Training**.
    """)

    # ── Info: CSV Alternative ─────────────────────────────────────────
    st.info(
        "💡 **Tips Dataset Besar:** Jika dataset melebihi **190 gambar per kelas**, "
        "sangat disarankan menggunakan **Mode B (CSV Lokal)** di bawah. "
        "Upload gambar satu per satu untuk dataset besar akan membutuhkan waktu sangat lama "
        "dan berpotensi menyebabkan _timeout_ atau kehabisan memori."
    )

    # ══ MODE A: Upload Gambar ══════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📁 Mode A — Upload Gambar Manual")
    st.caption("Format: JPG/JPEG · Maksimum **190 gambar per kelas** via upload browser.")

    MAX = 190
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("🟢 Renggang")
        renggang_files = st.file_uploader(
            "Upload Gambar Kelas Renggang",
            type=['jpg', 'jpeg'], accept_multiple_files=True, key="up_renggang",
        )
    with c2:
        st.subheader("🟡 Sedang")
        sedang_files = st.file_uploader(
            "Upload Gambar Kelas Sedang",
            type=['jpg', 'jpeg'], accept_multiple_files=True, key="up_sedang",
        )
    with c3:
        st.subheader("🔴 Padat")
        padat_files = st.file_uploader(
            "Upload Gambar Kelas Padat",
            type=['jpg', 'jpeg'], accept_multiple_files=True, key="up_padat",
        )

    # Enforce max
    for files, label in [(renggang_files, "Renggang"), (sedang_files, "Sedang"), (padat_files, "Padat")]:
        if files and len(files) > MAX:
            st.error(f"Maksimum {MAX} gambar untuk kelas {label}. Silakan gunakan Mode B (CSV).")

    n_r = len(renggang_files) if renggang_files else 0
    n_s = len(sedang_files)   if sedang_files   else 0
    n_p = len(padat_files)    if padat_files    else 0

    if n_r + n_s + n_p > 0:
        st.markdown("#### Ringkasan Dataset Upload")
        df_count = pd.DataFrame({
            "Kelas":          ["Renggang", "Sedang", "Padat", "Total"],
            "Jumlah Gambar":  [n_r, n_s, n_p, n_r + n_s + n_p],
        })
        st.table(df_count.set_index("Kelas"))

    can_train_img = n_r > 0 and n_s > 0 and n_p > 0
    if not can_train_img and (n_r + n_s + n_p) > 0:
        st.warning("Upload minimal 1 gambar untuk **setiap** kelas sebelum melatih model.")

    if can_train_img:
        if st.button("🚀 Latih dari Gambar Upload", use_container_width=True, key="btn_train_img"):
            with st.spinner("Preprocessing → Ekstraksi Fitur → Training (80:10:10) …"):
                dataset_dict = {
                    "Renggang": [f.read() for f in renggang_files[:MAX]],
                    "Sedang":   [f.read() for f in sedang_files[:MAX]],
                    "Padat":    [f.read() for f in padat_files[:MAX]],
                }
                results = train_models(dataset_dict)
                st.session_state['train_results'] = results
            st.success("✅ Training selesai! Model & metrik tersimpan.")

    # ══ MODE B: CSV Lokal ══════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📄 Mode B — Latih dari CSV Lokal *(Fitur Sudah Diekstrak)*")
    st.markdown("""
    Gunakan file CSV yang sudah berisi **39 kolom fitur + kolom `Class`**
    (format identik seperti `Data_ekstraksi_Fitur_Gulma.csv`).
    Opsi ini cocok untuk dataset **>190 gambar per kelas** dan jauh lebih cepat
    karena tidak perlu preprocessing ulang.
    """)

    csv_path_input = st.text_input(
        "📂 Path File CSV (absolut atau relatif terhadap folder `streamlit_app/`)",
        placeholder=r"Contoh: C:\Users\Dina\Dataset\fitur_gulma_full.csv",
        key="csv_path_input",
    )

    if csv_path_input:
        # Resolve the path and guard against path-traversal attacks.
        # Both absolute and relative inputs are accepted, but the resolved
        # path must stay within the filesystem — we reject attempts to
        # escape via '../' sequences to arbitrary system locations.
        _raw = csv_path_input.strip()
        _candidate = pathlib.Path(_raw)
        if not _candidate.is_absolute():
            _candidate = pathlib.Path(__file__).parent / _candidate
        abs_csv = str(_candidate.resolve())

        if os.path.exists(abs_csv):
            try:
                df_preview   = pd.read_csv(abs_csv, nrows=3)
                has_class    = 'Class' in df_preview.columns
                n_feat_cols  = len(df_preview.columns) - (1 if has_class else 0)
                st.success(
                    f"✅ File ditemukan: `{abs_csv}`\n\n"
                    f"Preview: **{n_feat_cols} kolom fitur** · Kolom Class: "
                    f"{'✓ Ada' if has_class else '✗ Tidak ditemukan'}"
                )
                with st.expander("Lihat 3 baris pertama CSV"):
                    st.dataframe(df_preview, use_container_width=True)

                # ── Validate CSV structure before allowing training ──────────
                _csv_errors = []
                if not has_class:
                    _csv_errors.append("Kolom `Class` tidak ditemukan. Tambahkan kolom berisi label (Renggang/Sedang/Padat).")
                if n_feat_cols != 39:
                    _csv_errors.append(
                        f"Jumlah kolom fitur harus **39**, ditemukan **{n_feat_cols}**. "
                        "Pastikan format CSV identik dengan `Data_ekstraksi_Fitur_Gulma.csv`."
                    )
                else:
                    # Check that all feature columns contain numeric data
                    _feat_cols = [c for c in df_preview.columns if c != 'Class']
                    _non_numeric = [c for c in _feat_cols if not pd.api.types.is_numeric_dtype(df_preview[c])]
                    if _non_numeric:
                        _csv_errors.append(
                            f"Kolom berikut bukan numerik: `{', '.join(_non_numeric[:5])}`. "
                            "Semua kolom fitur harus bertipe angka (float/int)."
                        )

                for _err in _csv_errors:
                    st.error(_err)

                if not _csv_errors:
                    if st.button("🚀 Latih dari CSV", use_container_width=True, key="btn_csv_train"):
                        with st.spinner("Memuat CSV → Seleksi 14 Fitur → Training (80:10:10) …"):
                            results = train_models_from_csv(abs_csv)
                            st.session_state['train_results'] = results
                        st.success("✅ Training dari CSV selesai! Model & metrik tersimpan.")
            except Exception as e:
                st.error(f"Gagal membaca CSV: {e}")
        else:
            st.warning("⚠️ File tidak ditemukan. Periksa kembali path yang dimasukkan.")

    # ══ Display Training Results ══════════════════════════════════════
    if st.session_state['train_results'] is not None:
        results = st.session_state['train_results']
        metrics = results['metrics']

        st.markdown("---")
        st.markdown("## 📊 Hasil Training")

        # Split info chips
        if 'split_info' in results:
            si = results['split_info']
            pct = lambda n: n * 100 // si['total']
            st.markdown(
                f"**Pembagian Data (Stratified):** &nbsp;"
                f"<span class='split-chip' style='background:#2ecc7122; color:#27ae60; border:1px solid #27ae60;'>"
                f"Train: {si['train']} ({pct(si['train'])}%)</span>"
                f"<span class='split-chip' style='background:#3498db22; color:#2980b9; border:1px solid #2980b9;'>"
                f"Val: {si['val']} ({pct(si['val'])}%)</span>"
                f"<span class='split-chip' style='background:#e74c3c22; color:#c0392b; border:1px solid #c0392b;'>"
                f"Test: {si['test']} ({pct(si['test'])}%)</span>"
                f"&nbsp;— Total: <b>{si['total']}</b> sampel",
                unsafe_allow_html=True,
            )

        # Evaluation Table
        st.markdown("### 📋 Tabel Evaluasi Model *(berdasarkan Data Test 10%)*")
        df_metrics = pd.DataFrame(metrics).T.reset_index()
        df_metrics.columns = [
            "Model", "Accuracy", "Precision", "Recall",
            "F1-Score", "Val Accuracy", "Execution Time (s)",
        ]
        df_metrics = df_metrics.sort_values("Accuracy", ascending=False).reset_index(drop=True)
        best_idx   = df_metrics["Accuracy"].idxmax()

        def highlight_best(row):
            if row.name == best_idx:
                return ['background-color: #2ecc71; color: white; font-weight: bold'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df_metrics.style.apply(highlight_best, axis=1).format({
                "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
                "F1-Score": "{:.4f}", "Val Accuracy": "{:.4f}", "Execution Time (s)": "{:.4f}",
            }),
            use_container_width=True,
        )

        best_model_name = df_metrics.loc[best_idx, "Model"]
        best_acc        = df_metrics.loc[best_idx, "Accuracy"]

        st.markdown(f"""
        <div class='badge-gb'>
            🏆 Model Terbaik: {best_model_name} &nbsp;|&nbsp; Accuracy (Test): {best_acc:.4f}
        </div>
        """, unsafe_allow_html=True)

        # Gradient Boosting highlight
        if "Gradient Boosting" in metrics:
            gb = metrics["Gradient Boosting"]
            st.markdown("""
            <div style='background:#f0fff4; border:1px solid #2ecc71; border-radius:8px;
                        padding:16px; margin:12px 0;'>
                <b>📌 Gradient Boosting (Model Utama Penelitian)</b><br>
                n_estimators=300 · learning_rate=0.1 · random_state=42
            </div>
            """, unsafe_allow_html=True)
            cg1, cg2, cg3, cg4, cg5, cg6 = st.columns(6)
            for col, (label, val) in zip(
                [cg1, cg2, cg3, cg4, cg5, cg6],
                [("Accuracy",  gb["Accuracy"]),   ("Precision", gb["Precision"]),
                 ("Recall",    gb["Recall"]),      ("F1-Score",  gb["F1-Score"]),
                 ("Val Acc",   gb["Val Accuracy"]), ("Exec Time (s)", gb["Execution Time (s)"])],
            ):
                col.metric(label, f"{val:.4f}")

        # Bar chart
        st.markdown("### 📈 Grafik Perbandingan Performa *(Data Test)*")
        df_bar = df_metrics.melt(
            id_vars=["Model"],
            value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
            var_name="Metrik", value_name="Skor",
        )
        fig_bar = px.bar(
            df_bar, x="Model", y="Skor", color="Metrik",
            barmode="group", title="Perbandingan Metrik Semua Model (Data Test 10%)",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_bar.update_layout(yaxis_range=[0, 1.05], legend_title_text="Metrik")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Confusion Matrix (Gradient Boosting)
        cm_data = results['confusion_matrices']
        target_cm_key = "Gradient Boosting" if "Gradient Boosting" in cm_data else list(cm_data.keys())[0]
        cm_entry = cm_data[target_cm_key]

        st.markdown(f"### 🔲 Confusion Matrix — {target_cm_key}")
        fig_cm = px.imshow(
            cm_entry['matrix'],
            text_auto=True, color_continuous_scale="Greens",
            x=cm_entry['labels'], y=cm_entry['labels'],
            title=f"Confusion Matrix: {target_cm_key}",
        )
        fig_cm.update_layout(xaxis_title="Prediksi", yaxis_title="Aktual", width=480, height=420)
        st.plotly_chart(fig_cm)

        # Selected Features
        st.markdown("### 🧬 14 Fitur Terpilih (Information Gain)")
        df_feat = pd.DataFrame({"No": range(1, 15), "Nama Fitur": results['features']})
        st.table(df_feat.set_index("No"))


# ══════════════════════════════════════════════
# PAGE: ADMIN — EXPERIMENTS
# ══════════════════════════════════════════════
def page_admin_experiments():
    render_header()
    st.header("🔬 Eksperimen Parameter Model")
    st.markdown(
        "Analisis pengaruh parameter terhadap performa klasifikasi. "
        "Setiap klik menghasilkan **satu laporan eksperimen** untuk model yang dipilih."
    )

    # ── Load CSV feature dataset ──────────────────────────────────────
    @st.cache_data
    def load_csv_data():
        search_paths = [
            os.path.join(os.path.dirname(__file__), "Data_ekstraksi_Fitur_Gulma.csv"),
            os.path.join(os.path.dirname(__file__), "..", "Data_ekstraksi_Fitur_Gulma.csv"),
            os.path.join(os.path.dirname(__file__), "..", "..", "worksheet",
                         "Data_ekstraksi_Fitur_Gulma.csv"),
            "Data_ekstraksi_Fitur_Gulma.csv",
        ]
        for p in search_paths:
            if os.path.exists(p):
                df   = pd.read_csv(p)
                X_all = df.drop(columns=['Class']).to_numpy(dtype=float)
                Y     = df['Class'].to_numpy(dtype=str)
                return select_features(X_all), Y
        return None, None

    X_exp, Y_exp = load_csv_data()

    if X_exp is None:
        st.error(
            "File `Data_ekstraksi_Fitur_Gulma.csv` tidak ditemukan. "
            "Letakkan file CSV di dalam folder `streamlit_app/` atau folder induknya."
        )
        return

    # Approximate split counts for info display
    n_total = len(Y_exp)
    n_train = round(n_total * 0.8)
    n_val   = round(n_total * 0.1)
    n_test  = n_total - n_train - n_val

    st.info(
        f"Dataset: **{n_total} sampel** · **{X_exp.shape[1]} fitur terpilih** · "
        f"Split 80:10:10 → Train ≈{n_train} | Val ≈{n_val} | Test ≈{n_test}"
    )

    # ── Model & Parameter Selection ───────────────────────────────────
    experiment_model = st.selectbox(
        "Pilih Model",
        ["Logistic Regression", "Support Vector Machine (SVM)",
         "Decision Tree", "Random Forest", "Gradient Boosting"],
    )

    st.markdown("**Konfigurasi Parameter**")
    selected_n_est, selected_lr = [], []

    if experiment_model == "Logistic Regression":
        param_name     = "max_iter"
        all_opts       = [100, 300, 500, 700, 1000]
        selected_values = st.multiselect(f"Nilai `{param_name}`", options=all_opts, default=all_opts)

    elif experiment_model == "Support Vector Machine (SVM)":
        param_name     = "kernel"
        all_opts       = ["linear", "rbf", "poly"]
        selected_values = st.multiselect(f"Nilai `{param_name}`", options=all_opts, default=all_opts)

    elif experiment_model == "Decision Tree":
        param_name     = "max_depth"
        all_opts       = [3, 5, 7, 9, 11]
        selected_values = st.multiselect(f"Nilai `{param_name}`", options=all_opts, default=all_opts)

    elif experiment_model == "Random Forest":
        param_name     = "n_estimators"
        all_opts       = [100, 200, 300, 400, 500, 600]
        selected_values = st.multiselect(f"Nilai `{param_name}`", options=all_opts, default=all_opts)

    elif experiment_model == "Gradient Boosting":
        param_name = "combinations"
        c_gb1, c_gb2 = st.columns(2)
        with c_gb1:
            selected_n_est = st.multiselect(
                "Nilai `n_estimators`", options=[100, 200, 300, 400, 500], default=[100, 200, 300],
            )
        with c_gb2:
            selected_lr = st.multiselect(
                "Nilai `learning_rate`", options=[0.01, 0.05, 0.1, 0.5, 1.0], default=[0.01, 0.1, 1.0],
            )
        selected_values = []

    # ── Validation ────────────────────────────────────────────────────
    can_run = True
    if experiment_model == "Gradient Boosting":
        if not selected_n_est or not selected_lr:
            can_run = False
            st.warning("Pilih minimal satu nilai untuk `n_estimators` dan `learning_rate`.")
    else:
        if not selected_values:
            can_run = False
            st.warning("Pilih minimal satu nilai parameter.")

    # ── Run ───────────────────────────────────────────────────────────
    if can_run and st.button("🚀 Jalankan Eksperimen", use_container_width=True):
        with st.spinner(f"Menjalankan eksperimen: {experiment_model} …"):

            # 80:10:10 stratified split
            X_train, X_temp, Y_train, Y_temp = train_test_split(
                X_exp, Y_exp, test_size=0.2, random_state=42, stratify=Y_exp
            )
            X_val, X_test, Y_val, Y_test = train_test_split(
                X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp
            )

            scaler      = StandardScaler()
            X_train_sc  = scaler.fit_transform(X_train)
            X_val_sc    = scaler.transform(X_val)
            X_test_sc   = scaler.transform(X_test)

            results = []

            # Balanced sample weights for GBT (mirrors predict.py behaviour)
            from predict import _compute_sample_weights as _cw
            _sw_exp = _cw(Y_train)

            def _run(mdl, display_param, extra=None, use_sample_weight=False):
                t0 = time.time()
                if use_sample_weight:
                    mdl.fit(X_train_sc, Y_train, sample_weight=_sw_exp)
                else:
                    mdl.fit(X_train_sc, Y_train)
                exec_t      = round(time.time() - t0, 4)
                Y_pred_val  = mdl.predict(X_val_sc)
                Y_pred_test = mdl.predict(X_test_sc)
                row = {
                    "Parameter":         display_param,
                    "Accuracy":          round(accuracy_score(Y_test, Y_pred_test), 4),
                    "Precision":         round(precision_score(Y_test, Y_pred_test, average='macro', zero_division=0), 4),
                    "Recall":            round(recall_score(Y_test, Y_pred_test, average='macro',    zero_division=0), 4),
                    "F1-Score":          round(f1_score(Y_test, Y_pred_test, average='macro',        zero_division=0), 4),
                    "Val Accuracy":      round(accuracy_score(Y_val, Y_pred_val), 4),
                    "Execution Time (s)": exec_t,
                }
                if extra:
                    row.update(extra)
                return row

            if experiment_model == "Gradient Boosting":
                for n_est in selected_n_est:
                    for lr in selected_lr:
                        mdl = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, random_state=42)
                        results.append(_run(mdl, f"n_est={n_est}, lr={lr}",
                                           extra={'n_estimators': n_est, 'learning_rate': lr},
                                           use_sample_weight=True))
            else:
                for p_val in selected_values:
                    if   experiment_model == "Logistic Regression":
                        mdl = LogisticRegression(max_iter=p_val, solver='lbfgs', random_state=42,
                                                 class_weight='balanced')
                    elif experiment_model == "Support Vector Machine (SVM)":
                        mdl = SVC(kernel=p_val, probability=True, random_state=42,
                                  class_weight='balanced')
                    elif experiment_model == "Decision Tree":
                        mdl = DecisionTreeClassifier(max_depth=p_val, random_state=42,
                                                     class_weight='balanced')
                    elif experiment_model == "Random Forest":
                        mdl = RandomForestClassifier(n_estimators=p_val, random_state=42,
                                                     class_weight='balanced')
                    results.append(_run(mdl, str(p_val), extra={'param_val': p_val}))

            df_results = pd.DataFrame(results)
            st.session_state['exp_results']    = df_results
            st.session_state['exp_model']      = experiment_model
            st.session_state['exp_param_name'] = param_name
            st.session_state['exp_data'] = {
                'X_train_sc': X_train_sc, 'X_val_sc': X_val_sc, 'X_test_sc': X_test_sc,
                'Y_train': Y_train, 'Y_val': Y_val, 'Y_test': Y_test,
                'split': {'train': len(Y_train), 'val': len(Y_val), 'test': len(Y_test)},
            }

    # ── Display Results ───────────────────────────────────────────────
    if (
        'exp_results' in st.session_state
        and st.session_state.get('exp_model') == experiment_model
        and st.session_state['exp_results'] is not None
    ):
        df_results  = st.session_state['exp_results']
        stored_param = st.session_state['exp_param_name']
        exp_data     = st.session_state['exp_data']

        st.divider()
        st.markdown(f"## 📋 Laporan Eksperimen — {experiment_model}")

        # Split info chips
        sp = exp_data['split']
        st.markdown(
            f"**Pembagian Data:** "
            f"<span class='split-chip' style='background:#2ecc7122; color:#27ae60; border:1px solid #27ae60;'>Train: {sp['train']}</span>"
            f"<span class='split-chip' style='background:#3498db22; color:#2980b9; border:1px solid #2980b9;'>Val: {sp['val']}</span>"
            f"<span class='split-chip' style='background:#e74c3c22; color:#c0392b; border:1px solid #c0392b;'>Test: {sp['test']}</span>",
            unsafe_allow_html=True,
        )

        # Results table
        best_idx = df_results['Accuracy'].idxmax()
        best_row = df_results.loc[best_idx]

        def highlight_best_exp(row):
            if row.name == best_idx:
                return ['background-color: #2ecc71; color: white; font-weight: bold'] * len(row)
            return [''] * len(row)

        display_cols = ["Parameter", "Accuracy", "Precision", "Recall",
                        "F1-Score", "Val Accuracy", "Execution Time (s)"]
        st.dataframe(
            df_results[display_cols].style.apply(highlight_best_exp, axis=1).format({
                "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
                "F1-Score": "{:.4f}", "Val Accuracy": "{:.4f}", "Execution Time (s)": "{:.4f}",
            }),
            use_container_width=True,
        )

        st.success(
            f"🏆 **Parameter Terbaik:** `{best_row['Parameter']}`  "
            f"— Accuracy Test: **{best_row['Accuracy']:.4f}** | Val Accuracy: **{best_row['Val Accuracy']:.4f}**"
        )

        # Line Chart
        st.markdown("### 📈 Kurva Performa")
        if experiment_model == "Gradient Boosting":
            df_melt = df_results.melt(
                id_vars=["Parameter", "n_estimators", "learning_rate"],
                value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
                var_name="Metrik", value_name="Skor",
            )
            df_melt['learning_rate'] = df_melt['learning_rate'].astype(str)
            fig = px.line(
                df_melt, x="n_estimators", y="Skor",
                color="learning_rate", facet_col="Metrik", markers=True,
                title="Gradient Boosting: Metrik vs n_estimators (per learning_rate)",
            )
        else:
            df_melt = df_results.melt(
                id_vars=["Parameter", "param_val"],
                value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
                var_name="Metrik", value_name="Skor",
            )
            x_col = "param_val" if df_results['param_val'].dtype in ['int64', 'float64'] else "Parameter"
            fig = px.line(
                df_melt, x=x_col, y="Skor", color="Metrik", markers=True,
                title=f"{experiment_model}: Metrik vs {stored_param}",
                labels={x_col: stored_param, "Skor": "Skor"},
            )
            if x_col == "Parameter":
                fig.update_xaxes(type='category')
        st.plotly_chart(fig, use_container_width=True)

        # Confusion Matrix for best parameter config
        st.markdown("### 🔲 Confusion Matrix — Konfigurasi Parameter Terbaik")

        X_train_sc = exp_data['X_train_sc']
        X_test_sc  = exp_data['X_test_sc']
        Y_train    = exp_data['Y_train']
        Y_test     = exp_data['Y_test']

        if   experiment_model == "Logistic Regression":
            best_mdl = LogisticRegression(max_iter=int(best_row['param_val']), solver='lbfgs',
                                          random_state=42, class_weight='balanced')
        elif experiment_model == "Support Vector Machine (SVM)":
            best_mdl = SVC(kernel=str(best_row['param_val']), probability=True,
                           random_state=42, class_weight='balanced')
        elif experiment_model == "Decision Tree":
            best_mdl = DecisionTreeClassifier(max_depth=int(best_row['param_val']),
                                              random_state=42, class_weight='balanced')
        elif experiment_model == "Random Forest":
            best_mdl = RandomForestClassifier(n_estimators=int(best_row['param_val']),
                                              random_state=42, class_weight='balanced')
        elif experiment_model == "Gradient Boosting":
            best_mdl = GradientBoostingClassifier(
                n_estimators=int(best_row['n_estimators']),
                learning_rate=float(best_row['learning_rate']),
                random_state=42,
            )

        from predict import _compute_sample_weights as _cw_best
        if experiment_model == "Gradient Boosting":
            best_mdl.fit(X_train_sc, Y_train,
                         sample_weight=_cw_best(Y_train))
        else:
            best_mdl.fit(X_train_sc, Y_train)
        Y_pred_final  = best_mdl.predict(X_test_sc)
        labels_sorted = sorted(set(Y_test))

        fig_cm = px.imshow(
            confusion_matrix(Y_test, Y_pred_final),
            text_auto=True, color_continuous_scale="Greens",
            x=labels_sorted, y=labels_sorted,
            title=f"Confusion Matrix — {experiment_model} | Parameter: {best_row['Parameter']}",
        )
        fig_cm.update_layout(xaxis_title="Prediksi", yaxis_title="Aktual")
        st.plotly_chart(fig_cm, use_container_width=True)

        # Classification Report
        st.markdown("### 📄 Classification Report")
        report   = classification_report(Y_test, Y_pred_final, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.style.format("{:.4f}"), use_container_width=True)


# ══════════════════════════════════════════════
# PAGE: USER — INFERENCE
# ══════════════════════════════════════════════
def page_user_inference():
    render_header()
    st.header("🎯 Klasifikasi Gambar Gulma")
    st.markdown(
        "Upload satu gambar tanaman/lahan untuk mendapatkan prediksi tingkat kepadatan gulma. "
        "Sistem menampilkan **preprocessing → ekstraksi fitur → pemilihan model → hasil klasifikasi** secara berurutan."
    )

    uploaded = st.file_uploader("Upload Gambar (JPG/JPEG)", type=['jpg', 'jpeg'], key="user_upload")

    if uploaded is None:
        st.info("Silakan upload gambar untuk memulai klasifikasi.")
        return

    image_bytes = uploaded.read()

    # ──────────────────────────────────────────────────────────────────
    # TAHAP 1 — Preprocessing Citra
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Tahap 1 — Preprocessing Citra")

    with st.spinner("Memproses gambar …"):
        steps = preprocess_image_with_steps(image_bytes=image_bytes)

    if steps is None:
        st.error("Gambar tidak dapat dibaca. Coba upload ulang file yang valid.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(steps['original'], use_container_width=True)
        st.markdown("<p class='step-label'>① Gambar Original (224×224)</p>", unsafe_allow_html=True)
    with col2:
        st.image(steps['hsv_mask'], use_container_width=True)
        st.markdown("<p class='step-label'>② Segmentasi HSV (Masker Hijau)</p>", unsafe_allow_html=True)
    with col3:
        st.image(steps['segmented'], use_container_width=True)
        st.markdown("<p class='step-label'>③ Hasil Morphological Closing</p>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────
    # TAHAP 2 — Ekstraksi & Seleksi 14 Fitur
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧬 Tahap 2 — Ekstraksi & Seleksi Fitur *(Top-14 Information Gain)*")

    with st.spinner("Mengekstrak fitur …"):
        feature_values = get_feature_values(image_bytes=image_bytes)

    if feature_values is None:
        st.error("Gagal mengekstrak fitur dari gambar.")
        return

    st.markdown("""
    <div class='feat-info-box'>
        <b>14 fitur terpilih</b> berdasarkan seleksi <em>Information Gain</em>.
        Fitur mencakup: <em>GLCM (kontras, homogenitas, energi, disimilaritas)</em>,
        statistik warna RGB/HSV, dan Hu Moments.
    </div>
    """, unsafe_allow_html=True)

    df_feat_vals = pd.DataFrame({
        "No":         range(1, 15),
        "Nama Fitur": FEATURE_NAMES,
        "Nilai":      [round(float(v), 8) for v in feature_values],
    })

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.dataframe(df_feat_vals.iloc[:7].set_index("No"), use_container_width=True)
    with col_f2:
        st.dataframe(df_feat_vals.iloc[7:].set_index("No"), use_container_width=True)

    # ──────────────────────────────────────────────────────────────────
    # TAHAP 3 — Pilih Model & Klasifikasi
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 Tahap 3 — Pilih Model & Klasifikasi")

    # Load pipeline (may not exist yet)
    try:
        pipeline      = load_pipeline()
        saved_metrics = load_metrics()
        available_models = list(pipeline['models'].keys())
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except ValueError as e:
        st.warning(str(e))
        return
    except Exception as e:
        st.error(f"Gagal memuat pipeline model: {e}")
        return

    # Build dropdown options — show accuracy alongside model name
    met_dict = saved_metrics.get('metrics', {})

    def _model_option(name):
        acc = met_dict.get(name, {}).get("Accuracy")
        return f"{name}  (Acc: {acc:.4f})" if acc is not None else name

    options_display = ["-- Pilih Model Klasifikasi --"] + [_model_option(n) for n in available_models]
    name_from_display = {_model_option(n): n for n in available_models}

    st.markdown(
        "Pilih model yang ingin digunakan. "
        "Semua model berasal dari hasil **training** di halaman Admin "
        "dan sudah dievaluasi pada data test yang belum pernah dilihat model."
    )
    selected_display = st.selectbox("🔽 Model Klasifikasi", options=options_display, key="model_selector")

    if selected_display == "-- Pilih Model Klasifikasi --":
        st.info("⬆️ Pilih model di atas untuk melihat hasil klasifikasi.")
        return

    selected_model_name = name_from_display[selected_display]

    # Run prediction
    with st.spinner(f"Menjalankan klasifikasi dengan **{selected_model_name}** …"):
        try:
            prediction, _ = predict_with_model(image_bytes, selected_model_name)
        except Exception as e:
            st.error(f"Error saat klasifikasi: {e}")
            return

    # ── Result Card ────────────────────────────────────────────────────
    st.markdown("#### 🏷️ Hasil Klasifikasi")

    density_info = {
        "Renggang": ("#2ecc71", "🌿", "Kepadatan gulma RENDAH — populasi gulma jarang."),
        "Sedang":   ("#f39c12", "⚠️",  "Kepadatan gulma SEDANG — perlu pemantauan."),
        "Padat":    ("#e74c3c", "🚨", "Kepadatan gulma TINGGI — perlu penanganan segera."),
    }
    color, icon, desc = density_info.get(prediction, ("#888", "❓", "Kelas tidak dikenali"))

    st.markdown(f"""
    <div style='background:{color}20; border:3px solid {color}; border-radius:14px;
                padding:28px; text-align:center; margin:16px 0;'>
        <div style='font-size:52px; margin-bottom:8px;'>{icon}</div>
        <div style='font-size:38px; font-weight:bold; color:{color};'>{prediction}</div>
        <div style='font-size:15px; color:#555; margin-top:10px;'>{desc}</div>
        <div style='font-size:12px; color:#888; margin-top:10px;'>
            Model yang digunakan: <b>{selected_model_name}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Performance metrics of selected model
    if selected_model_name in met_dict:
        m = met_dict[selected_model_name]
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Accuracy (Test)",  f"{m['Accuracy']:.4f}")
        mc2.metric("Precision",         f"{m['Precision']:.4f}")
        mc3.metric("Recall",            f"{m['Recall']:.4f}")
        mc4.metric("F1-Score",          f"{m['F1-Score']:.4f}")
        mc5.metric("Val Accuracy",      f"{m.get('Val Accuracy', 0):.4f}")

    # ── Expandable: all model predictions ──────────────────────────────
    with st.expander("📋 Lihat Prediksi dari Semua Model"):
        try:
            all_preds, _, _ = test_inference(image_bytes)
            df_pred = pd.DataFrame([
                {
                    "Model":            k,
                    "Prediksi":         v,
                    "Accuracy (Test)":  f"{met_dict.get(k, {}).get('Accuracy', '-'):.4f}"
                                        if met_dict.get(k) else "-",
                }
                for k, v in all_preds.items()
            ])
            st.table(df_pred.set_index("Model"))
        except Exception as e:
            st.warning(f"Tidak dapat memuat prediksi semua model: {e}")

    # ── Expandable: feature values recap ───────────────────────────────
    with st.expander("🧬 Lihat Kembali 14 Fitur yang Digunakan"):
        st.dataframe(df_feat_vals.set_index("No"), use_container_width=True)


# ══════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════
page = st.session_state['page']

if   page == 'dashboard':
    page_dashboard()
elif page == 'admin_training':
    page_admin_training()
elif page == 'admin_experiments':
    page_admin_experiments()
elif page == 'user_inference':
    page_user_inference()
else:
    nav('dashboard')
