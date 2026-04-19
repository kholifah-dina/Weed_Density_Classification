import os
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

from predict import (
    FEATURE_NAMES_19,
    FEATURE_NAMES_39,
    MODEL_SHORT,
    N_FEATURES_WITHOUT_GLCM,
    N_SELECT_BEST,
    _compute_sample_weights,
    _split_80_10_10,
    apply_information_gain,
    get_available_models,
    get_model_path,
    load_csv_for_experiment,
    load_model_bundle,
    prepare_features_from_images,
    run_inference,
    save_model_bundle,
    train_single_model,
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

.step-box {
    background: var(--light-green);
    border-left: 5px solid var(--green);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}

.step-label {
    font-weight: 600;
    color: var(--dark-green);
    font-size: 13px;
    text-align: center;
    margin-top: 6px;
}

.step-desc {
    font-size: 12px;
    color: #555;
    text-align: center;
    margin-top: 2px;
    line-height: 1.5;
}

.badge-best {
    background-color: var(--light-green);
    border-left: 5px solid var(--green);
    padding: 14px 18px;
    border-radius: 6px;
    font-size: 18px;
    font-weight: bold;
    color: var(--dark-green);
    text-align: center;
    margin: 16px 0;
}

.split-chip {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin: 2px 4px;
}

.info-box {
    background: #f0fff4;
    border: 1px solid #2ecc71;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 14px;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #2c3e50;
    border-bottom: 2px solid var(--green);
    padding-bottom: 6px;
    margin: 20px 0 12px 0;
}

/* ── Dark mode overrides ─────────────────────────────────────────── */
@media (prefers-color-scheme: dark) {
    :root {
        --light-green: rgba(46, 204, 113, 0.12);
    }
    .step-desc       { color: #b0b8c1 !important; }
    .step-label      { color: #a8e6cf !important; }
    .step-box        { background: var(--light-green) !important; color: #d0d8e0 !important; }
    .badge-best      { background: var(--light-green) !important; color: #a8e6cf !important; }
    .info-box        { background: rgba(46, 204, 113, 0.08) !important; color: #c8d6e0 !important; }
    .section-title   { color: #d0d8e0 !important; }
    .split-chip      { color: inherit !important; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
_defaults = {
    'page':              'training',
    # Menu 1 — Training
    'train_images':      None,       # dict {kelas: [bytes]}
    'train_step':        1,          # langkah aktif (1-4)
    'train_feat_mode':   '39',       # '19' atau '39'
    'train_ig_df':       None,       # DataFrame ranking IG
    'train_sel_names':   None,       # nama 14 fitur terpilih
    'train_X':           None,       # X setelah split & seleksi
    'train_y':           None,
    'train_scaler':      None,
    'train_selector':    None,
    'train_splits':      None,       # dict splits X/y
    'train_result':      None,       # hasil training terakhir
    'train_sample_steps': None,      # preprocessing steps dari 1 sample
    'train_feat_sample': None,       # sample DataFrame hasil ekstraksi fitur
    # Menu 3 — Eksperimen
    'exp_history':       [],         # list semua hasil eksperimen
    # Menu 2 — Testing
    'test_image_bytes':  None,
    'test_steps':        None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def nav(page):
    st.session_state['page'] = page
    st.rerun()


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Kepadatan Gulma")
    st.markdown("---")

    menu_items = [
        ('training',    '📚 Alur Pelatihan'),
        ('testing',     '🎯 Pengujian Gambar'),
        ('experiments', '🔬 Eksperimen Parameter'),
        ('dashboard',   '📊 Dashboard Hasil'),
    ]
    for page_key, label in menu_items:
        is_active = st.session_state['page'] == page_key
        if st.button(
            label,
            use_container_width=True,
            key=f"nav_{page_key}",
            type="primary" if is_active else "secondary",
        ):
            nav(page_key)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px; color:#aaa; text-align:center;'>"
        "Universitas Telkom Purwokerto<br>Tugas Akhir · HSV + Gradient Boosting"
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — Visualisasi Preprocessing (dipakai di Training & Testing)
# ══════════════════════════════════════════════════════════════════════════════
def render_preprocessing_steps(steps, title="🔍 Tahap Preprocessing Citra"):
    if title:
        st.markdown(f"### {title}")
    c1, c2, c3, c4 = st.columns(4)
    images_info = [
        (c1, steps['original'],  "① Resize 224×224",
         "Gambar diubah ukuran menjadi 224×224 piksel sebagai input standar sistem."),
        (c2, steps['blurred'],   "② Gaussian Blur (5×5)",
         "Gaussian Blur mengurangi noise pada gambar agar segmentasi warna lebih akurat."),
        (c3, steps['hsv_mask'],  "③ HSV Thresholding",
         "Piksel hijau (H:25–75°, S:40–255, V:50–255) diisolasi sebagai area gulma."),
        (c4, steps['segmented'], "④ Morphological Closing",
         "Morphological Closing (5×5) mengisi celah kecil pada mask gulma untuk hasil lebih bersih."),
    ]
    for col, img, label, desc in images_info:
        with col:
            st.image(img, use_container_width=True)
            st.markdown(f"<p class='step-label'>{label}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='step-desc'>{desc}</p>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MENU 1 — ALUR PELATIHAN
# ══════════════════════════════════════════════════════════════════════════════
def page_training():
    st.title("📚 Alur Pelatihan Model")
    st.markdown(
        "Ikuti langkah-langkah berikut secara berurutan untuk melatih model klasifikasi. "
        "Setiap tahap akan menampilkan visualisasi dan penjelasan proses yang terjadi."
    )

    step = st.session_state['train_step']

    # ── Progress indicator ──────────────────────────────────────────────
    steps_labels = ["① Upload Dataset", "② Preprocessing", "③ Ekstraksi Fitur", "④ Pilih & Latih Model"]
    cols_prog = st.columns(4)
    for i, (col, lbl) in enumerate(zip(cols_prog, steps_labels), start=1):
        with col:
            color = "#2ecc71" if i <= step else "#ddd"
            text_color = "white" if i <= step else "#999"
            st.markdown(
                f"<div style='background:{color}; color:{text_color}; border-radius:8px; "
                f"padding:8px; text-align:center; font-size:13px; font-weight:600;'>{lbl}</div>",
                unsafe_allow_html=True,
            )
    st.markdown("---")

    # ══ STEP 1 — Upload Dataset ════════════════════════════════════════
    if step >= 1:
        st.markdown("### Langkah 1 — Upload Dataset Gambar")
        st.markdown(
            "Upload gambar untuk setiap kelas gulma. Sistem akan memproses gambar-gambar ini "
            "untuk melatih model klasifikasi."
        )
        MAX = 190
        st.info("📎 Format yang diterima: **JPG / JPEG** · Maksimal **190 gambar per kelas** · Minimal **9 gambar per kelas**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**🟢 Renggang** — Populasi gulma jarang")
            renggang = st.file_uploader("Upload Kelas Renggang (maks. 190 gambar, JPG)", type=['jpg','jpeg'],
                                         accept_multiple_files=True, key="up_renggang")
        with c2:
            st.markdown("**🟡 Sedang** — Perlu pemantauan")
            sedang = st.file_uploader("Upload Kelas Sedang (maks. 190 gambar, JPG)", type=['jpg','jpeg'],
                                       accept_multiple_files=True, key="up_sedang")
        with c3:
            st.markdown("**🔴 Padat** — Perlu penanganan segera")
            padat = st.file_uploader("Upload Kelas Padat (maks. 190 gambar, JPG)", type=['jpg','jpeg'],
                                      accept_multiple_files=True, key="up_padat")

        for files, label in [(renggang,"Renggang"),(sedang,"Sedang"),(padat,"Padat")]:
            if files and len(files) > MAX:
                st.error(f"Maksimum {MAX} gambar untuk kelas {label}.")

        n_r = len(renggang) if renggang else 0
        n_s = len(sedang)   if sedang   else 0
        n_p = len(padat)    if padat    else 0
        total = n_r + n_s + n_p

        if total > 0:
            df_cnt = pd.DataFrame({
                "Kelas":         ["🟢 Renggang","🟡 Sedang","🔴 Padat","Total"],
                "Jumlah Gambar": [n_r, n_s, n_p, total],
            })
            st.table(df_cnt.set_index("Kelas"))

        can_next = n_r > 0 and n_s > 0 and n_p > 0
        if not can_next and total > 0:
            st.warning("Upload minimal 1 gambar untuk **setiap** kelas sebelum melanjutkan.")

        if can_next:
            if st.button("Lanjut ke Preprocessing →", use_container_width=True, key="btn_step1"):
                st.session_state['train_images'] = {
                    'Renggang': [f.read() for f in renggang[:MAX]],
                    'Sedang':   [f.read() for f in sedang[:MAX]],
                    'Padat':    [f.read() for f in padat[:MAX]],
                }
                # Ambil 1 sample gambar dari kelas Padat untuk visualisasi
                sample_bytes = st.session_state['train_images']['Padat'][0]
                with st.spinner("Memproses gambar sample…"):
                    st.session_state['train_sample_steps'] = preprocess_image_with_steps(image_bytes=sample_bytes)
                st.session_state['train_step'] = 2
                st.rerun()

    # ══ STEP 2 — Visualisasi Preprocessing ════════════════════════════
    if step >= 2:
        st.markdown("---")
        st.markdown(
            "### Langkah 2 — Visualisasi Preprocessing Citra\n"
            "Berikut adalah tahapan preprocessing yang diterapkan pada setiap gambar sebelum "
            "fitur diekstrak. Contoh gambar diambil dari kelas **Padat**."
        )
        steps_vis = st.session_state.get('train_sample_steps')
        if steps_vis:
            render_preprocessing_steps(steps_vis, title="")
        else:
            st.info("Gambar sample tidak tersedia.")

        st.markdown("""
        <div class='step-box'>
        <b>📌 Ringkasan Tahapan Preprocessing:</b><br>
        <b>1. Resize 224×224</b> — Menyamakan ukuran semua gambar agar input ke model konsisten.<br>
        <b>2. Gaussian Blur (kernel 5×5)</b> — Menghaluskan gambar dan mengurangi noise piksel
        agar segmentasi warna lebih akurat dan tidak terganggu variasi kecil warna.<br>
        <b>3. HSV Thresholding</b> — Mengkonversi gambar ke ruang warna HSV lalu mengisolasi
        piksel berwarna hijau (rentang H:25–75°, S:40–255, V:50–255) yang merupakan area gulma.<br>
        <b>4. Morphological Closing</b> — Mengisi lubang/celah kecil pada hasil segmentasi
        menggunakan operasi morfologi sehingga area gulma terdeteksi lebih utuh.
        </div>
        """, unsafe_allow_html=True)

        if step == 2:
            if st.button("Lanjut ke Ekstraksi Fitur →", use_container_width=True, key="btn_step2"):
                st.session_state['train_step'] = 3
                st.rerun()

    # ══ STEP 3 — Ekstraksi Fitur ══════════════════════════════════════
    if step >= 3:
        st.markdown("---")
        st.markdown(
            "### Langkah 3 — Ekstraksi Fitur\n"
            "Fitur numerik diekstrak dari setiap gambar yang sudah di-preprocess. "
            "Pilih jumlah fitur yang akan digunakan:"
        )

        feat_mode = st.radio(
            "Mode Fitur:",
            options=["19 Fitur (RGB + HSV + Hu Moments)", "39 Fitur (GLCM + RGB + HSV + Hu Moments)"],
            index=1 if st.session_state['train_feat_mode'] == '39' else 0,
            key="feat_mode_radio",
        )
        mode_code = '39' if '39' in feat_mode else '19'

        st.markdown(f"""
        <div class='info-box'>
        {'<b>Mode 19 Fitur</b> — Menggunakan fitur warna dan bentuk:<br>'
         '• RGB mean &amp; std: rata-rata dan variasi warna merah, hijau, biru (6 fitur)<br>'
         '• HSV mean &amp; std: rata-rata dan variasi hue, saturation, value (6 fitur)<br>'
         '• Hu Moments: deskriptor bentuk objek yang invariant terhadap rotasi/skala (7 fitur)<br>'
         'Semua 19 fitur digunakan langsung tanpa seleksi.'
         if mode_code == '19' else
         '<b>Mode 39 Fitur (dengan GLCM)</b> — Tambahan fitur tekstur GLCM:<br>'
         '• GLCM (Gray-Level Co-occurrence Matrix): 5 properti tekstur × 4 sudut (0°,45°,90°,135°) = 20 fitur<br>'
         '&nbsp;&nbsp;Properti: contrast, dissimilarity, homogeneity, energy, correlation<br>'
         '• RGB mean &amp; std (6 fitur) + HSV mean &amp; std (6 fitur) + Hu Moments (7 fitur)<br>'
         '<b>Information Gain (LAN)</b> dijalankan otomatis untuk memilih '
         f'<b>{N_SELECT_BEST} fitur terbaik</b> dari 39 fitur.'}
        </div>
        """, unsafe_allow_html=True)

        if step == 3:
            btn_label = "Jalankan Ekstraksi & Information Gain →" if mode_code == '39' else "Jalankan Ekstraksi Fitur →"
            if st.button(btn_label, use_container_width=True, key="btn_step3"):
                imgs = st.session_state.get('train_images')
                if imgs is None:
                    st.error("Data gambar tidak ditemukan. Kembali ke Langkah 1 dan upload ulang gambar.")
                    st.stop()
                with st.spinner("Mengekstrak fitur dari semua gambar…"):
                    X_all, y_all = prepare_features_from_images(imgs, feature_mode=mode_code)

                if len(X_all) == 0:
                    st.error("Tidak ada fitur yang berhasil diekstrak. Periksa gambar yang diupload.")
                    st.stop()

                from collections import Counter
                MIN_PER_CLASS = 9
                class_counts = Counter(y_all)
                min_count = min(class_counts.values())
                if min_count < MIN_PER_CLASS:
                    kelas_sedikit = [f"**{k}** ({v} gambar)" for k, v in sorted(class_counts.items()) if v < MIN_PER_CLASS]
                    st.error(
                        f"Setiap kelas membutuhkan minimal **{MIN_PER_CLASS} gambar** agar dapat dibagi menjadi "
                        f"80% train · 10% validasi · 10% test (stratified split). "
                        f"Kelas yang belum cukup: {', '.join(kelas_sedikit)}. "
                        "Silakan kembali ke Langkah 1 dan upload lebih banyak gambar per kelas."
                    )
                    st.stop()

                X_train, X_val, X_test, y_train, y_val, y_test = _split_80_10_10(X_all, y_all)

                if mode_code == '39':
                    feat_names = FEATURE_NAMES_39
                    selector, X_train_sel, X_val_sel, X_test_sel, sel_names, ig_df = \
                        apply_information_gain(X_train, y_train, X_val, X_test, feat_names)
                else:
                    selector   = None
                    X_train_sel, X_val_sel, X_test_sel = X_train, X_val, X_test
                    sel_names  = FEATURE_NAMES_19
                    ig_df      = None

                scaler      = StandardScaler()
                X_train_sc  = scaler.fit_transform(X_train_sel)
                X_val_sc    = scaler.transform(X_val_sel)
                X_test_sc   = scaler.transform(X_test_sel)

                # Simpan sample hasil ekstraksi (3 baris per kelas) untuk ditampilkan
                feat_cols_raw = FEATURE_NAMES_39 if mode_code == '39' else FEATURE_NAMES_19
                df_feat_all   = pd.DataFrame(X_all, columns=feat_cols_raw)
                df_feat_all['Kelas'] = y_all
                df_feat_sample = (
                    df_feat_all.groupby('Kelas', group_keys=False)
                    .head(3)
                    .reset_index(drop=True)
                )
                st.session_state['train_feat_sample'] = df_feat_sample

                st.session_state['train_feat_mode']  = mode_code
                st.session_state['train_ig_df']      = ig_df
                st.session_state['train_sel_names']  = sel_names
                st.session_state['train_scaler']     = scaler
                st.session_state['train_selector']   = selector
                st.session_state['train_splits'] = {
                    'X_train_sc': X_train_sc, 'X_val_sc': X_val_sc, 'X_test_sc': X_test_sc,
                    'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
                    'split_info': {
                        'total': len(y_all),
                        'train': len(y_train),
                        'val':   len(y_val),
                        'test':  len(y_test),
                    },
                }
                st.session_state['train_step'] = 4
                st.rerun()

        # Tampilkan hasil ekstraksi + IG jika sudah di step 4
        if step >= 4:
            ig_df       = st.session_state.get('train_ig_df')
            splits      = st.session_state.get('train_splits', {})
            si          = splits.get('split_info', {})
            mode_c      = st.session_state['train_feat_mode']
            df_feat_smp = st.session_state.get('train_feat_sample')

            if si:
                st.markdown(
                    f"**Pembagian Data (Stratified 80:10:10):** &nbsp;"
                    f"<span class='split-chip' style='background:#2ecc7122;color:#27ae60;border:1px solid #27ae60;'>Train: {si['train']}</span>"
                    f"<span class='split-chip' style='background:#3498db22;color:#2980b9;border:1px solid #2980b9;'>Val: {si['val']}</span>"
                    f"<span class='split-chip' style='background:#e74c3c22;color:#c0392b;border:1px solid #c0392b;'>Test: {si['test']}</span>"
                    f" — Total: <b>{si['total']}</b> sampel",
                    unsafe_allow_html=True,
                )

            # ── Tabel sample hasil ekstraksi fitur ───────────────────────
            if df_feat_smp is not None:
                n_feat_cols = len(df_feat_smp.columns) - 1  # exclude 'Kelas'
                st.markdown(f"#### 🧬 Contoh Hasil Ekstraksi Fitur ({n_feat_cols} Fitur)")
                st.markdown(
                    "Tabel berikut menampilkan **3 sampel per kelas** setelah preprocessing dan ekstraksi fitur. "
                    "Setiap baris merepresentasikan satu gambar dalam bentuk vektor numerik."
                )
                fmt_dict = {c: '{:.4f}' for c in df_feat_smp.columns if c != 'Kelas'}
                st.dataframe(
                    df_feat_smp.style.format(fmt_dict),
                    use_container_width=True,
                    height=230,
                )
                st.caption(
                    "Nilai setiap kolom adalah fitur numerik hasil ekstraksi dari area gulma yang tersegmentasi. "
                    "Kolom 'Kelas' adalah label kelas asli gambar."
                )

            # ── IG table + bar chart ──────────────────────────────────────
            if mode_c == '39' and ig_df is not None:
                st.markdown(f"#### 📊 Hasil Information Gain — Seleksi {N_SELECT_BEST} Fitur Terbaik dari 39")
                st.markdown(
                    "Information Gain mengukur seberapa besar setiap fitur membantu memisahkan kelas "
                    "Renggang, Sedang, dan Padat. Semakin tinggi skornya, semakin penting fitur tersebut."
                )

                # Bar chart IG
                fig_ig = px.bar(
                    ig_df,
                    x='IG Score', y='Nama Fitur',
                    orientation='h',
                    color='Dipilih',
                    color_discrete_map={'✅ Dipilih': '#2ecc71', '❌ Tidak Dipilih': '#bdc3c7'},
                    title=f'Information Gain Score — {N_SELECT_BEST} Fitur Terpilih (hijau)',
                )
                fig_ig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    height=600,
                    legend_title_text='Status Fitur',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_title='Information Gain Score',
                    yaxis_title='Nama Fitur',
                )
                st.plotly_chart(fig_ig, use_container_width=True)
                st.caption(
                    "Batang hijau = fitur yang dipilih masuk ke model. "
                    "Batang abu-abu = fitur yang tidak dipilih karena IG Score lebih rendah. "
                    "Urutan dari bawah ke atas: fitur paling informatif di atas."
                )

                # Tabel ranking
                def style_ig(row):
                    if row['Dipilih'] == '✅ Dipilih':
                        return ['background-color: #e8f8f5; font-weight: bold'] * len(row)
                    return ['color: #aaa'] * len(row)

                with st.expander("📋 Lihat Tabel Lengkap Ranking Semua Fitur"):
                    st.dataframe(
                        ig_df.style.apply(style_ig, axis=1).format({'IG Score': '{:.4f}'}),
                        use_container_width=True, height=350,
                    )

                sel_names = st.session_state.get('train_sel_names', [])
                st.success(f"✅ **{N_SELECT_BEST} Fitur Terpilih:** {', '.join(f'`{n}`' for n in sel_names)}")
            else:
                st.info(f"Semua **{N_FEATURES_WITHOUT_GLCM} fitur** digunakan langsung (tanpa seleksi).")

    # ══ STEP 4 — Pilih & Latih Model ═════════════════════════════════
    if step >= 4:
        st.markdown("---")
        st.markdown(
            "### Langkah 4 — Pilih Algoritma & Latih Model\n"
            "Pilih satu algoritma klasifikasi lalu klik **Latih**. "
            "Model akan disimpan secara mandiri dan bisa dipakai di menu Pengujian Gambar."
        )

        available = get_available_models()
        if available:
            st.info(f"Model yang sudah dilatih: {', '.join(f'**{m}**' for m in available)}")

        selected_algo = st.selectbox(
            "Pilih Algoritma:",
            list(MODEL_SHORT.keys()),
            key="algo_select",
        )

        # Penjelasan singkat tiap model
        algo_desc = {
            'Decision Tree':       "Pohon keputusan — mudah diinterpretasi, cepat dilatih, cocok untuk data dengan fitur diskriminatif jelas.",
            'Logistic Regression': "Regresi logistik — model linear sederhana, sangat cepat, baseline yang baik.",
            'SVM':                 "Support Vector Machine — mencari hyperplane terbaik, efektif untuk data high-dimensional.",
            'Random Forest':       "Ensemble dari banyak pohon keputusan (500 pohon) — robust terhadap overfitting, akurasi tinggi.",
            'Gradient Boosting':   "Model utama penelitian — ensemble sequential yang secara iteratif memperbaiki error, akurasi tertinggi.",
        }
        st.markdown(f"<div class='info-box'>💡 <b>{selected_algo}:</b> {algo_desc[selected_algo]}</div>", unsafe_allow_html=True)

        path_exists = os.path.exists(get_model_path(selected_algo))
        if path_exists:
            st.warning(f"⚠️ Model **{selected_algo}** sudah pernah dilatih. Klik Latih untuk melatih ulang.")

        splits = st.session_state.get('train_splits')
        if splits is None:
            st.error("Data training belum siap. Kembali ke Langkah 3.")
            return

        if st.button(f"🚀 Latih Model {selected_algo}", use_container_width=True, key="btn_train"):
            with st.spinner(f"Melatih {selected_algo}…"):
                mdl, metrics, cm = train_single_model(
                    selected_algo,
                    splits['X_train_sc'], splits['X_val_sc'], splits['X_test_sc'],
                    splits['y_train'], splits['y_val'], splits['y_test'],
                )
                _feat_mode_used = st.session_state.get('train_feat_mode', '19')
                _sel_names_used = st.session_state.get('train_sel_names') or FEATURE_NAMES_19
                save_model_bundle(
                    model_name=selected_algo,
                    model=mdl,
                    scaler=st.session_state['train_scaler'],
                    selector=st.session_state.get('train_selector'),
                    feature_mode=_feat_mode_used,
                    features_used=_sel_names_used,
                    metrics=metrics,
                    cm=cm,
                    split_info=splits['split_info'],
                )
                st.session_state['train_result'] = {
                    'model_name': selected_algo,
                    'metrics':    metrics,
                    'cm':         cm,
                    'split_info': splits['split_info'],
                }
            st.success(f"✅ Model **{selected_algo}** berhasil dilatih dan disimpan!")

        # Tampilkan hasil training terakhir
        result = st.session_state.get('train_result')
        if result and result['model_name'] == selected_algo:
            m    = result['metrics']
            si   = result['split_info']
            cm_d = result['cm']

            st.markdown(f"""
            <div class='badge-best'>
                🏆 {selected_algo} — Accuracy (Test): {m['Accuracy']:.4f}
            </div>
            """, unsafe_allow_html=True)

            mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
            mc1.metric("Accuracy",     f"{m['Accuracy']:.4f}")
            mc2.metric("Precision",    f"{m['Precision']:.4f}")
            mc3.metric("Recall",       f"{m['Recall']:.4f}")
            mc4.metric("F1-Score",     f"{m['F1-Score']:.4f}")
            mc5.metric("Val Accuracy", f"{m['Val Accuracy']:.4f}")
            mc6.metric("Exec Time (s)",f"{m['Execution Time (s)']:.4f}")

            fig_cm = px.imshow(
                cm_d['matrix'], text_auto=True,
                color_continuous_scale="Greens",
                x=cm_d['labels'], y=cm_d['labels'],
                title=f"Confusion Matrix — {selected_algo}",
            )
            fig_cm.update_layout(
                xaxis_title="Prediksi", yaxis_title="Aktual", width=480, height=400,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_cm)
            st.caption(
                "Confusion Matrix menunjukkan performa model per kelas. "
                "Diagonal (kiri-atas ke kanan-bawah) = prediksi benar. "
                "Angka di luar diagonal = prediksi salah — semakin kecil semakin baik."
            )

        # Tombol latih ulang model lain
        st.markdown("---")
        st.markdown("**Ingin melatih model lain?** Pilih algoritma berbeda di atas dan klik Latih.")
        if st.button("🔄 Mulai Ulang dari Langkah 1", key="btn_reset"):
            for k in ['train_images','train_step','train_feat_mode','train_ig_df',
                      'train_sel_names','train_scaler','train_selector',
                      'train_splits','train_result','train_sample_steps','train_feat_sample']:
                st.session_state[k] = _defaults.get(k)
            st.session_state['train_step'] = 1
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MENU 2 — PENGUJIAN GAMBAR
# ══════════════════════════════════════════════════════════════════════════════
def page_testing():
    st.title("🎯 Pengujian Gambar")
    st.markdown(
        "Upload satu gambar gulma untuk diklasifikasikan. Pilih model yang ingin digunakan "
        "dan lihat hasil prediksi beserta metrik evaluasinya."
    )

    available = get_available_models()
    if not available:
        st.warning(
            "Belum ada model yang tersedia. Silakan latih model terlebih dahulu "
            "di menu **📚 Alur Pelatihan**."
        )
        return

    uploaded = st.file_uploader("Upload Gambar (JPG/JPEG)", type=['jpg','jpeg'], key="test_upload")

    if uploaded is None:
        st.info("Silakan upload gambar untuk memulai pengujian.")
        return

    image_bytes = uploaded.read()
    st.session_state['test_image_bytes'] = image_bytes

    # ── Preprocessing ───────────────────────────────────────────────────
    with st.spinner("Memproses gambar…"):
        steps = preprocess_image_with_steps(image_bytes=image_bytes)

    if steps is None:
        st.error("Gambar tidak dapat dibaca. Coba upload ulang file yang valid.")
        return
    st.session_state['test_steps'] = steps

    render_preprocessing_steps(steps)

    # ── Pilih Model ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 Pilih Model Klasifikasi")

    model_options = available
    selected_model = st.selectbox(
        "Model yang tersedia (sudah dilatih):",
        model_options,
        key="test_model_select",
    )

    # Load bundle & tampilkan info
    try:
        bundle = load_model_bundle(selected_model)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    feat_mode = bundle.get('feature_mode', '19')
    n_feat    = bundle.get('n_features', 19)
    feat_used = bundle.get('features_used', [])
    si        = bundle.get('split_info', {})

    st.markdown(
        f"<div class='info-box'>"
        f"📋 Mode fitur: <b>{'39 fitur (GLCM + LAN → ' + str(n_feat) + ' fitur terpilih)' if feat_mode == '39' else '19 fitur (RGB + HSV + Hu Moments)'}</b><br>"
        f"Fitur yang digunakan: {', '.join(f'<code>{f}</code>' for f in feat_used[:8])}{'...' if len(feat_used) > 8 else ''}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Ekstraksi Fitur & Prediksi ──────────────────────────────────────
    with st.spinner(f"Mengklasifikasikan gambar dengan {selected_model}…"):
        try:
            prediction, features_raw, feat_names = run_inference(bundle, image_bytes)
        except Exception as e:
            st.error(f"Error saat klasifikasi: {e}")
            return

    # ── Tampil Nilai Fitur ──────────────────────────────────────────────
    st.markdown("---")
    n_raw = len(feat_names)
    st.markdown(f"### 🧬 Hasil Ekstraksi {n_raw} Fitur")
    df_feat = pd.DataFrame({
        "No":         range(1, n_raw + 1),
        "Nama Fitur": feat_names,
        "Nilai":      [round(float(v), 6) for v in features_raw],
    })
    half = max(1, n_raw // 2)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.dataframe(df_feat.iloc[:half].set_index("No"), use_container_width=True)
    with col_f2:
        st.dataframe(df_feat.iloc[half:].set_index("No"), use_container_width=True)

    # ── Hasil Prediksi ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏷️ Hasil Klasifikasi")

    density_info = {
        "Renggang": ("#2ecc71", "🌿", "Kepadatan gulma RENDAH — populasi gulma jarang. Tidak perlu penanganan segera."),
        "Sedang":   ("#f39c12", "⚠️", "Kepadatan gulma SEDANG — perlu pemantauan rutin agar tidak meningkat ke kelas Padat."),
        "Padat":    ("#e74c3c", "🚨", "Kepadatan gulma TINGGI — perlu penanganan segera untuk mencegah kerusakan tanaman."),
    }
    color, icon, desc = density_info.get(prediction, ("#888", "❓", "Kelas tidak dikenali"))

    st.markdown(f"""
    <div style='background:{color}20; border:3px solid {color}; border-radius:14px;
                padding:28px; text-align:center; margin:16px 0;'>
        <div style='font-size:52px; margin-bottom:8px;'>{icon}</div>
        <div style='font-size:38px; font-weight:bold; color:{color};'>{prediction}</div>
        <div style='font-size:15px; color:#555; margin-top:10px;'>{desc}</div>
        <div style='font-size:12px; color:#888; margin-top:10px;'>
            Model: <b>{selected_model}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrik Model ────────────────────────────────────────────────────
    metrics = bundle.get('metrics', {})
    if metrics:
        st.markdown("#### 📊 Metrik Evaluasi Model (dari Data Test Training)")
        st.markdown(
            "Metrik berikut dihitung pada data test (10%) yang **tidak pernah dilihat model** "
            "selama proses pelatihan — mencerminkan kemampuan generalisasi model."
        )
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Accuracy",       f"{metrics.get('Accuracy', 0):.4f}")
        mc2.metric("Precision",      f"{metrics.get('Precision', 0):.4f}")
        mc3.metric("Recall",         f"{metrics.get('Recall', 0):.4f}")
        mc4.metric("F1-Score",       f"{metrics.get('F1-Score', 0):.4f}")
        mc5.metric("Val Accuracy",   f"{metrics.get('Val Accuracy', 0):.4f}")

        if si:
            st.caption(
                f"Data test: {si.get('test', '?')} sampel "
                f"(dari total {si.get('total', '?')} sampel, split 80:10:10 stratified)"
            )

    # ── Confusion Matrix ────────────────────────────────────────────────
    cm_data = bundle.get('confusion_matrix', {})
    if cm_data:
        st.markdown("#### 🔲 Confusion Matrix")
        fig_cm = px.imshow(
            cm_data['matrix'], text_auto=True,
            color_continuous_scale="Greens",
            x=cm_data['labels'], y=cm_data['labels'],
            title=f"Confusion Matrix — {selected_model}",
        )
        fig_cm.update_layout(
            xaxis_title="Prediksi", yaxis_title="Aktual", width=480, height=400,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_cm)
        st.caption(
            "Baris = kelas aktual, Kolom = prediksi model. "
            "Angka pada diagonal = prediksi benar. Angka di luar diagonal = kesalahan klasifikasi."
        )

    # ── Bandingkan semua model ──────────────────────────────────────────
    if len(available) > 1:
        with st.expander("📋 Bandingkan Prediksi Semua Model pada Gambar Ini"):
            rows = []
            for m_name in available:
                try:
                    bndl = load_model_bundle(m_name)
                    pred, _, _ = run_inference(bndl, image_bytes)
                    acc = bndl.get('metrics', {}).get('Accuracy', '-')
                    rows.append({
                        'Model': m_name,
                        'Prediksi': pred,
                        'Accuracy (Test)': f"{acc:.4f}" if isinstance(acc, float) else acc,
                    })
                except Exception:
                    pass
            if rows:
                st.table(pd.DataFrame(rows).set_index('Model'))


# ══════════════════════════════════════════════════════════════════════════════
# MENU 3 — EKSPERIMEN PARAMETER
# ══════════════════════════════════════════════════════════════════════════════
def page_experiments():
    st.title("🔬 Eksperimen Parameter")
    st.markdown(
        "Analisis pengaruh parameter terhadap performa klasifikasi menggunakan dataset CSV penuh. "
        "Setiap eksperimen dicatat dan dapat dilihat di **Dashboard Hasil**."
    )

    # ── Cari CSV ────────────────────────────────────────────────────────
    _csv_candidates = [
        os.path.join(os.path.dirname(__file__), "Data_ekstraksi_Fitur_Gulma.csv"),
        os.path.join(os.path.dirname(__file__), "..", "Data_ekstraksi_Fitur_Gulma.csv"),
        "Data_ekstraksi_Fitur_Gulma.csv",
    ]
    CSV_PATH = next((p for p in _csv_candidates if os.path.exists(p)), None)

    if CSV_PATH is None:
        st.error(
            "File `Data_ekstraksi_Fitur_Gulma.csv` tidak ditemukan. "
            "Letakkan file di dalam folder `streamlit_app/`."
        )
        return

    # ── Pilih mode fitur ─────────────────────────────────────────────────
    st.markdown("#### Pilih Mode Fitur")
    feat_mode_exp = st.radio(
        "Mode Fitur untuk Eksperimen:",
        ["19 Fitur (tanpa GLCM)", "39 Fitur (dengan GLCM + Information Gain)"],
        key="exp_feat_mode",
        horizontal=True,
    )
    mode_code_exp = '39' if '39' in feat_mode_exp else '19'

    @st.cache_data(show_spinner="Memuat dataset CSV…")
    def _load_exp_data(csv_path, mode):
        X, y, sel, sel_names, scores_df = load_csv_for_experiment(csv_path, feature_mode=mode)
        return X, y, sel_names

    X_exp, Y_exp, feat_names_exp = _load_exp_data(CSV_PATH, mode_code_exp)
    n_total = len(Y_exp)
    n_train = round(n_total * 0.8)
    n_val   = round(n_total * 0.1)
    n_test  = n_total - n_train - n_val
    n_feat_used = X_exp.shape[1]

    st.info(
        f"Dataset: **{n_total} sampel** · **{n_feat_used} fitur** · "
        f"Split 80:10:10 → Train ≈{n_train} | Val ≈{n_val} | Test ≈{n_test}"
    )

    # ── Pilih Model ──────────────────────────────────────────────────────
    st.markdown("#### Pilih Model & Parameter")
    exp_model = st.selectbox(
        "Model:",
        ["Logistic Regression", "Support Vector Machine (SVM)",
         "Decision Tree", "Random Forest", "Gradient Boosting"],
        key="exp_model_select",
    )

    # Parameter options per model
    if exp_model == "Logistic Regression":
        param_name = "max_iter"
        all_opts   = [100, 300, 500, 700, 1000]
        sel_vals   = st.multiselect(f"Nilai `{param_name}`:", all_opts, default=all_opts, key="exp_p1")
        sel_n_est, sel_lr = [], []

    elif exp_model == "Support Vector Machine (SVM)":
        param_name = "kernel"
        all_opts   = ["linear", "rbf", "poly"]
        sel_vals   = st.multiselect(f"Nilai `{param_name}`:", all_opts, default=all_opts, key="exp_p1")
        sel_n_est, sel_lr = [], []

    elif exp_model == "Decision Tree":
        param_name = "max_depth"
        all_opts   = [3, 5, 7, 9, 11]
        sel_vals   = st.multiselect(f"Nilai `{param_name}`:", all_opts, default=all_opts, key="exp_p1")
        sel_n_est, sel_lr = [], []

    elif exp_model == "Random Forest":
        param_name = "n_estimators"
        all_opts   = [100, 200, 300, 400, 500]
        sel_vals   = st.multiselect(f"Nilai `{param_name}`:", all_opts, default=all_opts, key="exp_p1")
        sel_n_est, sel_lr = [], []

    elif exp_model == "Gradient Boosting":
        param_name = "combinations"
        c1e, c2e = st.columns(2)
        with c1e:
            sel_n_est = st.multiselect("Nilai `n_estimators`:", [100,200,300,400,500], default=[100,200,300], key="exp_nest")
        with c2e:
            sel_lr    = st.multiselect("Nilai `learning_rate`:", [0.01,0.05,0.1,0.5,1.0], default=[0.01,0.1,1.0], key="exp_lr")
        sel_vals = []

    # ── Validasi & Run ───────────────────────────────────────────────────
    can_run = True
    if exp_model == "Gradient Boosting":
        if not sel_n_est or not sel_lr:
            can_run = False
            st.warning("Pilih minimal satu nilai untuk `n_estimators` dan `learning_rate`.")
    else:
        if not sel_vals:
            can_run = False
            st.warning("Pilih minimal satu nilai parameter.")

    if can_run and st.button("🚀 Jalankan Eksperimen", use_container_width=True, key="btn_run_exp"):
        with st.status("Menjalankan eksperimen…", expanded=True) as status:
            X_tr, X_tmp, Y_tr, Y_tmp = train_test_split(X_exp, Y_exp, test_size=0.2, random_state=42, stratify=Y_exp)
            X_vl, X_te, Y_vl, Y_te  = train_test_split(X_tmp, Y_tmp, test_size=0.5, random_state=42, stratify=Y_tmp)
            scaler_e   = StandardScaler()
            X_tr_sc    = scaler_e.fit_transform(X_tr)
            X_vl_sc    = scaler_e.transform(X_vl)
            X_te_sc    = scaler_e.transform(X_te)
            _sw        = _compute_sample_weights(Y_tr)

            def _run_one(mdl, param_str, use_sw=False):
                t0 = time.time()
                mdl.fit(X_tr_sc, Y_tr, sample_weight=_sw) if use_sw else mdl.fit(X_tr_sc, Y_tr)
                exec_t = round(time.time() - t0, 4)
                yp_vl  = mdl.predict(X_vl_sc)
                yp_te  = mdl.predict(X_te_sc)
                return {
                    'Model':             exp_model,
                    'Feature Mode':      feat_mode_exp,
                    'Parameter':         param_str,
                    'Accuracy':          round(accuracy_score(Y_te, yp_te), 4),
                    'Precision':         round(precision_score(Y_te, yp_te, average='macro', zero_division=0), 4),
                    'Recall':            round(recall_score(Y_te, yp_te, average='macro', zero_division=0), 4),
                    'F1-Score':          round(f1_score(Y_te, yp_te, average='macro', zero_division=0), 4),
                    'Val Accuracy':      round(accuracy_score(Y_vl, yp_vl), 4),
                    'Execution Time (s)': exec_t,
                    '_y_test':           Y_te,
                    '_y_pred':           yp_te,
                }

            new_rows = []
            if exp_model == "Gradient Boosting":
                for ne in sel_n_est:
                    for lr in sel_lr:
                        label = f"n_estimators={ne}, learning_rate={lr}"
                        status.write(f"⏳ Melatih {exp_model} — {label}")
                        mdl = GradientBoostingClassifier(n_estimators=ne, learning_rate=lr, random_state=42)
                        row = _run_one(mdl, f"n_est={ne}, lr={lr}", use_sw=True)
                        new_rows.append(row)
                        status.write(f"✅ Selesai — {label} | Accuracy: {row['Accuracy']:.4f}")
            else:
                for pv in sel_vals:
                    status.write(f"⏳ Melatih {exp_model} — {param_name} = {pv}")
                    if exp_model == "Logistic Regression":
                        mdl = LogisticRegression(max_iter=pv, solver='lbfgs', random_state=42, class_weight='balanced')
                    elif exp_model == "Support Vector Machine (SVM)":
                        mdl = SVC(kernel=pv, probability=True, random_state=42, class_weight='balanced')
                    elif exp_model == "Decision Tree":
                        mdl = DecisionTreeClassifier(max_depth=pv, random_state=42, class_weight='balanced')
                    elif exp_model == "Random Forest":
                        mdl = RandomForestClassifier(n_estimators=pv, random_state=42, class_weight='balanced')
                    row = _run_one(mdl, str(pv))
                    new_rows.append(row)
                    status.write(f"✅ Selesai — {param_name} = {pv} | Accuracy: {row['Accuracy']:.4f}")

            st.session_state['exp_history'].extend(new_rows)
            status.update(label="✅ Semua eksperimen selesai!", state="complete")

    # ── Tampil hasil eksperimen saat ini ──────────────────────────────────
    history = st.session_state.get('exp_history', [])
    # Filter untuk model yang sedang dipilih
    current_rows = [r for r in history if r.get('Model') == exp_model]

    if current_rows:
        st.divider()
        st.markdown(f"## 📋 Hasil Eksperimen — {exp_model}")

        df_res = pd.DataFrame(current_rows)
        display_cols = ["Parameter", "Feature Mode", "Accuracy", "Precision",
                        "Recall", "F1-Score", "Val Accuracy", "Execution Time (s)"]

        best_idx = df_res['Accuracy'].idxmax()
        best_row = df_res.loc[best_idx]

        def _hl_best(row):
            return (['background-color:#2ecc71; color:white; font-weight:bold'] * len(row)
                    if row.name == best_idx else [''] * len(row))

        st.dataframe(
            df_res[display_cols].style.apply(_hl_best, axis=1).format({
                "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
                "F1-Score": "{:.4f}", "Val Accuracy": "{:.4f}", "Execution Time (s)": "{:.4f}",
            }),
            use_container_width=True,
        )
        st.success(
            f"🏆 **Parameter Terbaik:** `{best_row['Parameter']}` "
            f"— Accuracy: **{best_row['Accuracy']:.4f}** | Val Accuracy: **{best_row['Val Accuracy']:.4f}**"
        )

        # Grafik
        if exp_model == "Gradient Boosting":
            df_melt = df_res.melt(
                id_vars=["Parameter"], value_vars=["Accuracy","Precision","Recall","F1-Score"],
                var_name="Metrik", value_name="Skor"
            )
            fig = px.bar(df_melt, x="Parameter", y="Skor", color="Metrik", barmode="group",
                         title=f"Gradient Boosting: Metrik per Kombinasi Parameter")
        else:
            # Coba sort numerik dulu (untuk max_iter, max_depth, n_estimators).
            # Jika tidak bisa (misal kernel SVM yang berupa string), pakai sort string.
            df_res_sorted = df_res.copy()
            numeric_params = pd.to_numeric(df_res_sorted["Parameter"], errors='coerce')
            if numeric_params.notna().all():
                df_res_sorted["_sort"] = numeric_params
                df_res_sorted = df_res_sorted.sort_values("_sort").drop(columns=["_sort"])
            else:
                df_res_sorted = df_res_sorted.sort_values("Parameter")
            df_melt = df_res_sorted.melt(
                id_vars=["Parameter"], value_vars=["Accuracy","Precision","Recall","F1-Score"],
                var_name="Metrik", value_name="Skor"
            )
            fig = px.line(df_melt, x="Parameter", y="Skor", color="Metrik", markers=True,
                          title=f"{exp_model}: Metrik vs {param_name}")
            fig.update_xaxes(type='category')
        fig.update_layout(yaxis_range=[0, 1.05], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Grafik menunjukkan pengaruh nilai parameter terhadap metrik evaluasi. "
            "Skor mendekati 1.0 = performa terbaik. Perhatikan nilai parameter yang menghasilkan Accuracy dan F1-Score tertinggi."
        )

        # Confusion Matrix parameter terbaik
        st.markdown("### 🔲 Confusion Matrix — Parameter Terbaik")
        labels_s = sorted(set(best_row['_y_test']))
        fig_cm   = px.imshow(
            confusion_matrix(best_row['_y_test'], best_row['_y_pred'], labels=labels_s),
            text_auto=True, color_continuous_scale="Greens",
            x=labels_s, y=labels_s,
            title=f"Confusion Matrix — {exp_model} | {best_row['Parameter']}",
        )
        fig_cm.update_layout(
            xaxis_title="Prediksi", yaxis_title="Aktual",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption(
            "Confusion Matrix untuk konfigurasi parameter terbaik (Accuracy tertinggi). "
            "Diagonal = prediksi benar per kelas. Sel off-diagonal = kesalahan yang perlu diperhatikan."
        )

        # Classification report
        with st.expander("📄 Classification Report — Parameter Terbaik"):
            rpt     = classification_report(best_row['_y_test'], best_row['_y_pred'], output_dict=True)
            df_rpt  = pd.DataFrame(rpt).transpose().round(4)
            # Kolom support berisi jumlah sampel (integer), tampilkan sebagai int
            if 'support' in df_rpt.columns:
                df_rpt['support'] = df_rpt['support'].fillna(0).astype(int)
            st.dataframe(df_rpt.fillna(''), use_container_width=True)

        if st.button("🗑️ Hapus Riwayat Eksperimen Model Ini", key="btn_clear_exp"):
            st.session_state['exp_history'] = [r for r in history if r.get('Model') != exp_model]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MENU 4 — DASHBOARD HASIL
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.title("📊 Dashboard Hasil")
    st.markdown(
        "Rekap menyeluruh seluruh hasil sistem — dari preprocessing, perbandingan model, "
        "hingga semua hasil eksperimen parameter. Cocok untuk laporan dan presentasi."
    )

    # ── SEKSI 1: Rekap Preprocessing ────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔍 Rekap Alur Preprocessing")
    st.markdown(
        "Setiap gambar yang masuk ke sistem melewati 4 tahap preprocessing sebelum fitur diekstrak:"
    )
    col_pre = st.columns(4)
    steps_info = [
        ("① Resize 224×224",       "Gambar diubah ukuran menjadi 224×224 piksel sebagai standar input."),
        ("② Gaussian Blur (5×5)", "Menghaluskan gambar dan meredam noise agar segmentasi akurat."),
        ("③ HSV Thresholding",     "Mengisolasi piksel berwarna hijau (H:25–75°) sebagai area gulma."),
        ("④ Morphological Closing","Mengisi celah kecil pada mask agar segmentasi lebih utuh."),
    ]
    for col, (label, desc) in zip(col_pre, steps_info):
        with col:
            st.markdown(
                f"<div class='step-box' style='text-align:center;'>"
                f"<b>{label}</b><br><small style='color:#555;'>{desc}</small></div>",
                unsafe_allow_html=True
            )

    sample_steps = st.session_state.get('train_sample_steps')
    if sample_steps:
        render_preprocessing_steps(sample_steps, title="Contoh Hasil Preprocessing (dari Training Terakhir)")
    else:
        st.info("Jalankan training di menu **Alur Pelatihan** untuk melihat visualisasi preprocessing di sini.")

    # ── SEKSI 2: Perbandingan Model ──────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🏆 Perbandingan Semua Model Terlatih")

    available = get_available_models()
    if available:
        rows = []
        for m_name in available:
            try:
                b = load_model_bundle(m_name)
                m = b.get('metrics', {})
                rows.append({
                    'Model':          m_name,
                    'Mode Fitur':     b.get('feature_mode','?') + ' fitur',
                    'Jumlah Fitur':   b.get('n_features', '?'),
                    'Accuracy':       m.get('Accuracy', '-'),
                    'Precision':      m.get('Precision', '-'),
                    'Recall':         m.get('Recall', '-'),
                    'F1-Score':       m.get('F1-Score', '-'),
                    'Val Accuracy':   m.get('Val Accuracy', '-'),
                    'Exec Time (s)':  m.get('Execution Time (s)', '-'),
                })
            except Exception:
                pass

        if rows:
            df_comp = pd.DataFrame(rows).set_index('Model')
            best_m  = df_comp['Accuracy'].idxmax()

            def _hl(row):
                return (['background-color:#2ecc71; color:white; font-weight:bold'] * len(row)
                        if row.name == best_m else [''] * len(row))

            st.dataframe(
                df_comp.style.apply(_hl, axis=1).format({
                    c: "{:.4f}" for c in ['Accuracy','Precision','Recall','F1-Score','Val Accuracy','Exec Time (s)']
                    if c in df_comp.columns
                }),
                use_container_width=True,
            )
            st.markdown(
                f"<div class='badge-best'>🏆 Model Terbaik: {best_m} — "
                f"Accuracy: {df_comp.loc[best_m,'Accuracy']:.4f}</div>",
                unsafe_allow_html=True
            )

            # Bar chart perbandingan
            df_bar = pd.DataFrame(rows).melt(
                id_vars=['Model'], value_vars=['Accuracy','Precision','Recall','F1-Score'],
                var_name='Metrik', value_name='Skor',
            )
            fig_bar = px.bar(
                df_bar, x='Model', y='Skor', color='Metrik', barmode='group',
                title='Perbandingan Metrik Semua Model (Data Test 10%)',
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig_bar.update_layout(yaxis_range=[0, 1.05], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption(
                "Perbandingan Accuracy, Precision, Recall, dan F1-Score semua model pada data test (10%). "
                "Model dengan batang tertinggi di semua metrik adalah kandidat terbaik untuk digunakan."
            )

            # Confusion matrix per model
            with st.expander("🔲 Lihat Confusion Matrix Semua Model"):
                cm_cols = st.columns(min(len(rows), 3))
                for i, row_d in enumerate(rows):
                    try:
                        b   = load_model_bundle(row_d['Model'])
                        cmd = b.get('confusion_matrix', {})
                        if cmd:
                            fig_c = px.imshow(
                                cmd['matrix'], text_auto=True,
                                color_continuous_scale="Greens",
                                x=cmd['labels'], y=cmd['labels'],
                                title=row_d['Model'],
                            )
                            fig_c.update_layout(
                                width=340, height=320,
                                xaxis_title="Prediksi", yaxis_title="Aktual",
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            )
                            cm_cols[i % len(cm_cols)].plotly_chart(fig_c)
                            cm_cols[i % len(cm_cols)].caption("Diagonal = prediksi benar")
                    except Exception:
                        pass
    else:
        st.info("Belum ada model yang dilatih. Kunjungi menu **📚 Alur Pelatihan**.")

    # ── SEKSI 3: Rekap Eksperimen ────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔬 Rekap Semua Hasil Eksperimen Parameter")

    history = st.session_state.get('exp_history', [])
    if history:
        df_hist = pd.DataFrame(history)
        display_h = ["Model","Feature Mode","Parameter","Accuracy","Precision",
                     "Recall","F1-Score","Val Accuracy","Execution Time (s)"]
        st.dataframe(
            df_hist[display_h].style.format({
                c: "{:.4f}" for c in ["Accuracy","Precision","Recall","F1-Score",
                                       "Val Accuracy","Execution Time (s)"]
            }),
            use_container_width=True,
        )

        # Grafik per model
        for m_name in df_hist['Model'].unique():
            df_m = df_hist[df_hist['Model'] == m_name]
            fig_exp = px.line(
                df_m.melt(id_vars=['Parameter'], value_vars=['Accuracy','F1-Score'],
                           var_name='Metrik', value_name='Skor'),
                x='Parameter', y='Skor', color='Metrik', markers=True,
                title=f"Eksperimen — {m_name}",
            )
            fig_exp.update_xaxes(type='category')
            fig_exp.update_layout(yaxis_range=[0, 1.05], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_exp, use_container_width=True)
            st.caption(
                f"Tren Accuracy dan F1-Score {m_name} untuk setiap variasi parameter yang diuji. "
                "Garis mendatar atau menurun mengindikasikan parameter tersebut kurang berpengaruh pada model ini."
            )
    else:
        st.info("Belum ada hasil eksperimen. Kunjungi menu **🔬 Eksperimen Parameter**.")

    # ── SEKSI 4: Penjelasan Sistem ───────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📖 Penjelasan Sistem (untuk Pembaca Umum)")
    with st.expander("Apa itu Klasifikasi Kepadatan Gulma?", expanded=False):
        st.markdown("""
        **Gulma** adalah tanaman liar yang tumbuh di sekitar tanaman utama dan dapat mengganggu
        pertumbuhan serta mengurangi hasil panen. Sistem ini mengklasifikasikan lahan menjadi 3 tingkat:
        - 🟢 **Renggang** — Gulma sedikit, tidak perlu tindakan segera.
        - 🟡 **Sedang** — Gulma mulai banyak, perlu dipantau dan direncanakan penanganannya.
        - 🔴 **Padat** — Gulma sangat banyak, perlu penanganan segera agar tidak merusak tanaman.
        """)
    with st.expander("Apa itu HSV Thresholding?", expanded=False):
        st.markdown("""
        **HSV** (Hue, Saturation, Value) adalah cara merepresentasikan warna yang lebih mendekati
        persepsi manusia dibandingkan RGB. Dalam sistem ini:
        - **Hue 25–75°** merepresentasikan rentang warna hijau (warna gulma).
        - Piksel yang masuk rentang ini dianggap sebagai area gulma dan dipertahankan.
        - Piksel di luar rentang (tanah, langit, dll.) dibuang menjadi hitam.
        Teknik ini disebut **Segmentasi Warna** — memisahkan objek berdasarkan warnanya.
        """)
    with st.expander("Apa itu GLCM dan Information Gain?", expanded=False):
        st.markdown("""
        **GLCM** (Gray-Level Co-occurrence Matrix) mengukur **tekstur** gambar — seberapa sering
        pasangan piksel dengan nilai tertentu muncul berdekatan. Dari GLCM diekstrak 5 properti:
        Contrast (kekasaran), Dissimilarity (perbedaan), Homogeneity (keseragaman),
        Energy (keseragaman kuadrat), dan Correlation (keterkaitan antar piksel),
        masing-masing dari 4 sudut (0°, 45°, 90°, 135°) = **20 fitur GLCM**.

        **Information Gain (LAN)** adalah teknik seleksi fitur yang mengukur seberapa besar
        setiap fitur membantu membedakan kelas. Dari 39 fitur, dipilih **14 fitur terbaik**
        yang paling informatif untuk klasifikasi gulma.
        """)
    with st.expander("Apa itu Gradient Boosting?", expanded=False):
        st.markdown("""
        **Gradient Boosting** adalah metode ensemble yang membangun model secara bertahap (sequential).
        Setiap model baru dilatih untuk memperbaiki kesalahan model sebelumnya. Proses ini diulang
        sebanyak **300 kali** (n_estimators=300) dengan kecepatan belajar (learning_rate=0.1).
        Hasilnya adalah model yang sangat akurat karena setiap iterasi fokus pada kasus yang sulit.
        Model ini dipilih sebagai **model utama penelitian** karena menghasilkan akurasi tertinggi.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
_page = st.session_state['page']

if   _page == 'training':
    page_training()
elif _page == 'testing':
    page_testing()
elif _page == 'experiments':
    page_experiments()
elif _page == 'dashboard':
    page_dashboard()
else:
    nav('training')
