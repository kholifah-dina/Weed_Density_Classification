from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler

from predict import (
    FEATURE_NAMES_19,
    FEATURE_NAMES_39,
    N_SELECT_BEST,
    _split_80_10_10,
    apply_information_gain,
    cleanup_non_top5,
    load_trained_bundle,
    prepare_features_from_images,
    run_inference,
    save_trained_bundle,
    train_with_custom_params,
)
from preprocessing import preprocess_image_with_steps

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Kepadatan Gulma",
    layout="wide",
    page_icon="🌿",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

:root {
    --green:       #2ecc71;
    --dark-green:  #27ae60;
    --light-green: rgba(46,204,113,0.10);
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
    background: rgba(46,204,113,0.08);
    border: 1px solid rgba(46,204,113,0.35);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 14px;
}
.model-card {
    background: rgba(46,204,113,0.08);
    border: 1px solid rgba(46,204,113,0.35);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 14px;
    line-height: 1.8;
}
.top5-badge {
    display: inline-block;
    background: rgba(46,204,113,0.20);
    color: #1e8449;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
    margin: 2px 0;
}

/* ── Dark mode ───────────────────────────────────────────────────────── */
@media (prefers-color-scheme: dark) {
    :root { --light-green: rgba(46,204,113,0.12); }
    .step-desc   { color: #b0b8c1 !important; }
    .step-label  { color: #a8e6cf !important; }
    .step-box    { background: var(--light-green) !important; color: #d0d8e0 !important; }
    .badge-best  { background: var(--light-green) !important; color: #a8e6cf !important; }
    .info-box    { background: rgba(46,204,113,0.08) !important; color: #c8d6e0 !important; }
    .model-card  { background: rgba(46,204,113,0.08) !important; color: #d0d8e0 !important; }
    .split-chip  { color: inherit !important; }
    .top5-badge  { color: #a8e6cf !important; }
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
_defaults = {
    # Pemodelan — training steps
    'train_step':         1,
    'train_images':       None,
    'train_sample_steps': None,
    'train_feat_mode':    '39',
    'train_ig_df':        None,
    'train_sel_names':    None,
    'train_scaler':       None,
    'train_selector':     None,
    'train_splits':       None,
    'train_feat_sample':  None,
    # Training history (persists when step is reset)
    'algo_counters':      {'LR': 0, 'SVM': 0, 'DT': 0, 'RF': 0, 'GB': 0},
    'model_run_history':  [],
    'top5_model_ids':     [],
    # Implementasi
    'impl_test_runs':     [],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Kepadatan Gulma")
    st.markdown("---")
    page_choice = st.radio(
        "Pilih Halaman:",
        ["🔬 Pemodelan", "🎯 Implementasi"],
        key="sidebar_nav",
    )
    st.markdown("---")

    top5_ids_sidebar = st.session_state.get('top5_model_ids', [])
    if top5_ids_sidebar:
        st.markdown("**💾 Top 5 Model Tersimpan:**")
        for mid in top5_ids_sidebar:
            st.markdown(f"<span class='top5-badge'>🏆 {mid}</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='font-size:12px;color:#aaa;'>Belum ada model Top 5</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#aaa;text-align:center;'>"
        "Universitas Telkom Purwokerto<br>Tugas Akhir · HSV + Gradient Boosting"
        "</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
ALGO_SHORT_MAP = {
    'Logistic Regression': 'LR',
    'SVM':                 'SVM',
    'Decision Tree':       'DT',
    'Random Forest':       'RF',
    'Gradient Boosting':   'GB',
}

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def render_preprocessing_steps(steps, title="🔍 Tahap Preprocessing Citra"):
    if title:
        st.markdown(f"### {title}")
    c1, c2, c3, c4 = st.columns(4)
    for col, img, label, desc in [
        (c1, steps['original'],  "① Resize 224×224",
         "Gambar diubah ukuran menjadi 224×224 piksel sebagai input standar sistem."),
        (c2, steps['blurred'],   "② Gaussian Blur (5×5)",
         "Gaussian Blur mengurangi noise agar segmentasi warna lebih akurat."),
        (c3, steps['hsv_mask'],  "③ HSV Thresholding",
         "Piksel hijau (H:25–75°, S:40–255, V:50–255) diisolasi sebagai area gulma."),
        (c4, steps['segmented'], "④ Morphological Closing",
         "Morphological Closing (5×5) mengisi celah kecil pada mask gulma."),
    ]:
        with col:
            st.image(img, use_container_width=True)
            st.markdown(f"<p class='step-label'>{label}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='step-desc'>{desc}</p>",   unsafe_allow_html=True)


def _update_top5():
    """Sort model_run_history by Accuracy, keep top 5, update disk files."""
    history = st.session_state['model_run_history']
    if not history:
        st.session_state['top5_model_ids'] = []
        cleanup_non_top5([])
        return
    sorted_h = sorted(history, key=lambda r: r['metrics']['Accuracy'], reverse=True)
    top5_ids = [r['model_id'] for r in sorted_h[:5]]
    st.session_state['top5_model_ids'] = top5_ids
    cleanup_non_top5(top5_ids)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1 — PEMODELAN
# ──────────────────────────────────────────────────────────────────────────────
def page_pemodelan():
    st.title("🔬 Pemodelan — Model Engineering")
    st.markdown(
        "Ikuti langkah berikut secara berurutan: upload dataset, lihat visualisasi preprocessing, "
        "ekstraksi fitur, lalu latih model dengan berbagai algoritma dan parameter."
    )

    step = st.session_state['train_step']

    # ── Progress indicator ────────────────────────────────────────────────
    steps_labels = ["① Upload Dataset", "② Preprocessing", "③ Ekstraksi Fitur", "④ Pelatihan Dinamis"]
    cols_prog = st.columns(4)
    for i, (col, lbl) in enumerate(zip(cols_prog, steps_labels), start=1):
        with col:
            color      = "#2ecc71" if i <= step else "#ddd"
            text_color = "white"   if i <= step else "#999"
            st.markdown(
                f"<div style='background:{color};color:{text_color};border-radius:8px;"
                f"padding:8px;text-align:center;font-size:13px;font-weight:600;'>{lbl}</div>",
                unsafe_allow_html=True,
            )
    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════
    # STEP 1 — UPLOAD DATASET
    # ════════════════════════════════════════════════════════════════════
    if step >= 1:
        st.markdown("### Langkah 1 — Upload Dataset Gambar")
        st.info("📎 Format: **JPG / JPEG** · Maks. **190 gambar/kelas** · Min. **9 gambar/kelas**")

        MAX = 190
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**🟢 Renggang** — Populasi gulma jarang")
            renggang = st.file_uploader(
                "Upload Kelas Renggang", type=['jpg', 'jpeg'],
                accept_multiple_files=True, key="up_renggang",
            )
        with c2:
            st.markdown("**🟡 Sedang** — Perlu pemantauan")
            sedang = st.file_uploader(
                "Upload Kelas Sedang", type=['jpg', 'jpeg'],
                accept_multiple_files=True, key="up_sedang",
            )
        with c3:
            st.markdown("**🔴 Padat** — Perlu penanganan segera")
            padat = st.file_uploader(
                "Upload Kelas Padat", type=['jpg', 'jpeg'],
                accept_multiple_files=True, key="up_padat",
            )

        for files, label in [(renggang, "Renggang"), (sedang, "Sedang"), (padat, "Padat")]:
            if files and len(files) > MAX:
                st.error(f"Maksimum {MAX} gambar untuk kelas {label}.")

        n_r   = len(renggang) if renggang else 0
        n_s   = len(sedang)   if sedang   else 0
        n_p   = len(padat)    if padat    else 0
        total = n_r + n_s + n_p

        if total > 0:
            st.table(pd.DataFrame({
                "Kelas":         ["🟢 Renggang", "🟡 Sedang", "🔴 Padat", "Total"],
                "Jumlah Gambar": [n_r, n_s, n_p, total],
            }).set_index("Kelas"))

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
                sample_bytes = st.session_state['train_images']['Padat'][0]
                with st.spinner("Memproses gambar sample…"):
                    _sample_steps = preprocess_image_with_steps(image_bytes=sample_bytes)
                if _sample_steps is None:
                    st.error("Gagal memproses gambar sample. Pastikan file tidak rusak.")
                    st.stop()
                st.session_state['train_sample_steps'] = _sample_steps
                st.session_state['train_step'] = 2
                st.rerun()

    # ════════════════════════════════════════════════════════════════════
    # STEP 2 — VISUALISASI PREPROCESSING
    # ════════════════════════════════════════════════════════════════════
    if step >= 2:
        st.markdown("---")
        st.markdown(
            "### Langkah 2 — Visualisasi Preprocessing Citra\n"
            "Tahapan preprocessing diterapkan pada setiap gambar sebelum fitur diekstrak. "
            "Contoh gambar diambil dari kelas **Padat**."
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
        <b>2. Gaussian Blur (kernel 5×5)</b> — Menghaluskan gambar dan mengurangi noise piksel.<br>
        <b>3. HSV Thresholding</b> — Mengisolasi piksel berwarna hijau (H:25–75°, S:40–255, V:50–255).<br>
        <b>4. Morphological Closing</b> — Mengisi celah kecil pada mask segmentasi.
        </div>
        """, unsafe_allow_html=True)

        if step == 2:
            if st.button("Lanjut ke Ekstraksi Fitur →", use_container_width=True, key="btn_step2"):
                st.session_state['train_step'] = 3
                st.rerun()

    # ════════════════════════════════════════════════════════════════════
    # STEP 3 — EKSTRAKSI FITUR
    # ════════════════════════════════════════════════════════════════════
    if step >= 3:
        st.markdown("---")
        st.markdown(
            "### Langkah 3 — Ekstraksi Fitur\n"
            "Pilih mode fitur yang akan diekstrak dari setiap gambar yang sudah di-preprocess:"
        )
        feat_mode = st.radio(
            "Mode Fitur:",
            options=["19 Fitur (RGB + HSV + Hu Moments)", "39 Fitur (GLCM + RGB + HSV + Hu Moments)"],
            index=1 if st.session_state['train_feat_mode'] == '39' else 0,
            key="feat_mode_radio",
        )
        mode_code = '39' if '39' in feat_mode else '19'

        if mode_code == '19':
            info_text = (
                "<b>Mode 19 Fitur</b> — RGB mean &amp; std (6) + HSV mean &amp; std (6) + "
                "Hu Moments (7). Semua 19 fitur digunakan langsung tanpa seleksi."
            )
        else:
            info_text = (
                f"<b>Mode 39 Fitur</b> — GLCM 20 fitur tekstur + RGB (6) + HSV (6) + Hu Moments (7) = 39 fitur.<br>"
                f"<b>Information Gain (LAN)</b> memilih <b>{N_SELECT_BEST} fitur terbaik</b> secara otomatis."
            )
        st.markdown(f"<div class='info-box'>{info_text}</div>", unsafe_allow_html=True)

        if step == 3:
            btn_label = (
                "Jalankan Ekstraksi & Information Gain →"
                if mode_code == '39' else "Jalankan Ekstraksi Fitur →"
            )
            if st.button(btn_label, use_container_width=True, key="btn_step3"):
                imgs = st.session_state.get('train_images')
                if imgs is None:
                    st.error("Data gambar tidak ditemukan. Kembali ke Langkah 1.")
                    st.stop()

                with st.spinner("Mengekstrak fitur dari semua gambar…"):
                    X_all, y_all = prepare_features_from_images(imgs, feature_mode=mode_code)

                if len(X_all) == 0:
                    st.error("Tidak ada fitur yang berhasil diekstrak. Periksa gambar yang diupload.")
                    st.stop()

                MIN_PER_CLASS = 9
                class_counts  = Counter(y_all)
                min_count     = min(class_counts.values())
                if min_count < MIN_PER_CLASS:
                    kelas_sedikit = [
                        f"**{k}** ({v} gambar)"
                        for k, v in sorted(class_counts.items()) if v < MIN_PER_CLASS
                    ]
                    st.error(
                        f"Setiap kelas butuh minimal **{MIN_PER_CLASS} gambar** (split 80:10:10 stratified). "
                        f"Kelas kurang: {', '.join(kelas_sedikit)}."
                    )
                    st.stop()

                X_train, X_val, X_test, y_train, y_val, y_test = _split_80_10_10(X_all, y_all)

                if mode_code == '39':
                    feat_names = FEATURE_NAMES_39
                    selector, X_train_sel, X_val_sel, X_test_sel, sel_names, ig_df = \
                        apply_information_gain(X_train, y_train, X_val, X_test, feat_names)
                else:
                    selector     = None
                    X_train_sel, X_val_sel, X_test_sel = X_train, X_val, X_test
                    sel_names    = FEATURE_NAMES_19
                    ig_df        = None

                scaler     = StandardScaler()
                X_train_sc = scaler.fit_transform(X_train_sel)
                X_val_sc   = scaler.transform(X_val_sel)
                X_test_sc  = scaler.transform(X_test_sel)

                feat_cols_raw  = FEATURE_NAMES_39 if mode_code == '39' else FEATURE_NAMES_19
                df_feat_all    = pd.DataFrame(X_all, columns=feat_cols_raw)
                df_feat_all['Kelas'] = y_all
                df_feat_sample = (
                    df_feat_all.groupby('Kelas', group_keys=False)
                    .head(3).reset_index(drop=True)
                )

                st.session_state.update({
                    'train_feat_mode':   mode_code,
                    'train_ig_df':       ig_df,
                    'train_sel_names':   sel_names,
                    'train_scaler':      scaler,
                    'train_selector':    selector,
                    'train_feat_sample': df_feat_sample,
                    'train_splits': {
                        'X_train_sc': X_train_sc, 'X_val_sc': X_val_sc, 'X_test_sc': X_test_sc,
                        'y_train':    y_train,     'y_val':    y_val,    'y_test':    y_test,
                        'split_info': {
                            'total': len(y_all), 'train': len(y_train),
                            'val':   len(y_val), 'test':  len(y_test),
                        },
                    },
                    'train_step': 4,
                })
                st.rerun()

        # ── Show extraction results once step >= 4 ────────────────────
        if step >= 4:
            ig_df   = st.session_state.get('train_ig_df')
            splits  = st.session_state.get('train_splits', {})
            si      = splits.get('split_info', {})
            mode_c  = st.session_state['train_feat_mode']
            df_smp  = st.session_state.get('train_feat_sample')

            if si:
                st.markdown(
                    f"**Pembagian Data (Stratified 80:10:10):** &nbsp;"
                    f"<span class='split-chip' style='background:#2ecc7122;color:#27ae60;border:1px solid #27ae60;'>Train: {si['train']}</span>"
                    f"<span class='split-chip' style='background:#3498db22;color:#2980b9;border:1px solid #2980b9;'>Val: {si['val']}</span>"
                    f"<span class='split-chip' style='background:#e74c3c22;color:#c0392b;border:1px solid #c0392b;'>Test: {si['test']}</span>"
                    f" — Total: <b>{si['total']}</b> sampel",
                    unsafe_allow_html=True,
                )

            if df_smp is not None:
                n_feat_cols = len(df_smp.columns) - 1
                st.markdown(f"#### 🧬 Contoh Hasil Ekstraksi Fitur ({n_feat_cols} Fitur)")
                fmt_dict = {c: '{:.4f}' for c in df_smp.columns if c != 'Kelas'}
                st.dataframe(df_smp.style.format(fmt_dict), use_container_width=True, height=230)
                st.caption("3 sampel per kelas — setiap baris = satu gambar direpresentasikan sebagai vektor numerik.")

            if mode_c == '39' and ig_df is not None:
                sel_names_disp = st.session_state.get('train_sel_names', [])

                st.markdown(f"#### 📊 Grafik Information Gain — {N_SELECT_BEST} Fitur Terbaik dari 39")
                st.markdown("Batang **hijau** = dipilih · Batang **abu-abu** = tidak dipilih.")
                fig_ig = px.bar(
                    ig_df, x='IG Score', y='Nama Fitur', orientation='h',
                    color='Dipilih',
                    color_discrete_map={'✅ Dipilih': '#2ecc71', '❌ Tidak Dipilih': '#bdc3c7'},
                    title=f'Information Gain Score — {N_SELECT_BEST} Fitur Terpilih',
                )
                fig_ig.update_layout(
                    yaxis={'categoryorder': 'total ascending'}, height=620,
                    legend_title_text='Status Fitur',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_title='IG Score', yaxis_title='Nama Fitur',
                )
                st.plotly_chart(fig_ig, use_container_width=True)
                st.caption("Skor mendekati 1 = fitur paling informatif untuk membedakan kelas gulma.")

                st.markdown("#### 📋 Tabel Ranking 39 Fitur berdasarkan Information Gain")
                def style_ig(row):
                    if row['Dipilih'] == '✅ Dipilih':
                        return ['background-color:rgba(46,204,113,0.18);color:#1e8449;font-weight:bold'] * len(row)
                    return ['color:#888888'] * len(row)
                st.dataframe(
                    ig_df.style.apply(style_ig, axis=1).format({'IG Score': '{:.4f}'}),
                    use_container_width=True, height=400,
                )
                st.caption("Baris hijau tebal = fitur yang masuk ke model. Diurutkan dari IG Score tertinggi.")

                st.markdown(f"#### ✅ {N_SELECT_BEST} Fitur Terpilih untuk Pelatihan Model")
                df_sel = ig_df[ig_df['Dipilih'] == '✅ Dipilih'].reset_index(drop=True)
                df_sel.index += 1
                df_sel_disp = df_sel[['Nama Fitur', 'IG Score']].copy()
                df_sel_disp.index.name = 'No'
                def _style_sel(row):
                    max_s = df_sel_disp['IG Score'].max()
                    alpha = 0.1 + 0.35 * (row['IG Score'] / max_s if max_s > 0 else 0)
                    return [f'background-color:rgba(46,204,113,{alpha:.2f});color:#1e8449;font-weight:bold'] * len(row)
                st.dataframe(
                    df_sel_disp.style.apply(_style_sel, axis=1).format({'IG Score': '{:.4f}'}),
                    use_container_width=True, height=int(N_SELECT_BEST * 38 + 40),
                )
                st.success(f"✅ **{N_SELECT_BEST} Fitur:** {', '.join(f'`{n}`' for n in sel_names_disp)}")

            else:
                st.markdown("#### 📋 Daftar 19 Fitur yang Digunakan")
                st.markdown("Semua 19 fitur berikut digunakan langsung sebagai input model tanpa seleksi:")
                _feat_19_meta = [
                    ("R_mean",     "RGB",        "Rata-rata intensitas kanal Merah"),
                    ("G_mean",     "RGB",        "Rata-rata intensitas kanal Hijau"),
                    ("B_mean",     "RGB",        "Rata-rata intensitas kanal Biru"),
                    ("R_std",      "RGB",        "Standar deviasi kanal Merah"),
                    ("G_std",      "RGB",        "Standar deviasi kanal Hijau"),
                    ("B_std",      "RGB",        "Standar deviasi kanal Biru"),
                    ("H_mean",     "HSV",        "Rata-rata Hue (warna dominan)"),
                    ("S_mean",     "HSV",        "Rata-rata Saturation (kepekatan warna)"),
                    ("V_mean",     "HSV",        "Rata-rata Value (kecerahan)"),
                    ("H_std",      "HSV",        "Standar deviasi Hue"),
                    ("S_std",      "HSV",        "Standar deviasi Saturation"),
                    ("V_std",      "HSV",        "Standar deviasi Value"),
                    ("HuMoment_1", "Hu Moments", "Momen Hu ke-1 — invariant skala & rotasi"),
                    ("HuMoment_2", "Hu Moments", "Momen Hu ke-2"),
                    ("HuMoment_3", "Hu Moments", "Momen Hu ke-3"),
                    ("HuMoment_4", "Hu Moments", "Momen Hu ke-4"),
                    ("HuMoment_5", "Hu Moments", "Momen Hu ke-5"),
                    ("HuMoment_6", "Hu Moments", "Momen Hu ke-6"),
                    ("HuMoment_7", "Hu Moments", "Momen Hu ke-7 — invariant refleksi"),
                ]
                df_19 = pd.DataFrame(_feat_19_meta, columns=["Nama Fitur", "Kelompok", "Deskripsi"])
                df_19.index += 1
                df_19.index.name = "No"
                def _color_group(val):
                    palette = {
                        "RGB":        "background-color:rgba(46,204,113,0.20);color:#1e8449;font-weight:bold",
                        "HSV":        "background-color:rgba(52,152,219,0.20);color:#1a6fa6;font-weight:bold",
                        "Hu Moments": "background-color:rgba(230,126,34,0.20);color:#b7580a;font-weight:bold",
                    }
                    return palette.get(val, '')
                st.dataframe(
                    df_19.style.map(_color_group, subset=["Kelompok"]),
                    use_container_width=True, height=int(19 * 38 + 40),
                )
                st.caption("Hijau = RGB · Biru = HSV · Oranye = Hu Moments.")
                st.success("✅ **19 Fitur** digunakan langsung tanpa seleksi Information Gain.")

    # ════════════════════════════════════════════════════════════════════
    # STEP 4 — PELATIHAN DINAMIS
    # ════════════════════════════════════════════════════════════════════
    if step >= 4:
        st.markdown("---")
        st.markdown("### Langkah 4 — Pelatihan Dinamis")
        st.markdown(
            "Pilih algoritma dan parameter, lalu klik **Latih**. Setiap percobaan diberi ID unik "
            "dan dicatat di tabel rekap. **5 model dengan akurasi tertinggi** disimpan otomatis "
            "untuk digunakan di halaman Implementasi."
        )

        splits = st.session_state.get('train_splits')
        if splits is None:
            st.error("Data training belum siap. Kembali ke Langkah 3.")
            return

        # ── Algorithm selector ────────────────────────────────────────
        algo_full = st.selectbox(
            "Pilih Algoritma:",
            list(ALGO_SHORT_MAP.keys()),
            key="step4_algo",
        )
        algo_short = ALGO_SHORT_MAP[algo_full]

        algo_desc = {
            'Logistic Regression': "Model linear sederhana — sangat cepat, baseline yang kuat.",
            'SVM':                 "Support Vector Machine — efektif untuk data high-dimensional.",
            'Decision Tree':       "Pohon keputusan — mudah diinterpretasi, cepat dilatih.",
            'Random Forest':       "Ensemble 100–500 pohon — robust terhadap overfitting.",
            'Gradient Boosting':   "Model utama penelitian — ensemble sequential, akurasi tinggi.",
        }
        st.markdown(
            f"<div class='info-box'>💡 <b>{algo_full}:</b> {algo_desc[algo_full]}</div>",
            unsafe_allow_html=True,
        )

        # ── Parameter selector (fixed values per skripsi) ─────────────
        params = {}
        if algo_full == 'Logistic Regression':
            params['max_iter'] = st.selectbox(
                "max_iter:", [100, 300, 500, 700, 1000], key="p_lr_maxiter",
            )
            param_str = f"max_iter={params['max_iter']}"

        elif algo_full == 'SVM':
            params['kernel'] = st.selectbox(
                "kernel:", ["linear", "rbf", "poly"], key="p_svm_kernel",
            )
            param_str = f"kernel={params['kernel']}"

        elif algo_full == 'Decision Tree':
            params['max_depth'] = st.selectbox(
                "max_depth:", [3, 5, 7, 9, 11], key="p_dt_depth",
            )
            param_str = f"max_depth={params['max_depth']}"

        elif algo_full == 'Random Forest':
            params['n_estimators'] = st.selectbox(
                "n_estimators:", [100, 200, 300, 400, 500], key="p_rf_nest",
            )
            param_str = f"n_estimators={params['n_estimators']}"

        elif algo_full == 'Gradient Boosting':
            c1p, c2p = st.columns(2)
            with c1p:
                params['n_estimators'] = st.selectbox(
                    "n_estimators:", [100, 200, 300], key="p_gb_nest",
                )
            with c2p:
                params['learning_rate'] = st.selectbox(
                    "learning_rate:", [0.01, 0.1, 1], key="p_gb_lr",
                )
            param_str = f"n_estimators={params['n_estimators']}, lr={params['learning_rate']}"

        # Next model ID preview
        cur_count = st.session_state['algo_counters'].get(algo_short, 0)
        next_id   = f"{algo_short}{cur_count + 1}"
        st.markdown(
            f"<div style='font-size:13px;color:#888;margin-bottom:8px;'>"
            f"Model berikutnya akan diberi ID: <b style='color:#2ecc71;'>{next_id}</b></div>",
            unsafe_allow_html=True,
        )

        # ── Train button ──────────────────────────────────────────────
        if st.button(f"🚀 Latih {next_id} ({param_str})", use_container_width=True, key="btn_train"):
            with st.spinner(f"Melatih {next_id} — {param_str}…"):
                mdl, metrics, cm_data = train_with_custom_params(
                    algo_short, params,
                    splits['X_train_sc'], splits['X_val_sc'], splits['X_test_sc'],
                    splits['y_train'],    splits['y_val'],    splits['y_test'],
                )

            # Assign ID
            st.session_state['algo_counters'][algo_short] = cur_count + 1
            model_id = next_id

            # Build and save bundle
            feat_mode_used = st.session_state.get('train_feat_mode', '19')
            sel_names_used = st.session_state.get('train_sel_names') or FEATURE_NAMES_19
            bundle = {
                'model':            mdl,
                'scaler':           st.session_state['train_scaler'],
                'selector':         st.session_state.get('train_selector'),
                'feature_mode':     feat_mode_used,
                'features_used':    sel_names_used,
                'n_features':       len(sel_names_used),
                'metrics':          metrics,
                'confusion_matrix': cm_data,
                'split_info':       splits['split_info'],
                'model_id':         model_id,
                'algo_short':       algo_short,
                'algo_full':        algo_full,
                'param_str':        param_str,
                'params':           params,
            }
            save_trained_bundle(model_id, bundle)

            # Append to history (lightweight — no model objects)
            st.session_state['model_run_history'].append({
                'model_id':   model_id,
                'algo_short': algo_short,
                'algo_full':  algo_full,
                'param_str':  param_str,
                'metrics':    metrics,
                'cm':         cm_data,
                'feat_mode':  feat_mode_used,
            })

            _update_top5()
            top5_ids_now = st.session_state['top5_model_ids']
            if model_id in top5_ids_now:
                st.success(f"✅ **{model_id}** ({param_str}) berhasil dilatih dan masuk **Top 5**! 🏆")
            else:
                st.success(f"✅ **{model_id}** ({param_str}) berhasil dilatih.")
                st.info(f"Belum masuk Top 5. Top 5 saat ini: {', '.join(top5_ids_now)}")
            st.rerun()

        # ── Recap table ───────────────────────────────────────────────
        history = st.session_state.get('model_run_history', [])
        if history:
            st.markdown("---")
            st.markdown("### 📋 Rekap Semua Pelatihan (Real-time)")
            st.markdown(
                "Baris **hijau tebal** = masuk Top 5 (akan tersedia di halaman Implementasi). "
                "Kolom diurutkan sesuai urutan pelatihan."
            )
            top5_ids_now = st.session_state.get('top5_model_ids', [])
            df_hist = pd.DataFrame([{
                'Model ID':  r['model_id'],
                'Algoritma': r['algo_full'],
                'Parameter': r['param_str'],
                'Accuracy':  r['metrics']['Accuracy'],
                'Precision': r['metrics']['Precision'],
                'Recall':    r['metrics']['Recall'],
                'F1-Score':  r['metrics']['F1-Score'],
                'Waktu (s)': r['metrics']['Execution Time (s)'],
                'Status':    '🏆 Top 5' if r['model_id'] in top5_ids_now else '—',
            } for r in history])

            def _hl_top5(row):
                if row['Status'] == '🏆 Top 5':
                    return ['background-color:rgba(46,204,113,0.22);color:#1e8449;font-weight:bold'] * len(row)
                return [''] * len(row)

            st.dataframe(
                df_hist.style.apply(_hl_top5, axis=1).format({
                    'Accuracy':  '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall':    '{:.4f}',
                    'F1-Score':  '{:.4f}',
                    'Waktu (s)': '{:.4f}',
                }),
                use_container_width=True,
                height=min(500, len(history) * 38 + 55),
            )

            best = max(history, key=lambda r: r['metrics']['Accuracy'])
            st.markdown(
                f"<div class='badge-best'>🏆 Terbaik saat ini: <b>{best['model_id']}</b> "
                f"({best['param_str']}) — Accuracy: {best['metrics']['Accuracy']:.4f} "
                f"| Precision: {best['metrics']['Precision']:.4f} "
                f"| Recall: {best['metrics']['Recall']:.4f} "
                f"| F1-Score: {best['metrics']['F1-Score']:.4f}</div>",
                unsafe_allow_html=True,
            )

            if top5_ids_now:
                st.info(
                    f"💾 **Top 5 tersimpan:** {' · '.join(top5_ids_now)} "
                    "— Siap digunakan di halaman **🎯 Implementasi**."
                )

        # ── Reset buttons ─────────────────────────────────────────────
        st.markdown("---")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("🔄 Mulai Ulang dari Langkah 1", key="btn_reset_step"):
                for k in ['train_images', 'train_step', 'train_feat_mode', 'train_ig_df',
                          'train_sel_names', 'train_scaler', 'train_selector',
                          'train_splits', 'train_feat_sample', 'train_sample_steps']:
                    st.session_state[k] = _defaults.get(k)
                st.session_state['train_step'] = 1
                st.rerun()
        with col_r2:
            if st.button("🗑️ Hapus Semua Riwayat Pelatihan", key="btn_reset_history"):
                st.session_state['model_run_history'] = []
                st.session_state['algo_counters']     = {'LR': 0, 'SVM': 0, 'DT': 0, 'RF': 0, 'GB': 0}
                st.session_state['top5_model_ids']    = []
                cleanup_non_top5([])
                st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# HELPER — Render implementation history
# ──────────────────────────────────────────────────────────────────────────────
def _render_impl_history():
    runs = st.session_state.get('impl_test_runs', [])
    if not runs:
        return

    st.markdown("---")
    # Latest run
    latest = runs[-1]
    st.markdown(
        f"### 📊 Hasil Pengujian — **{latest['model_id']}** "
        f"({latest['algo']}, {latest['param_str']})"
    )

    df_results = pd.DataFrame(latest['results'])
    df_results.columns = ['No', 'Gambar', 'Label Aktual', 'Prediksi', 'Hasil']
    df_results = df_results.set_index('No')

    def _style_hasil(val):
        if val == 'TRUE':
            return 'color:#27ae60; font-weight:bold'
        elif val == 'FALSE':
            return 'color:#e74c3c; font-weight:bold'
        return ''

    st.dataframe(
        df_results.style.map(_style_hasil, subset=['Hasil']),
        use_container_width=True,
        height=min(450, len(latest['results']) * 38 + 55),
    )

    # Summary
    st.markdown("---")
    s1, s2, s3 = st.columns(3)
    s1.metric("Jumlah TRUE",  latest['true_count'])
    s2.metric("Jumlah FALSE", latest['false_count'])
    s3.metric("Presisinya",   f"{latest['presisi'] * 100:.1f}%")

    # Multi-model comparison table
    if len(runs) > 1:
        st.markdown("---")
        st.markdown("### 📋 Rekap Semua Pengujian Model")
        df_runs = pd.DataFrame([{
            'Model':           r['model_id'],
            'Algoritma':       r['algo'],
            'Parameter':       r['param_str'],
            'Jml Gambar':      r['n_gambar'],
            'Jumlah TRUE':     r['true_count'],
            'Jumlah FALSE':    r['false_count'],
            'Presisi':         f"{r['presisi'] * 100:.1f}%",
        } for r in runs])
        st.dataframe(df_runs, use_container_width=True)

        avg_presisi = sum(r['presisi'] for r in runs) / len(runs)
        st.markdown(
            f"<div style='text-align:center;font-size:16px;font-weight:600;"
            f"padding:12px;background:rgba(46,204,113,0.10);border-radius:8px;'>"
            f"📊 Rata-rata Presisi: <b>{avg_presisi * 100:.1f}%</b> "
            f"dari {len(runs)} pengujian model</div>",
            unsafe_allow_html=True,
        )

        # Conclusion
        st.markdown("---")
        st.markdown("### 💬 Kesimpulan")
        best_run  = max(runs, key=lambda r: r['presisi'])
        worst_run = min(runs, key=lambda r: r['presisi'])
        if avg_presisi >= 0.90:
            kes = (
                f"Model-model yang diuji menunjukkan keandalan **sangat tinggi** "
                f"(rata-rata presisi ≥ 90%). "
                f"Model terbaik adalah **{best_run['model_id']}** "
                f"dengan presisi **{best_run['presisi'] * 100:.1f}%**. "
                "Sistem siap digunakan untuk klasifikasi kepadatan gulma secara real-time."
            )
        elif avg_presisi >= 0.70:
            kes = (
                f"Model-model yang diuji menunjukkan keandalan **cukup baik** "
                f"(rata-rata presisi 70–89%). "
                f"Model terbaik: **{best_run['model_id']}** "
                f"(presisi {best_run['presisi'] * 100:.1f}%). "
                f"Model **{worst_run['model_id']}** "
                f"(presisi {worst_run['presisi'] * 100:.1f}%) perlu evaluasi lebih lanjut."
            )
        else:
            kes = (
                f"Rata-rata presisi masih di bawah 70% ({avg_presisi * 100:.1f}%). "
                "Disarankan menambah data training, mencoba kombinasi parameter berbeda, "
                "atau menggunakan mode 39 fitur dengan seleksi Information Gain."
            )
        st.markdown(f"<div class='step-box'>{kes}</div>", unsafe_allow_html=True)

    elif len(runs) == 1:
        p = runs[0]['presisi']
        st.markdown("---")
        st.markdown("### 💬 Kesimpulan")
        if p >= 0.90:
            kes = (
                f"Model **{runs[0]['model_id']}** menunjukkan keandalan **sangat tinggi** "
                f"dengan presisi **{p * 100:.1f}%**."
            )
        elif p >= 0.70:
            kes = (
                f"Model **{runs[0]['model_id']}** menunjukkan keandalan **cukup baik** "
                f"dengan presisi **{p * 100:.1f}%**."
            )
        else:
            kes = (
                f"Model **{runs[0]['model_id']}** memerlukan evaluasi lebih lanjut. "
                f"Presisi **{p * 100:.1f}%** masih di bawah 70%."
            )
        st.markdown(f"<div class='step-box'>{kes}</div>", unsafe_allow_html=True)

    if st.button("🗑️ Hapus Riwayat Pengujian", key="btn_clear_impl"):
        st.session_state['impl_test_runs'] = []
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 2 — IMPLEMENTASI
# ──────────────────────────────────────────────────────────────────────────────
def page_implementasi():
    st.title("🎯 Implementasi — Evaluasi Model")
    st.markdown(
        "Pilih model dari Top 5 hasil pemodelan, upload gambar (maks. 10), tentukan label aktual "
        "secara manual, lalu klik **Uji Model** untuk melihat hasil evaluasi."
    )

    top5_ids = st.session_state.get('top5_model_ids', [])
    if not top5_ids:
        st.warning(
            "⚠️ Belum ada model yang tersimpan. "
            "Latih model terlebih dahulu di halaman **🔬 Pemodelan**."
        )
        _render_impl_history()
        return

    # ── Model selector ────────────────────────────────────────────────
    st.markdown("### 🤖 Pilih Model")
    selected_id = st.selectbox(
        "Model (dari Top 5 hasil pemodelan):",
        top5_ids,
        key="impl_model_select",
    )

    try:
        bundle = load_trained_bundle(selected_id)
    except FileNotFoundError:
        st.error(
            f"File model **{selected_id}** tidak ditemukan di disk. "
            "Coba latih ulang di halaman Pemodelan."
        )
        _render_impl_history()
        return

    m = bundle.get('metrics', {})
    st.markdown(
        f"<div class='model-card'>"
        f"<b>Model:</b> {selected_id} &nbsp;|&nbsp; "
        f"<b>Algoritma:</b> {bundle.get('algo_full','?')} &nbsp;|&nbsp; "
        f"<b>Parameter:</b> {bundle.get('param_str','?')} &nbsp;|&nbsp; "
        f"<b>Mode Fitur:</b> {bundle.get('feature_mode','?')} fitur<br>"
        f"<b>Accuracy:</b> {m.get('Accuracy',0):.4f} &nbsp;|&nbsp; "
        f"<b>Precision:</b> {m.get('Precision',0):.4f} &nbsp;|&nbsp; "
        f"<b>Recall:</b> {m.get('Recall',0):.4f} &nbsp;|&nbsp; "
        f"<b>F1-Score:</b> {m.get('F1-Score',0):.4f}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Batch upload ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📤 Upload Gambar Uji (Maks. 10 Gambar)")
    st.info("📎 Format: **JPG / JPEG** · Maksimal **10 gambar** per pengujian")
    uploaded_files = st.file_uploader(
        "Upload gambar uji:",
        type=['jpg', 'jpeg'],
        accept_multiple_files=True,
        key="impl_upload",
    )

    if uploaded_files:
        files_to_test = uploaded_files[:10]
        if len(uploaded_files) > 10:
            st.warning(
                f"Hanya 10 gambar pertama yang diproses ({len(uploaded_files)} gambar diupload)."
            )

        # ── Ground truth labels ───────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🏷️ Tentukan Label Aktual")
        st.markdown(
            "Untuk setiap gambar, pilih label aktual yang sudah Anda identifikasi secara visual:"
        )

        n_cols    = min(len(files_to_test), 5)
        grid_cols = st.columns(n_cols)
        label_map = {}
        for i, f in enumerate(files_to_test):
            col = grid_cols[i % n_cols]
            with col:
                f.seek(0)
                st.image(f.read(), caption=f.name, use_container_width=True)
                label_map[i] = st.selectbox(
                    f"Label #{i + 1}",
                    ["Renggang", "Sedang", "Padat"],
                    key=f"impl_label_{i}",
                )

        # ── Uji button ────────────────────────────────────────────────
        st.markdown("---")
        if st.button("🔍 Uji Model", use_container_width=True, key="btn_impl_test"):
            results = []
            errors  = []
            with st.spinner(f"Mengklasifikasikan {len(files_to_test)} gambar dengan {selected_id}…"):
                for i, f in enumerate(files_to_test):
                    f.seek(0)
                    img_bytes = f.read()
                    try:
                        prediction, _, _ = run_inference(bundle, img_bytes)
                    except Exception as e:
                        errors.append(f"Gambar {f.name}: {e}")
                        continue
                    aktual = label_map[i]
                    hasil  = prediction == aktual
                    results.append({
                        'no':       i + 1,
                        'gambar':   f.name,
                        'aktual':   aktual,
                        'prediksi': prediction,
                        'hasil':    'TRUE' if hasil else 'FALSE',
                    })

            if errors:
                for err in errors:
                    st.error(f"⚠️ {err}")

            if results:
                true_count  = sum(1 for r in results if r['hasil'] == 'TRUE')
                false_count = len(results) - true_count
                presisi     = true_count / len(results)

                st.session_state['impl_test_runs'].append({
                    'model_id':    selected_id,
                    'algo':        bundle.get('algo_full', '?'),
                    'param_str':   bundle.get('param_str', '?'),
                    'results':     results,
                    'true_count':  true_count,
                    'false_count': false_count,
                    'presisi':     presisi,
                    'n_gambar':    len(results),
                })
                st.rerun()

    _render_impl_history()


# ──────────────────────────────────────────────────────────────────────────────
# ROUTER
# ──────────────────────────────────────────────────────────────────────────────
if "Pemodelan" in page_choice:
    page_pemodelan()
else:
    page_implementasi()
