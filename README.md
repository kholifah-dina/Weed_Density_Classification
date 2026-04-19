# 🌿 Weed Density Classification

Sistem klasifikasi kepadatan gulma berbasis *Machine Learning* menggunakan **Streamlit**.  
Proyek ini merupakan bagian dari Tugas Akhir/Skripsi di **Universitas Telkom Purwokerto**.

> **Pipeline:** Resize → Gaussian Blur → Segmentasi HSV → Morphological Closing →  
> Ekstraksi Fitur (19 atau 39) → *Information Gain* (14 terbaik) → **Gradient Boosting Classifier**

---

## Daftar Isi

1. [Tentang Sistem](#tentang-sistem)
2. [Struktur Proyek](#struktur-proyek)
3. [Tech Stack](#tech-stack)
4. [Cara Menjalankan Lokal](#cara-menjalankan-lokal)
5. [Panduan Kolaborasi](#panduan-kolaborasi)
6. [Alur Penggunaan Aplikasi](#alur-penggunaan-aplikasi)
7. [Pipeline Machine Learning](#pipeline-machine-learning)
8. [Kelas Kepadatan Gulma](#kelas-kepadatan-gulma)
9. [Troubleshooting](#troubleshooting)

---

## Tentang Sistem

Aplikasi web interaktif yang mengklasifikasikan **tingkat kepadatan gulma** pada foto lahan pertanian ke dalam tiga kelas:

| Kelas | Deskripsi |
|-------|-----------|
| 🟢 **Renggang** | Kepadatan gulma rendah — populasi jarang |
| 🟡 **Sedang** | Kepadatan gulma sedang — perlu pemantauan |
| 🔴 **Padat** | Kepadatan gulma tinggi — perlu penanganan segera |

Sistem menggunakan **4 menu sidebar** yang bisa diakses langsung tanpa login:

| Menu | Fungsi |
|------|--------|
| 📚 Alur Pelatihan | Training model 4 langkah dengan visualisasi preprocessing, tabel ekstraksi fitur, dan grafik Information Gain |
| 🎯 Pengujian Gambar | Upload 1 gambar → pilih model → prediksi + nilai fitur + metrik + Confusion Matrix |
| 🔬 Eksperimen Parameter | Tuning hyperparameter dengan progress real-time, grafik metrik, dan Classification Report |
| 📊 Dashboard Hasil | Rekap preprocessing, perbandingan semua model, dan grafik semua eksperimen |

> Mendukung **Light Mode** dan **Dark Mode** — warna teks dan elemen UI menyesuaikan tema secara otomatis.

---

## Struktur Proyek

```
weed-density-app/
│
├── requirements.txt                  # Dependensi Python (dengan version pinning)
├── runtime.txt                       # Versi Python (3.10)
├── .gitignore                        # File yang dikecualikan dari git
├── README.md                         # Dokumentasi ini
├── PANDUAN_PENGGUNAAN.md             # Panduan penggunaan lengkap
│
└── streamlit_app/
    ├── app.py                        # UI Streamlit — 4 menu sidebar
    ├── predict.py                    # Training, inference, save/load per model
    ├── preprocessing.py              # Pipeline preprocessing (resize, blur, HSV, morph)
    ├── feature_extraction.py         # Ekstraksi 19 fitur atau 39 fitur (dengan GLCM)
    ├── Data_ekstraksi_Fitur_Gulma.csv  # Dataset (2.097 sampel, 39 fitur + Class)
    └── models/                       # Folder model — dibuat otomatis saat training
        ├── DT.joblib                 # Decision Tree (dibuat setelah training)
        ├── LR.joblib                 # Logistic Regression
        ├── SVM.joblib                # Support Vector Machine
        ├── RF.joblib                 # Random Forest
        └── GB.joblib                 # Gradient Boosting ← Model Utama Penelitian
```

> ⚠️ **File `.joblib` tidak disertakan di GitHub** (gitignored karena ukuran besar).  
> Setelah clone, buka menu **📚 Alur Pelatihan** untuk melatih model terlebih dahulu.

---

## Tech Stack

| Komponen | Library / Versi |
|----------|----------------|
| Web Framework | `streamlit >= 1.28` |
| Machine Learning | `scikit-learn >= 1.3` |
| Image Processing | `opencv-python-headless >= 4.8` |
| Image Features (GLCM) | `scikit-image >= 0.21` |
| Data Manipulation | `pandas >= 2.0`, `numpy >= 1.24` |
| Visualisasi | `plotly >= 5.17` |
| Model Persistence | `joblib >= 1.3` |
| Runtime | Python 3.10 |

---

## Cara Menjalankan Lokal

### Prasyarat

- **Python 3.10** sudah terpasang → cek: `python --version`
- **Git** sudah terpasang → cek: `git --version`

### 1. Clone Repositori

```bash
git clone https://github.com/kholifah-dina/Weed_Density_Classification.git
cd Weed_Density_Classification
```

### 2. Buat Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

```bash
cd streamlit_app
streamlit run app.py
```

Browser terbuka otomatis di **http://localhost:8501**

> 💡 Setelah pertama kali clone, langsung buka menu **📚 Alur Pelatihan** untuk melatih model sebelum bisa melakukan pengujian.

---

## Panduan Kolaborasi

### Alur Kerja Git

```bash
# ── SEBELUM mulai bekerja ────────────────────────────────────────
git checkout main
git pull origin main           # ambil update terbaru

git checkout -b nama-fitur     # buat branch baru
# Contoh: git checkout -b fix-hsv-threshold

# ── SAAT bekerja ─────────────────────────────────────────────────
# edit file yang diperlukan...

git status                     # cek file yang berubah
git add streamlit_app/app.py   # tambah file spesifik
git commit -m "fix: deskripsi singkat perubahan"

# ── SETELAH selesai ───────────────────────────────────────────────
git push origin nama-fitur     # push ke GitHub
# Buka GitHub → Compare & pull request
```

### Mengambil Update dari Rekan (Sync)

```bash
git checkout main
git pull origin main
```

### Konvensi Pesan Commit

| Prefix | Kapan Digunakan |
|--------|----------------|
| `feat:` | Menambahkan fitur baru |
| `fix:` | Memperbaiki bug |
| `refactor:` | Mengubah struktur kode tanpa mengubah perilaku |
| `docs:` | Mengubah dokumentasi |
| `style:` | Perubahan tampilan/UI |
| `data:` | Menambah atau mengubah dataset |

### File yang TIDAK Boleh Di-commit

File berikut sudah dikecualikan via `.gitignore`:

| File/Folder | Alasan |
|------------|--------|
| `streamlit_app/models/*.joblib` | File model biner — ukuran besar, di-generate ulang via training |
| `__pycache__/`, `*.pyc` | Cache Python otomatis |
| `venv/` | Virtual environment lokal |
| `.env` | Variabel lingkungan sensitif |

---

## Alur Penggunaan Aplikasi

### 📚 Alur Pelatihan (Training)

Proses training dibagi menjadi **4 langkah berurutan** yang harus diikuti secara urut:

#### Langkah 1 — Upload Dataset Gambar

- Upload gambar untuk **3 kelas**: Renggang, Sedang, Padat
- Format yang diterima: **JPG / JPEG**
- **Minimal 9 gambar per kelas** (syarat untuk stratified split 80:10:10)
- **Maksimal 190 gambar per kelas**
- Sistem otomatis menampilkan ringkasan jumlah gambar per kelas

#### Langkah 2 — Visualisasi Preprocessing

Sistem menampilkan **4 tahap preprocessing** dari contoh gambar kelas Padat:

| Tahap | Proses |
|-------|--------|
| ① Resize 224×224 | Menyamakan ukuran semua gambar |
| ② Gaussian Blur (5×5) | Meredam noise untuk segmentasi akurat |
| ③ HSV Thresholding | Isolasi piksel hijau (H:25–75°, S:40–255, V:50–255) |
| ④ Morphological Closing | Mengisi celah pada mask gulma |

#### Langkah 3 — Ekstraksi Fitur

Pilih salah satu mode fitur:

| Mode | Fitur | Seleksi | Input ke Model |
|------|-------|---------|---------------|
| **19 Fitur** | RGB mean/std (6) + HSV mean/std (6) + Hu Moments (7) | Tidak ada | 19 fitur langsung |
| **39 Fitur** | GLCM 5×4 sudut (20) + RGB/HSV/Hu (19) | Information Gain | 14 fitur terbaik |

Setelah klik **Jalankan Ekstraksi**, sistem menampilkan:
- **Tabel contoh hasil ekstraksi** — 3 sampel per kelas dalam bentuk vektor numerik
- **Mode 39 Fitur:**
  - Grafik bar chart Information Gain (hijau = dipilih, abu-abu = tidak dipilih)
  - Tabel ranking semua 39 fitur berdasarkan IG Score
  - Tabel 14 fitur terpilih dengan gradient warna IG Score
- **Mode 19 Fitur:**
  - Tabel daftar 19 fitur dengan kelompok (RGB / HSV / Hu Moments) dan deskripsi

> 📌 14 fitur terpilih bersifat **dinamis** — ditentukan otomatis oleh Information Gain berdasarkan dataset yang digunakan.

#### Langkah 4 — Pilih Algoritma & Latih Model

- Pilih algoritma dari dropdown: DT / LR / SVM / RF / GB
- Klik **Latih** — model disimpan otomatis ke `models/{NAMA}.joblib`
- Hasil langsung ditampilkan: metrik 6 kolom + Confusion Matrix
- Ulangi dengan algoritma berbeda tanpa perlu upload ulang gambar

---

### 🎯 Pengujian Gambar (Testing)

1. Buka **🎯 Pengujian Gambar**
2. Upload 1 gambar gulma (JPG/JPEG)
3. Lihat visualisasi 4 tahap preprocessing otomatis
4. Pilih model dari dropdown (hanya model yang sudah dilatih)
5. Sistem menampilkan:
   - Tabel nilai semua fitur yang diekstrak dari gambar
   - **Hasil prediksi**: Renggang / Sedang / Padat dengan warna dan deskripsi
   - **Metrik evaluasi** model (Accuracy, Precision, Recall, F1-Score, Val Accuracy)
   - **Confusion Matrix** interaktif dengan caption penjelasan
6. Ganti model di dropdown untuk membandingkan prediksi semua model pada gambar yang sama

---

### 🔬 Eksperimen Parameter

1. Buka **🔬 Eksperimen Parameter**
2. Pilih mode fitur (19 atau 39 fitur)
3. Pilih model dan nilai parameter yang ingin diuji (multiselect)
4. Klik **Jalankan Eksperimen** — progress tampil real-time dengan label bersih per iterasi
5. Hasil ditampilkan:
   - Tabel metrik semua kombinasi (baris terbaik di-highlight hijau)
   - Grafik tren metrik vs nilai parameter (line/bar chart)
   - Confusion Matrix parameter terbaik
   - Classification Report dalam expander

> 💡 Hasil eksperimen terakumulasi — jalankan beberapa kali untuk membandingkan kombinasi parameter berbeda. Klik **Hapus Riwayat** untuk menghapus hasil model tertentu.

---

### 📊 Dashboard Hasil

1. Buka **📊 Dashboard Hasil**
2. Lihat rekap lengkap:
   - **Rekap Preprocessing** — penjelasan 4 tahap + contoh gambar hasil dari training terakhir
   - **Perbandingan Semua Model** — tabel + bar chart Accuracy/Precision/Recall/F1 semua model terlatih
   - **Confusion Matrix semua model** dalam expander
   - **Grafik semua eksperimen** per model
3. Cocok untuk laporan, presentasi, atau sidang skripsi

> 📖 Untuk panduan detail setiap menu, lihat [PANDUAN_PENGGUNAAN.md](PANDUAN_PENGGUNAAN.md)

---

## Pipeline Machine Learning

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Foto Lahan Gulma                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
            ┌────────────────▼────────────────┐
            │        preprocessing.py          │
            │  ① Resize 224×224 px             │
            │  ② Gaussian Blur (kernel 5×5)    │
            │  ③ Konversi BGR → HSV            │
            │  ④ Threshold HSV                 │
            │     H:25–75°, S:40–255, V:50–255 │
            │  ⑤ Morphological Closing (5×5)  │
            └────────────────┬────────────────┘
                             │
            ┌────────────────▼────────────────┐
            │      feature_extraction.py       │
            │                                  │
            │  Mode 19 Fitur (tanpa GLCM):     │
            │  • RGB mean & std  (6 fitur)     │
            │  • HSV mean & std  (6 fitur)     │
            │  • Hu Moments      (7 fitur)     │
            │                                  │
            │  Mode 39 Fitur (dengan GLCM):    │
            │  • GLCM 5×4 sudut  (20 fitur)   │
            │  • RGB + HSV + Hu  (19 fitur)   │
            │  → Information Gain → Top 14    │
            └────────────────┬────────────────┘
                             │
            ┌────────────────▼────────────────┐
            │            predict.py            │
            │  StandardScaler (normalisasi)    │
            │  Split 80:10:10 (stratified)     │
            │  Min. 9 sampel/kelas             │
            │                                  │
            │  5 Algoritma (tiap disimpan      │
            │  terpisah sebagai .joblib):      │
            │  • Decision Tree   → DT.joblib   │
            │  • Logistic Reg.   → LR.joblib   │
            │  • SVM (RBF)       → SVM.joblib  │
            │  • Random Forest   → RF.joblib   │
            │  • Gradient Boost* → GB.joblib   │
            └────────────────┬────────────────┘
                             │
            ┌────────────────▼────────────────┐
            │  OUTPUT: Renggang / Sedang / Padat│
            └─────────────────────────────────┘

  * Model utama penelitian
```

### Mode Fitur & Seleksi

| Mode | Jumlah Fitur Input | Seleksi | Fitur ke Model |
|------|-------------------|---------|---------------|
| Tanpa GLCM | 19 | Tidak ada | 19 fitur langsung |
| Dengan GLCM | 39 | Information Gain (LAN) | 14 fitur terbaik |

### Dataset

- **Total:** 2.097 sampel
- **Distribusi:** Renggang 610 (29%) · Sedang 869 (41%) · Padat 618 (30%)
- **Split:** 80% Train · 10% Validasi · 10% Test (stratified)
- **File:** `streamlit_app/Data_ekstraksi_Fitur_Gulma.csv`

### Konfigurasi Model Default

| Model | Parameter Utama |
|-------|----------------|
| Decision Tree | `max_depth=3`, `class_weight='balanced'` |
| Logistic Regression | `solver='lbfgs'`, `max_iter=300`, `class_weight='balanced'` |
| SVM | `kernel='rbf'`, `C=5`, `gamma=0.01`, `class_weight='balanced'` |
| Random Forest | `n_estimators=500`, `class_weight='balanced'` |
| Gradient Boosting | `n_estimators=300`, `learning_rate=0.1`, `sample_weight` (balanced) |

---

## Kelas Kepadatan Gulma

| Kelas | Warna | Penjelasan |
|-------|-------|-----------|
| 🟢 **Renggang** | Hijau | Gulma tumbuh jarang, tidak mengganggu tanaman utama secara signifikan |
| 🟡 **Sedang** | Kuning | Gulma mulai kompetitif, perlu pemantauan dan pengendalian dini |
| 🔴 **Padat** | Merah | Gulma sangat rapat, kompetisi nutrisi tinggi, butuh penanganan segera |

---

## Troubleshooting

### "Belum ada model yang tersedia"
File `.joblib` belum ada — buka menu **📚 Alur Pelatihan** dan latih minimal satu model.

### "Setiap kelas membutuhkan minimal 9 gambar"
Stratified split 80:10:10 membutuhkan minimal **9 gambar per kelas** (27 total).  
Tambah gambar hingga setiap kelas memiliki minimal 9 gambar, lalu ulangi dari Langkah 1.

### Error install: `ERROR: Could not build wheels for opencv`
```bash
pip install opencv-python-headless   # gunakan versi headless (tanpa GUI)
```

### Port 8501 sudah dipakai
```bash
streamlit run app.py --server.port 8502
```

### `ModuleNotFoundError` saat menjalankan app
```bash
venv\Scripts\activate          # Windows — aktifkan virtual environment
pip install -r requirements.txt
```

### Hasil eksperimen hilang setelah refresh browser
Hasil eksperimen disimpan di session state (memori tab browser) — simpan screenshot atau catat hasilnya sebelum menutup/refresh halaman.

### Training sangat lambat
Normal untuk Random Forest (500 pohon) dan Gradient Boosting (300 iterasi). Tunggu hingga selesai. Progress ditampilkan real-time di status box.

### Grafik tidak terbaca di Dark Mode
Semua grafik menggunakan background transparan dan elemen UI memiliki CSS dark mode override otomatis. Pastikan menggunakan Streamlit versi terbaru (`streamlit >= 1.28`).

> 📖 Troubleshooting lebih lengkap ada di [PANDUAN_PENGGUNAAN.md](PANDUAN_PENGGUNAAN.md)

---

## Kontak & Informasi

**Peneliti:** Kholifah Dina  
**Institusi:** Universitas Telkom Purwokerto  
**Topik:** Klasifikasi Kepadatan Gulma menggunakan HSV Segmentation & Gradient Boosting  

---

*Sistem ini dikembangkan untuk keperluan penelitian Tugas Akhir.*
