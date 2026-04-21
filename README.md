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

Sistem menggunakan **2 halaman** yang diakses melalui sidebar:

| Halaman | Fungsi |
|---------|--------|
| 🔬 **Pemodelan** | Training model 4 langkah: upload dataset → preprocessing → ekstraksi fitur → pelatihan dinamis multi-algoritma dengan ID unik dan tabel rekap real-time |
| 🎯 **Implementasi** | Batch upload hingga 10 gambar → tentukan label aktual → prediksi model → tabel TRUE/FALSE → ringkasan presisi & narasi kesimpulan |

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
    ├── app.py                        # UI Streamlit — 2 halaman (Pemodelan & Implementasi)
    ├── predict.py                    # Training, inference, save/load model dinamis
    ├── preprocessing.py              # Pipeline preprocessing (resize, blur, HSV, morph)
    ├── feature_extraction.py         # Ekstraksi 19 fitur atau 39 fitur (dengan GLCM)
    ├── Data_ekstraksi_Fitur_Gulma.csv  # Dataset (2.097 sampel, 39 fitur + Class)
    └── models/                       # Folder model — dibuat otomatis saat training
        └── TRAINED_<ID>.joblib       # Contoh: TRAINED_LR1.joblib, TRAINED_GB3.joblib
```

> ⚠️ **File `.joblib` tidak disertakan di GitHub** (gitignored karena ukuran besar).  
> Setelah clone, buka halaman **🔬 Pemodelan** untuk melatih model terlebih dahulu.

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

> 💡 Setelah pertama kali clone, langsung buka halaman **🔬 Pemodelan** untuk melatih model sebelum bisa melakukan pengujian di halaman **🎯 Implementasi**.

---

## Panduan Kolaborasi

### Alur Kerja Git

```bash
# ── SEBELUM mulai bekerja ────────────────────────────────────────
git checkout main
git pull origin main           # ambil update terbaru

git checkout -b nama-fitur     # buat branch baru

# ── SAAT bekerja ─────────────────────────────────────────────────
git status
git add streamlit_app/app.py
git commit -m "feat: deskripsi singkat perubahan"

# ── SETELAH selesai ───────────────────────────────────────────────
git push origin nama-fitur
# Buka GitHub → Compare & pull request
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

| File/Folder | Alasan |
|------------|--------|
| `streamlit_app/models/*.joblib` | File model biner — di-generate ulang via training |
| `__pycache__/`, `*.pyc` | Cache Python otomatis |
| `venv/` | Virtual environment lokal |
| `.env` | Variabel lingkungan sensitif |

---

## Alur Penggunaan Aplikasi

### 🔬 Halaman Pemodelan

Proses training dibagi menjadi **4 langkah berurutan**:

#### Langkah 1 — Upload Dataset Gambar

- Upload gambar untuk **3 kelas**: Renggang, Sedang, Padat
- Format: **JPG / JPEG** · **Minimal 9 gambar/kelas** · **Maksimal 190 gambar/kelas**
- Sistem menampilkan ringkasan jumlah gambar per kelas secara otomatis

#### Langkah 2 — Visualisasi Preprocessing

Sistem menampilkan **4 tahap preprocessing** dari contoh gambar kelas Padat:

| Tahap | Proses |
|-------|--------|
| ① Resize 224×224 | Menyamakan ukuran semua gambar |
| ② Gaussian Blur (5×5) | Meredam noise untuk segmentasi akurat |
| ③ HSV Thresholding | Isolasi piksel hijau (H:25–75°, S:40–255, V:50–255) |
| ④ Morphological Closing | Mengisi celah pada mask gulma |

#### Langkah 3 — Ekstraksi Fitur

| Mode | Fitur | Seleksi | Input ke Model |
|------|-------|---------|---------------|
| **19 Fitur** | RGB mean/std (6) + HSV mean/std (6) + Hu Moments (7) | Tidak ada | 19 fitur langsung |
| **39 Fitur** | GLCM 5×4 sudut (20) + RGB/HSV/Hu (19) | Information Gain | 14 fitur terbaik |

Output: tabel sampel fitur + grafik IG + tabel ranking + 14 fitur terpilih.

#### Langkah 4 — Pelatihan Dinamis

- Pilih algoritma dan **parameter fixed sesuai skripsi**:

| Algoritma | Parameter | Nilai Tersedia |
|-----------|-----------|---------------|
| Logistic Regression | `max_iter` | 100, 300, 500, 700, 1000 |
| SVM | `kernel` | linear, rbf, poly |
| Decision Tree | `max_depth` | 3, 5, 7, 9, 11 |
| Random Forest | `n_estimators` | 100, 200, 300, 400, 500 |
| Gradient Boosting | `n_estimators` + `learning_rate` | (100/200/300) + (0.01/0.1/1) |

- Setiap klik **Latih** → model diberi **ID unik sekuensial**: LR1, LR2, SVM1, GB1, dst.
- **Tabel rekap real-time** bertambah setiap training: Model ID · Algoritma · Parameter · Accuracy · Precision · Recall · F1-Score · Waktu
- **5 model akurasi tertinggi** disimpan otomatis ke disk dan ditampilkan di sidebar

---

### 🎯 Halaman Implementasi

1. Pilih model dari **Top 5** yang tersimpan (contoh: LR2, SVM1, GB3)
2. Upload hingga **10 gambar** sekaligus
3. Untuk setiap gambar → pilih **Label Aktual** (Renggang / Sedang / Padat) dari dropdown
4. Klik **Uji Model** → tabel hasil muncul:

| No | Gambar | Label Aktual | Prediksi | Hasil |
|----|--------|-------------|----------|-------|
| 1 | img1.jpg | Padat | Padat | **TRUE** (hijau) |
| 2 | img2.jpg | Sedang | Padat | **FALSE** (merah) |

5. Summary otomatis: **Jumlah TRUE · Jumlah FALSE · Presisinya (%)**
6. Uji model lain → rekap multi-model + **rata-rata presisi** + **narasi kesimpulan**

---

## Pipeline Machine Learning

```
┌────────────────────────────────────────────────────────────────┐
│                   INPUT: Foto Lahan Gulma                       │
└───────────────────────────┬────────────────────────────────────┘
                            │
           ┌────────────────▼────────────────┐
           │        preprocessing.py          │
           │  ① Resize 224×224 px             │
           │  ② Gaussian Blur (kernel 5×5)    │
           │  ③ Konversi BGR → HSV            │
           │  ④ Threshold H:25–75°            │
           │  ⑤ Morphological Closing (5×5)  │
           └────────────────┬────────────────┘
                            │
           ┌────────────────▼────────────────┐
           │      feature_extraction.py       │
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
           │  Training dinamis per parameter: │
           │  • LR1, LR2, …  → max_iter      │
           │  • SVM1, SVM2,… → kernel        │
           │  • DT1, DT2, …  → max_depth     │
           │  • RF1, RF2, …  → n_estimators  │
           │  • GB1, GB2, …  → n_est + lr    │
           │                                  │
           │  Top 5 → TRAINED_<ID>.joblib    │
           └────────────────┬────────────────┘
                            │
           ┌────────────────▼────────────────┐
           │  OUTPUT: Renggang / Sedang / Padat│
           └─────────────────────────────────┘
```

### Dataset

- **Total:** 2.097 sampel
- **Distribusi:** Renggang 610 (29%) · Sedang 869 (41%) · Padat 618 (30%)
- **Split:** 80% Train · 10% Validasi · 10% Test (stratified)
- **File:** `streamlit_app/Data_ekstraksi_Fitur_Gulma.csv`

---

## Kelas Kepadatan Gulma

| Kelas | Warna | Penjelasan |
|-------|-------|-----------|
| 🟢 **Renggang** | Hijau | Gulma tumbuh jarang, tidak mengganggu tanaman utama secara signifikan |
| 🟡 **Sedang** | Kuning | Gulma mulai kompetitif, perlu pemantauan dan pengendalian dini |
| 🔴 **Padat** | Merah | Gulma sangat rapat, kompetisi nutrisi tinggi, butuh penanganan segera |

---

## Troubleshooting

### "Belum ada model yang tersimpan" di halaman Implementasi
Latih minimal satu model di halaman **🔬 Pemodelan** hingga masuk **Top 5**.

### "Setiap kelas membutuhkan minimal 9 gambar"
Stratified split 80:10:10 membutuhkan minimal **9 gambar per kelas** (27 total).  
Tambah gambar lalu ulangi dari Langkah 1.

### Error install: `ERROR: Could not build wheels for opencv`
```bash
pip install opencv-python-headless
```

### Port 8501 sudah dipakai
```bash
streamlit run app.py --server.port 8502
```

### `ModuleNotFoundError` saat menjalankan app
```bash
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Hasil pengujian hilang setelah refresh browser
Data rekap pelatihan dan pengujian disimpan di **session state** (memori tab browser).  
Simpan screenshot sebelum menutup atau refresh halaman.

### Grafik tidak terbaca di Dark Mode
Semua grafik menggunakan background transparan dan CSS dark mode override otomatis.  
Pastikan menggunakan Streamlit versi terbaru (`streamlit >= 1.28`).

> 📖 Troubleshooting lebih lengkap ada di [PANDUAN_PENGGUNAAN.md](PANDUAN_PENGGUNAAN.md)

---

## Kontak & Informasi

**Peneliti:** Kholifah Dina  
**Institusi:** Universitas Telkom Purwokerto  
**Topik:** Klasifikasi Kepadatan Gulma menggunakan HSV Segmentation & Gradient Boosting  

---

*Sistem ini dikembangkan untuk keperluan penelitian Tugas Akhir.*
