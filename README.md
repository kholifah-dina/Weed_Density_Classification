# 🌿 Weed Density Classification

Sistem klasifikasi kepadatan gulma berbasis *Machine Learning* menggunakan **Streamlit**.  
Proyek ini merupakan bagian dari Tugas Akhir/Skripsi di **Universitas Telkom Purwokerto**.

> **Pipeline:** Resize → Gaussian Blur → Segmentasi HSV → Morphological Closing →  
> Ekstraksi Fitur (19 atau 39) → *Information Gain* (14 terbaik) → **Gradient Boosting Classifier**  
> + **Stacking Ensemble**: Baseline (LR/SVM/DT/RF) + GB sebagai meta-learner

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
| 🔬 **Pemodelan** | Training model 4 langkah berurutan: upload dataset → preprocessing → ekstraksi fitur → pelatihan dinamis (Algoritma Tunggal + Kombinasi Baseline+GB) |
| 🎯 **Implementasi** | Upload gambar uji per kelas (maks. 50/kelas) → inferensi otomatis → tabel pivot hasil semua model → summary presisi per kelas → narasi kesimpulan |

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
├── PANDUAN_PENGGUNAAN.md             # Panduan lengkap + alur activity diagram
│
└── streamlit_app/
    ├── app.py                        # UI Streamlit — 2 halaman (Pemodelan & Implementasi)
    ├── predict.py                    # Training, stacking, inference, save/load model
    ├── preprocessing.py              # Pipeline preprocessing (resize, blur, HSV, morph)
    ├── feature_extraction.py         # Ekstraksi 19 fitur atau 39 fitur (dengan GLCM)
    ├── Data_ekstraksi_Fitur_Gulma.csv  # Dataset (2.097 sampel, 39 fitur + Class)
    └── models/                       # Folder model — dibuat otomatis saat training
        ├── TRAINED_LR1.joblib        # Contoh model tunggal
        ├── TRAINED_GB2.joblib
        └── TRAINED_LRGB1.joblib      # Contoh model kombinasi (Stacking)
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
- Format: **JPG / JPEG** · Minimal **9 gambar/kelas** · Maksimal **190 gambar/kelas**
- Sistem menampilkan ringkasan jumlah gambar per kelas secara otomatis
- Klik **Lanjut ke Langkah 2** setelah semua kelas terisi

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

Output: tabel sampel fitur + grafik Information Gain + tabel ranking + 14 fitur terpilih.

#### Langkah 4 — Pelatihan Dinamis

Langkah 4 memiliki **2 tab pelatihan**:

---

**Tab 🔵 Algoritma Tunggal**

Pilih algoritma dan **satu atau lebih** nilai parameter (multi-select):

| Algoritma | Parameter | Nilai Tersedia |
|-----------|-----------|---------------|
| Logistic Regression | `max_iter` | 100, 300, 500, 700, 1000 |
| SVM | `kernel` | linear, rbf, poly |
| Decision Tree | `max_depth` | 3, 5, 7, 9, 11 |
| Random Forest | `n_estimators` | 100, 200, 300, 400, 500 |
| Gradient Boosting | `n_estimators` + `learning_rate` | (100/200/300) × (0.01/0.1/1) |

- Memilih beberapa nilai sekaligus → semua dilatih dalam satu klik (LR1, LR2, LR3 dst.)
- **1 model terbaik per algoritma** disimpan otomatis (LR terbaik, SVM terbaik, dst.)
- **Tabel rekap real-time**: Model ID · Algoritma · Parameter · Accuracy · Precision · Recall · F1 · Waktu
- Baris hijau = model terbaik algoritma tersebut

---

**Tab 🔗 Kombinasi Baseline + GB (Stacking)**

Menggabungkan parameter terbaik algoritma baseline dengan Gradient Boosting sebagai meta-learner:

| Kombinasi | ID Model | Syarat |
|-----------|----------|--------|
| LR + GB | `LRGB1` | LR dan GB sudah dilatih |
| SVM + GB | `SVMGB1` | SVM dan GB sudah dilatih |
| DT + GB | `DTGB1` | DT dan GB sudah dilatih |
| RF + GB | `RFGB1` | RF dan GB sudah dilatih |

- Parameter diambil **otomatis** dari model terbaik masing-masing algoritma
- Setiap kombinasi hanya dilatih **1 kali** (ID tetap, tidak sequential)
- Setelah dilatih → tombol berubah jadi badge hasil (Accuracy/Precision/Recall/F1)

---

### 🎯 Halaman Implementasi

1. Pilih model dari daftar **model terbaik per algoritma** yang tersimpan
2. Upload gambar uji **per kelas** ke 3 zona:

| Zona | Kelas | Batas |
|------|-------|-------|
| 🟢 | Renggang | Maks. 50 gambar |
| 🟡 | Sedang | Maks. 50 gambar |
| 🔴 | Padat | Maks. 50 gambar |

3. Label aktual **otomatis** dari zona upload — tidak perlu pilih manual
4. Klik **Uji Model** → progress bar per gambar → hasil inferensi
5. Tabel hasil individual (gambar, label aktual, prediksi, TRUE/FALSE)
6. Uji model lain → **tabel pivot** merangkum semua model:

| Data ke- | Gambar | Aktual | LR1 Prediksi | LR1 Hasil | SVM1 Prediksi | SVM1 Hasil | ... |
|----------|--------|--------|-------------|----------|--------------|----------|-----|
| 1 | img1.jpg | Padat | Padat | TRUE | Renggang | FALSE | ... |

7. Summary per model: Jumlah TRUE · Jumlah FALSE · Total · Presisinya · Benar per Kelas
8. Rata-rata presisi + **narasi kesimpulan** otomatis

---

## Pipeline Machine Learning

```
┌────────────────────────────────────────────────────────────────────┐
│                     INPUT: Foto Lahan Gulma                         │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
             ┌────────────────▼────────────────┐
             │          preprocessing.py         │
             │  ① Resize 224×224 px              │
             │  ② Gaussian Blur (kernel 5×5)     │
             │  ③ Konversi BGR → HSV             │
             │  ④ Threshold H:25–75, S:40–255    │
             │  ⑤ Morphological Closing (5×5)   │
             └────────────────┬────────────────┘
                              │
             ┌────────────────▼────────────────┐
             │        feature_extraction.py      │
             │  Mode 19 Fitur (tanpa GLCM):      │
             │  • RGB mean & std  (6 fitur)      │
             │  • HSV mean & std  (6 fitur)      │
             │  • Hu Moments      (7 fitur)      │
             │                                   │
             │  Mode 39 Fitur (dengan GLCM):     │
             │  • GLCM 5×4 sudut  (20 fitur)    │
             │  • RGB + HSV + Hu  (19 fitur)    │
             │  → Information Gain → Top 14     │
             └────────────────┬────────────────┘
                              │
             ┌────────────────▼────────────────┐
             │             predict.py            │
             │  StandardScaler (normalisasi)     │
             │  Split 80:10:10 (stratified)      │
             │  Min. 9 sampel/kelas              │
             │                                   │
             │  [Algoritma Tunggal]              │
             │  LR1–LRn   → max_iter            │
             │  SVM1–SVMn → kernel              │
             │  DT1–DTn   → max_depth           │
             │  RF1–RFn   → n_estimators        │
             │  GB1–GBn   → n_est + lr          │
             │                                   │
             │  [Kombinasi Stacking]             │
             │  LRGB1  = LR(best) + GB(best)   │
             │  SVMGB1 = SVM(best) + GB(best)  │
             │  DTGB1  = DT(best) + GB(best)   │
             │  RFGB1  = RF(best) + GB(best)   │
             │                                   │
             │  1 terbaik per algo → TRAINED_   │
             └────────────────┬────────────────┘
                              │
             ┌────────────────▼────────────────┐
             │    OUTPUT: Renggang / Sedang / Padat │
             └──────────────────────────────────┘
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
Latih minimal satu model di halaman **🔬 Pemodelan** sampai muncul di sidebar.

### "Setiap kelas membutuhkan minimal 9 gambar"
Stratified split 80:10:10 membutuhkan minimal **9 gambar per kelas** (27 total).  
Tambah gambar lalu ulangi dari Langkah 1.

### Tombol Kombinasi masih 🔒 Belum siap
Latih algoritma baseline (LR/SVM/DT/RF) dan Gradient Boosting di tab Algoritma Tunggal terlebih dahulu. Tombol kombinasi baru aktif setelah keduanya ada.

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

### Hasil pelatihan/pengujian hilang setelah refresh browser
Data disimpan di **session state** (memori tab browser).  
Simpan screenshot atau export tabel sebelum menutup/refresh halaman.  
File model (`.joblib`) tetap tersimpan di disk meskipun browser di-refresh.

### Grafik tidak terbaca di Dark Mode
Semua grafik menggunakan background transparan dan CSS dark mode override otomatis.  
Pastikan menggunakan Streamlit versi terbaru (`streamlit >= 1.28`).

> 📖 Panduan lengkap langkah demi langkah ada di [PANDUAN_PENGGUNAAN.md](PANDUAN_PENGGUNAAN.md)

---

## Kontak & Informasi

**Peneliti:** Kholifah Dina  
**Institusi:** Universitas Telkom Purwokerto  
**Topik:** Klasifikasi Kepadatan Gulma menggunakan HSV Segmentation & Gradient Boosting + Stacking Ensemble

---

*Sistem ini dikembangkan untuk keperluan penelitian Tugas Akhir.*
