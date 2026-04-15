# 🌿 Weed Density Classification

Sistem klasifikasi kepadatan gulma berbasis *Machine Learning* menggunakan **Streamlit**.  
Proyek ini merupakan bagian dari Tugas Akhir/Skripsi di **Universitas Telkom Purwokerto**.

> **Pipeline:** Resize → Gaussian Blur → Segmentasi HSV → Morphological Closing →  
> Ekstraksi 14 Fitur Terbaik (*Information Gain*) → **Gradient Boosting Classifier**

---

## Daftar Isi

1. [Tentang Sistem](#tentang-sistem)
2. [Struktur Proyek](#struktur-proyek)
3. [Tech Stack](#tech-stack)
4. [Cara Menjalankan Lokal](#cara-menjalankan-lokal)
5. [Panduan Kolaborasi (Pull & Kontribusi)](#panduan-kolaborasi)
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

Sistem memiliki dua mode pengguna:

- **Admin** — melakukan training model, eksperimen hyperparameter, dan evaluasi
- **User** — mengupload foto dan mendapatkan hasil klasifikasi instan

---

## Struktur Proyek

```
weed-density-app/
│
├── requirements.txt                  # Dependensi Python (dengan version pinning)
├── runtime.txt                       # Versi Python yang digunakan (3.10)
├── .gitignore                        # File yang dikecualikan dari git
├── README.md                         # Dokumentasi ini
│
└── streamlit_app/
    ├── app.py                        # Aplikasi Streamlit utama (UI + routing)
    ├── predict.py                    # Pipeline training & inferensi model
    ├── preprocessing.py              # Preprocessing citra (resize, blur, HSV, morph)
    ├── feature_extraction.py         # Ekstraksi fitur GLCM, RGB, HSV, Hu Moments
    ├── Data_ekstraksi_Fitur_Gulma.csv  # Dataset fitur (2.097 sampel, 39 fitur)
    └── models/
        └── weed_metrics.joblib       # Metrik evaluasi model terlatih
        # weed_models.joblib          # File model (di-generate setelah training, tidak di-track git)
```

---

## Tech Stack

| Komponen | Library / Versi |
|----------|----------------|
| Web Framework | `streamlit >= 1.28` |
| Machine Learning | `scikit-learn >= 1.3` |
| Image Processing | `opencv-python-headless >= 4.8` |
| Image Features | `scikit-image >= 0.21` |
| Data Manipulation | `pandas >= 2.0`, `numpy >= 1.24` |
| Visualisasi | `plotly >= 5.17` |
| Model Persistence | `joblib >= 1.3` |
| Runtime | Python 3.10 |

---

## Cara Menjalankan Lokal

### Prasyarat

- **Python 3.10** sudah terpasang  
  Cek dengan: `python --version`
- **Git** sudah terpasang  
  Cek dengan: `git --version`

---

### 1. Clone Repositori

```bash
git clone https://github.com/kholifah-dina/Weed_Density_Classification.git
cd Weed_Density_Classification
```

---

### 2. Buat Virtual Environment

Sangat disarankan menggunakan virtual environment agar dependensi tidak bentrok dengan proyek lain.

**Windows (Command Prompt / PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

Setelah aktif, prompt terminal akan menampilkan `(venv)`.

---

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

Proses ini mengunduh semua library yang diperlukan. Pastikan koneksi internet aktif.  
Estimasi waktu: 2–5 menit (tergantung kecepatan internet).

---

### 4. Jalankan Aplikasi

```bash
cd streamlit_app
streamlit run app.py
```

Browser akan otomatis terbuka di `http://localhost:8501`.  
Jika tidak terbuka otomatis, buka manual di browser.

---

## Panduan Kolaborasi

Bagian ini ditujukan untuk **teman yang ingin berkontribusi** pada proyek ini.

### Pertama Kali (Clone & Setup)

```bash
# 1. Clone repositori
git clone https://github.com/kholifah-dina/Weed_Density_Classification.git
cd Weed_Density_Classification

# 2. Buat virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 3. Install dependensi
pip install -r requirements.txt

# 4. Jalankan aplikasi
cd streamlit_app
streamlit run app.py
```

---

### Alur Kerja Kolaborasi (Git Workflow)

Gunakan alur ini setiap kali ingin membuat perubahan:

```bash
# ── SEBELUM mulai bekerja ─────────────────────────────────────────────────────

# 1. Pastikan kamu di branch main dan ambil update terbaru
git checkout main
git pull origin main

# 2. Buat branch baru untuk fitur/perbaikan yang akan dikerjakan
#    Nama branch: deskriptif dan singkat, gunakan tanda-hubung
git checkout -b nama-fitur-kamu
#    Contoh: git checkout -b fix-preprocessing-threshold
#    Contoh: git checkout -b add-export-csv-feature

# ── SAAT bekerja ──────────────────────────────────────────────────────────────

# 3. Kerjakan perubahan di file yang diperlukan...

# 4. Cek file apa saja yang berubah
git status

# 5. Tambahkan file yang ingin di-commit (spesifik, jangan git add -A sembarangan)
git add streamlit_app/app.py
git add streamlit_app/preprocessing.py
# dst...

# 6. Commit dengan pesan yang jelas
git commit -m "fix: perbaiki threshold HSV untuk kondisi pencahayaan rendah"

# ── SETELAH selesai ───────────────────────────────────────────────────────────

# 7. Push branch kamu ke GitHub
git push origin nama-fitur-kamu

# 8. Buka GitHub → klik "Compare & pull request"
#    Tulis deskripsi perubahan yang kamu buat, lalu submit PR
```

---

### Mengambil Update dari Teman (Sync)

Jika teman kamu sudah push perubahan ke `main` dan kamu ingin mengambilnya:

```bash
# Pastikan semua perubahan lokalmu sudah di-commit atau di-stash
git stash                  # simpan perubahan sementara (jika ada)

git checkout main
git pull origin main       # ambil update terbaru

git stash pop              # kembalikan perubahan sementara (jika ada)
```

---

### Konvensi Pesan Commit

Gunakan format berikut agar riwayat git mudah dibaca:

| Prefix | Kapan digunakan |
|--------|----------------|
| `feat:` | Menambahkan fitur baru |
| `fix:` | Memperbaiki bug |
| `refactor:` | Mengubah struktur kode tanpa mengubah perilaku |
| `docs:` | Mengubah dokumentasi atau README |
| `style:` | Perubahan tampilan/UI |
| `data:` | Menambah atau mengubah dataset |

**Contoh pesan commit yang baik:**
```
feat: tambah ekspor hasil klasifikasi ke PDF
fix: perbaiki crash ketika gambar hitam putih diupload
refactor: pindahkan konstanta HSV ke config terpusat
docs: update README dengan panduan deployment
```

---

### File yang TIDAK Boleh Di-commit

File berikut sudah dikecualikan via `.gitignore`:

- `streamlit_app/models/weed_models.joblib` — file model biner (besar, di-generate ulang via Training)
- `__pycache__/` dan `*.pyc` — cache Python
- `venv/` — virtual environment lokal
- `.env` — variabel lingkungan sensitif

> **Catatan:** Setelah clone, kamu **tidak akan** menemukan file `weed_models.joblib`.  
> Ini normal. Jalankan Training di halaman Admin untuk men-generate model.

---

## Alur Penggunaan Aplikasi

### Sebagai Admin

1. Buka aplikasi → klik **"Masuk sebagai Admin"**
2. **Mode A — Upload Gambar Manual:**
   - Upload foto gulma untuk masing-masing kelas (Renggang, Sedang, Padat)
   - Maksimal 190 gambar per kelas
   - Format: JPG/JPEG
   - Klik **"Latih dari Gambar Upload"**
3. **Mode B — CSV Lokal (Disarankan untuk dataset besar):**
   - Masukkan path ke file CSV yang sudah berisi 39 kolom fitur + kolom `Class`
   - Klik **"Latih dari CSV"**
4. Setelah training, lihat tabel evaluasi, grafik perbandingan, dan confusion matrix
5. Buka **"Eksperimen Parameter"** untuk eksperimen hyperparameter

### Sebagai User

1. Buka aplikasi → klik **"Masuk sebagai User"**
2. Upload foto lahan/tanaman (JPG/JPEG)
3. Lihat hasil preprocessing (Tahap 1)
4. Lihat 14 nilai fitur yang diekstrak (Tahap 2)
5. Pilih model klasifikasi dari dropdown (Tahap 3)
6. Baca hasil prediksi: **Renggang / Sedang / Padat**

---

## Pipeline Machine Learning

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT: Foto Lahan                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │        preprocessing.py          │
              │  1. Resize → 224 × 224 px        │
              │  2. Gaussian Blur (kernel 5×5)   │
              │  3. Konversi BGR → HSV            │
              │  4. Threshold HSV (H:25-75°,     │
              │     S:40-255, V:50-255)           │
              │  5. Morphological Closing (5×5)  │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │      feature_extraction.py       │
              │  Ekstrak 39 Fitur:               │
              │  • GLCM (5 prop × 4 sudut = 20)  │
              │  • RGB mean & std (6)            │
              │  • HSV mean & std (6)            │
              │  • Hu Moments (7)                │
              │                                  │
              │  → Seleksi Top-14 (Info. Gain)   │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │            predict.py            │
              │  StandardScaler → Normalisasi    │
              │  5 Model Classifier:             │
              │  • Logistic Regression           │
              │  • SVM (RBF, C=5, γ=0.01)        │
              │  • Decision Tree (depth=3)       │
              │  • Random Forest (500 trees)     │
              │  • Gradient Boosting* (300 trees)│
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │    OUTPUT: Renggang/Sedang/Padat │
              └─────────────────────────────────┘

  * Model utama penelitian
```

### 14 Fitur Terpilih (Information Gain)

| No | Nama Fitur | Kategori |
|----|-----------|---------|
| 1 | homogeneity_90deg | GLCM |
| 2 | homogeneity_45deg | GLCM |
| 3 | energy_45deg | GLCM |
| 4 | energy_135deg | GLCM |
| 5 | energy_90deg | GLCM |
| 6 | energy_0deg | GLCM |
| 7 | homogeneity_0deg | GLCM |
| 8 | homogeneity_135deg | GLCM |
| 9 | dissimilarity_90deg | GLCM |
| 10 | dissimilarity_135deg | GLCM |
| 11 | dissimilarity_45deg | GLCM |
| 12 | dissimilarity_0deg | GLCM |
| 13 | HuMoment_6 | Hu Moments |
| 14 | G_mean | Warna RGB |

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

### Model belum tersedia / Error "Model belum tersedia"

File model (`weed_models.joblib`) tidak di-track di git karena ukurannya besar.  
**Solusi:** Masuk sebagai Admin → Training Model → latih ulang model.

---

### Error saat install dependensi: `ERROR: Could not build wheels for opencv`

Pastikan kamu menginstall versi **headless**:
```bash
pip install opencv-python-headless
```
Jangan install `opencv-python` (versi dengan GUI) — akan konflik di server.

---

### Port 8501 sudah dipakai

```bash
streamlit run app.py --server.port 8502
```

---

### Streamlit tidak terbuka otomatis di browser

Buka manual di browser: `http://localhost:8501`

---

### `ModuleNotFoundError` saat menjalankan app

Pastikan virtual environment sudah aktif dan dependensi sudah diinstall:
```bash
# Aktifkan venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

# Install ulang
pip install -r requirements.txt
```

---

### CSV tidak diterima: "Jumlah kolom fitur harus 39"

Pastikan format CSV identik dengan `Data_ekstraksi_Fitur_Gulma.csv`:
- 39 kolom fitur (GLCM + RGB/HSV + Hu Moments)
- 1 kolom `Class` berisi label: `Renggang`, `Sedang`, atau `Padat`
- Semua kolom fitur bertipe numerik (float/int)

---

## Kontak & Informasi

**Peneliti:** Kholifah Dina  
**Institusi:** Universitas Telkom Purwokerto  
**Topik:** Klasifikasi Kepadatan Gulma menggunakan HSV Segmentation & Gradient Boosting  

---

*Sistem ini dikembangkan untuk keperluan penelitian Tugas Akhir.*
