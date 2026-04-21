# 📖 Panduan Penggunaan Sistem Klasifikasi Kepadatan Gulma

**Sistem:** Weed Density Classification System  
**Versi:** 5.0 (Langkah 4: Multiselect + Tab Kombinasi Stacking · Implementasi: Upload Per Kelas 50/kelas · Pivot Table)  
**Pemilik:** Kholifah Dina — Tugas Akhir, Universitas Telkom Purwokerto  

---

## Daftar Isi

1. [Persyaratan Sistem](#1-persyaratan-sistem)
2. [Cara Menjalankan Aplikasi](#2-cara-menjalankan-aplikasi)
3. [Navigasi Sistem](#3-navigasi-sistem)
4. [Halaman 1 — Pemodelan](#4-halaman-1--pemodelan)
   - [Langkah 1 — Upload Dataset](#-langkah-1--upload-dataset)
   - [Langkah 2 — Preprocessing](#%EF%B8%8F-langkah-2--visualisasi-preprocessing)
   - [Langkah 3 — Ekstraksi Fitur](#-langkah-3--ekstraksi-fitur)
   - [Langkah 4 — Pelatihan Dinamis](#-langkah-4--pelatihan-dinamis)
     - [Tab A: Algoritma Tunggal](#tab-a--algoritma-tunggal)
     - [Tab B: Kombinasi Baseline + GB (Stacking)](#tab-b--kombinasi-baseline--gb-stacking)
5. [Halaman 2 — Implementasi](#5-halaman-2--implementasi)
6. [Alur Penggunaan yang Disarankan](#6-alur-penggunaan-yang-disarankan)
7. [Activity Diagram — Panduan untuk Pembuat Diagram](#7-activity-diagram--panduan-untuk-pembuat-diagram)
8. [Penjelasan Istilah Teknis](#8-penjelasan-istilah-teknis)
9. [Troubleshooting (Masalah & Solusi)](#9-troubleshooting-masalah--solusi)
10. [Struktur File Sistem](#10-struktur-file-sistem)

---

## 1. Persyaratan Sistem

### Software
| Komponen | Versi Minimum | Catatan |
|----------|--------------|---------|
| Python   | 3.10         | Wajib versi 3.10 |
| pip      | 23+          | Untuk install library |

### Library Python (otomatis terinstall via requirements.txt)
| Library | Versi |
|---------|-------|
| streamlit | ≥ 1.28 |
| scikit-learn | ≥ 1.3 |
| opencv-python-headless | ≥ 4.8 |
| scikit-image | ≥ 0.21 |
| pandas | ≥ 2.0 |
| numpy | ≥ 1.24 |
| plotly | ≥ 5.17 |
| joblib | ≥ 1.3 |

### Hardware (Minimum)
- RAM: 4 GB (disarankan 8 GB untuk training dataset besar)
- Penyimpanan: 500 MB kosong
- Koneksi internet (hanya untuk load font Google saat pertama kali)

> 🌙 Sistem mendukung **Light Mode** dan **Dark Mode** secara otomatis.

---

## 2. Cara Menjalankan Aplikasi

### Langkah 1 — Clone Repository (hanya pertama kali)
```bash
git clone https://github.com/kholifah-dina/Weed_Density_Classification.git
cd Weed_Density_Classification
```

### Langkah 2 — Buat Virtual Environment (disarankan)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Langkah 3 — Install Library
```bash
pip install -r requirements.txt
```

### Langkah 4 — Jalankan Aplikasi
```bash
cd streamlit_app
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser: **http://localhost:8501**

> ⚠️ **Catatan Penting:** File model (`TRAINED_LR1.joblib`, dll.) **tidak disertakan** di GitHub. Setelah clone, Anda **harus melakukan training** terlebih dahulu melalui halaman **🔬 Pemodelan** sebelum bisa menggunakan halaman **🎯 Implementasi**.

---

## 3. Navigasi Sistem

Sistem menggunakan **sidebar di sebelah kiri** dengan **2 halaman**:

```
🌿 Kepadatan Gulma
│
├── Pilih Halaman:
│   ○ 🔬 Pemodelan      ← Training model langkah demi langkah
│   ○ 🎯 Implementasi   ← Evaluasi model dengan gambar uji
│
└── 💾 Top 5 Model Tersimpan:
    🏆 LR1  🏆 SVM1  🏆 GB1  🏆 LRGB1  (contoh)
```

Sidebar juga menampilkan **model terbaik per algoritma** (1 model terbaik dari setiap jenis algoritma yang sudah dilatih).

---

## 4. Halaman 1 — Pemodelan

### Tujuan
Melatih model klasifikasi gulma dari dataset gambar secara bertahap. Setiap percobaan training diberi **ID unik** dan dicatat di **tabel rekap real-time**. Model terbaik per algoritma disimpan otomatis untuk digunakan di halaman Implementasi.

---

### 📁 Langkah 1 — Upload Dataset

1. Buka halaman **🔬 Pemodelan** dari sidebar
2. Perhatikan info box: `📎 Format JPG/JPEG · Maks. 190 gambar/kelas · Min. 9 gambar/kelas`
3. Upload gambar untuk **3 kelas** menggunakan area drag-and-drop:
   - 🟢 **Renggang** — gambar lahan dengan gulma jarang
   - 🟡 **Sedang** — gambar lahan dengan gulma sedang
   - 🔴 **Padat** — gambar lahan dengan gulma padat
4. Syarat upload:
   - Format: **JPG atau JPEG** saja (bukan PNG)
   - **Minimal 9 gambar per kelas** (wajib untuk stratified split 80:10:10)
   - **Maksimal 190 gambar per kelas**
5. Tabel ringkasan jumlah gambar per kelas otomatis muncul
6. Klik **"Lanjut ke Preprocessing →"**

> 💡 **Tips:** Disarankan minimal 50 gambar/kelas untuk hasil yang lebih akurat.

---

### 🖼️ Langkah 2 — Visualisasi Preprocessing

Sistem menampilkan **4 tahap preprocessing** otomatis dari 1 gambar contoh kelas Padat:

| Tahap | Nama | Penjelasan |
|-------|------|-----------|
| ① | Resize 224×224 | Gambar diubah ukuran menjadi 224×224 piksel sebagai standar input |
| ② | Gaussian Blur (5×5) | Menghaluskan gambar dan meredam noise agar segmentasi lebih akurat |
| ③ | HSV Thresholding | Mengisolasi piksel hijau (H:25–75°, S:40–255, V:50–255) sebagai area gulma |
| ④ | Morphological Closing | Mengisi celah kecil pada mask agar area gulma terdeteksi utuh |

Baca penjelasan setiap tahap, lalu klik **"Lanjut ke Ekstraksi Fitur →"**

---

### 🧬 Langkah 3 — Ekstraksi Fitur

Pilih salah satu mode fitur:

| Mode | Jumlah Fitur | Seleksi | Input ke Model |
|------|-------------|---------|---------------|
| **19 Fitur** | RGB (6) + HSV (6) + Hu Moments (7) | Tidak ada | 19 fitur langsung |
| **39 Fitur** | GLCM (20) + RGB + HSV + Hu (19) | Information Gain | 14 fitur terbaik |

**Detail Mode 19 Fitur:**
- RGB mean & std (6 fitur): rata-rata dan variasi kanal R, G, B
- HSV mean & std (6 fitur): rata-rata dan variasi Hue, Saturation, Value
- Hu Moments (7 fitur): deskriptor bentuk invariant terhadap rotasi dan skala

**Detail Mode 39 Fitur:**
- GLCM: 5 properti tekstur × 4 sudut (0°, 45°, 90°, 135°) = 20 fitur
- Ditambah 19 fitur warna dan bentuk = total 39 fitur
- **Information Gain** otomatis memilih **14 fitur terbaik** dari 39 fitur

Klik **"Jalankan Ekstraksi Fitur →"** (atau "Jalankan Ekstraksi & Information Gain →" untuk mode 39).

**Output setelah ekstraksi berhasil:**
- **Info Pembagian Data** — chip berwarna dengan jumlah sampel Train / Validasi / Test
- **Tabel Contoh Hasil Ekstraksi** — 3 sampel per kelas (9 baris total)
- **Mode 39** — Bar chart IG, tabel ranking 39 fitur, tabel 14 fitur terpilih
- **Mode 19** — Tabel daftar 19 fitur dengan kelompok dan deskripsi

---

### 🤖 Langkah 4 — Pelatihan Dinamis

Langkah 4 memiliki **2 tab**:

```
[🔵 Algoritma Tunggal]  [🔗 Kombinasi Baseline + GB]
```

---

#### Tab A — Algoritma Tunggal

Melatih satu algoritma dengan satu atau banyak kombinasi parameter sekaligus.

**Cara Penggunaan:**

1. **Pilih Algoritma** dari dropdown:

| Algoritma | Kode | Karakteristik |
|-----------|------|--------------|
| Logistic Regression | LR | Model linear sederhana, sangat cepat |
| SVM | SVM | Efektif untuk data high-dimensional |
| Decision Tree | DT | Mudah diinterpretasi, cepat dilatih |
| Random Forest | RF | Ensemble pohon, robust terhadap overfitting |
| Gradient Boosting | GB | Ensemble bertahap, akurasi tinggi |

2. **Pilih satu atau banyak nilai parameter** (multiselect — bisa pilih lebih dari satu):

| Algoritma | Parameter | Nilai Tersedia |
|-----------|-----------|---------------|
| Logistic Regression | `max_iter` | 100, 300, 500, 700, 1000 |
| SVM | `kernel` | linear, rbf, poly |
| Decision Tree | `max_depth` | 3, 5, 7, 9, 11 |
| Random Forest | `n_estimators` | 100, 200, 300, 400, 500 |
| Gradient Boosting | `n_estimators` | 100, 200, 300 |
| Gradient Boosting | `learning_rate` | 0.01, 0.1, 1 |

> **Catatan GB:** Karena GB memiliki 2 parameter (`n_estimators` dan `learning_rate`), semua kombinasi yang dipilih akan dilatih otomatis secara cross-product. Misal memilih 3 nilai `n_estimators` dan 3 nilai `learning_rate` = **9 model** dilatih sekaligus.

3. Pratinjau **daftar kombinasi** dan **ID yang akan dihasilkan** muncul di bawah form.

4. Klik **"🚀 Latih [N] Kombinasi"** — sistem melatih semua kombinasi secara berurutan.

5. Setiap model selesai: notifikasi sukses muncul dengan ID model (contoh: **LR1**, **SVM2**, **GB3**).

**ID Model Sekuensial:**
- Setiap kombinasi mendapat ID unik berurutan per algoritma
- LR: LR1, LR2, LR3, ...
- SVM: SVM1, SVM2, ...
- GB: GB1, GB2, GB3, ... (urutan dari cross-product n_estimators × learning_rate)

**Tabel Rekap Real-time:**

| Model ID | Algoritma | Parameter | Accuracy | Precision | Recall | F1-Score | Waktu (s) | Status |
|----------|-----------|-----------|----------|-----------|--------|----------|-----------|--------|
| LR1 | Logistic Regression | max_iter=100 | 0.8500 | 0.8612 | 0.8500 | 0.8421 | 1.23 | 🏆 Best LR |
| LR2 | Logistic Regression | max_iter=300 | 0.8800 | 0.8901 | 0.8800 | 0.8750 | 1.45 | 🏆 Best LR |
| SVM1 | SVM | kernel=rbf | 0.9000 | 0.9123 | 0.9000 | 0.8967 | 2.30 | 🏆 Best SVM |

- **Kolom Status** menunjukkan apakah model ini adalah **best per algoritma** saat ini
- Sistem otomatis menyimpan **1 model terbaik per algoritma** berdasarkan Accuracy

**Auto-Save Best Per Algo:**
- Setiap selesai training, sistem membandingkan model baru dengan model terbaik sebelumnya untuk algoritma yang sama
- Jika lebih baik, model baru menggantikan yang lama (file `.joblib` lama dihapus)
- Sidebar menampilkan daftar model terbaik yang aktif per algoritma

**Tombol di bawah:**
- **"🔄 Mulai Ulang dari Langkah 1"** — reset langkah 1-4, riwayat pelatihan tetap
- **"🗑️ Hapus Semua Riwayat Pelatihan"** — hapus semua riwayat + semua file model di disk

---

#### Tab B — Kombinasi Baseline + GB (Stacking)

Menggabungkan algoritma baseline terbaik dengan Gradient Boosting terbaik menggunakan teknik **Stacking Ensemble**.

**Konsep Stacking:**
```
Input Fitur
    ↓
[Base Estimator: LR / SVM / DT / RF]  ← parameter terbaik dari Tab A
    ↓  (prediksi probabilitas, cv=5)
[Meta Learner: Gradient Boosting]     ← parameter terbaik GB dari Tab A
    ↓
Prediksi Akhir
```

- **Base Estimator**: salah satu dari LR, SVM, DT, atau RF (menggunakan parameter terbaik yang sudah ditemukan di Tab A)
- **Meta Learner**: Gradient Boosting (menggunakan parameter terbaik GB dari Tab A)
- **Cross-validation**: 5-fold pada data training agar meta-learner tidak overfitting
- **passthrough=True**: fitur asli juga ikut diteruskan ke meta-learner

**Prasyarat sebelum bisa melatih kombinasi:**
- Model baseline terbaik untuk algoritma tersebut sudah ada (misal: best LR sudah ada untuk melatih LR+GB)
- Model GB terbaik sudah ada

**4 Kombinasi yang Tersedia:**

| ID Tetap | Nama Kombinasi | Base Estimator |
|----------|---------------|----------------|
| **LRGB1** | LR + GB (Stacking) | Logistic Regression terbaik |
| **SVMGB1** | SVM + GB (Stacking) | SVM terbaik |
| **DTGB1** | DT + GB (Stacking) | Decision Tree terbaik |
| **RFGB1** | RF + GB (Stacking) | Random Forest terbaik |

> **Penting — ID Tetap:** Berbeda dengan Tab A yang menggunakan ID berurutan (LR1, LR2, ...), setiap kombinasi di Tab B hanya memiliki **1 ID yang tetap** (LRGB1, SVMGB1, DTGB1, RFGB1). Setelah dilatih, kombinasi tersebut **tidak bisa dilatih ulang** dan tombol berubah menjadi tampilan hasil.

**Cara Penggunaan:**

1. Pastikan sudah melatih model baseline dan GB di Tab A
2. Buka **Tab B: Kombinasi Baseline + GB**
3. Setiap kombinasi ditampilkan sebagai **kartu**:
   - **Tombol "🚀 Latih LRGB1"** — muncul jika belum dilatih dan prasyarat terpenuhi
   - **🔒 Belum siap** — muncul jika prasyarat belum terpenuhi (base atau GB belum ada)
   - **Badge hasil** — muncul jika sudah pernah dilatih (tidak bisa diulang)
4. Klik tombol untuk melatih satu kombinasi
5. Hasil ditampilkan: Accuracy, Precision, Recall, F1-Score, Waktu

---

## 5. Halaman 2 — Implementasi

### Tujuan
Mengevaluasi model terpilih menggunakan gambar uji nyata. Label kelas sudah ditentukan otomatis dari zona upload per kelas — tidak perlu input manual per gambar.

---

### 🤖 Pilih Model

1. Buka halaman **🎯 Implementasi** dari sidebar
2. Dropdown **"Model (dari Top 5 hasil pemodelan)"** menampilkan model-model terbaik yang tersimpan
3. Pilih model yang ingin diuji (contoh: LR1, SVM1, LRGB1)
4. Info card model muncul otomatis: nama algoritma, parameter, mode fitur, metrik training

> ⚠️ Jika dropdown kosong: kembali ke **🔬 Pemodelan** dan latih minimal satu model.

---

### 📤 Upload Gambar Uji Per Kelas

Berbeda dari versi sebelumnya, upload gambar kini menggunakan **3 zona terpisah per kelas** — label aktual sudah otomatis sesuai zona upload, tanpa input manual.

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  🟢 Renggang    │  │  🟡 Sedang      │  │  🔴 Padat       │
│  Upload JPG     │  │  Upload JPG     │  │  Upload JPG     │
│  (maks. 50)     │  │  (maks. 50)     │  │  (maks. 50)     │
│                 │  │                 │  │                 │
│  [5 file ✓]    │  │  [3 file ✓]    │  │  [0 file]       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

- **Maks. 50 gambar per kelas** (total maks. 150 gambar)
- Jika upload lebih dari 50 untuk satu kelas, hanya 50 pertama yang diproses
- Tidak perlu mengisi label aktual — label = nama zona upload

**Ringkasan upload** muncul di bawah zona:

| Kelas | Jumlah Gambar |
|-------|--------------|
| 🟢 Renggang | 50 |
| 🟡 Sedang | 48 |
| 🔴 Padat | 50 |
| **Total** | **148** |

---

### 🔍 Uji Model & Lihat Hasil

1. Klik **"🔍 Uji [N] Gambar dengan [ID Model]"**
2. **Progress bar** dan status teks menampilkan proses real-time:
   ```
   Mengklasifikasi gambar 23/148: img_023.jpg (Sedang)
   [========================>         ] 15%
   ```
3. Setiap gambar diproses: preprocessing → ekstraksi fitur → prediksi

---

### 📊 Tabel Hasil (Pivot Table)

Hasil ditampilkan sebagai **pivot table** — setiap model diuji menjadi satu kelompok kolom:

```
              │        LR1          │        LRGB1        │
Data ke- │ Gambar  │ Aktual │ Prediksi │ Hasil │ Prediksi │ Hasil │
─────────┼─────────┼────────┼──────────┼───────┼──────────┼───────┤
    1    │ img1    │ R      │ R        │ TRUE  │ R        │ TRUE  │
    2    │ img2    │ S      │ R        │ FALSE │ S        │ TRUE  │
    3    │ img3    │ P      │ P        │ TRUE  │ P        │ TRUE  │
   ...   │ ...     │ ...    │ ...      │ ...   │ ...      │ ...   │
```

- **TRUE** (teks hijau bold) = prediksi sesuai label aktual
- **FALSE** (teks merah bold) = prediksi tidak sesuai
- Setiap model membentuk grup kolom tersendiri dengan header ID model

**Baris Summary di bawah tabel pivot:**

| Summary | LR1 | LRGB1 |
|---------|-----|-------|
| Jumlah TRUE | 120 | 135 |
| Jumlah FALSE | 28 | 13 |
| Total Gambar | 148 | 148 |
| Presisinya | 81.1% | 91.2% |
| 🟢 Benar Renggang | 47/50 | 49/50 |
| 🟡 Benar Sedang | 40/48 | 44/48 |
| 🔴 Benar Padat | 33/50 | 42/50 |

---

### 📋 Rekap Multi-Model

Setiap kali Anda **mengganti model dan menguji lagi**, baris summary ditambahkan ke tabel pivot. Semua pengujian tampil dalam satu tabel untuk perbandingan langsung.

**Tombol:** "🗑️ Hapus Riwayat Pengujian" — menghapus semua hasil uji.

---

### 💬 Narasi Kesimpulan Otomatis

| Rata-rata Presisi | Kesimpulan |
|-------------------|-----------|
| ≥ 90% | "Model menunjukkan keandalan **sangat tinggi**..." |
| 70–89% | "Model menunjukkan keandalan **cukup baik**..." |
| < 70% | "Diperlukan evaluasi lebih lanjut..." |

---

## 6. Alur Penggunaan yang Disarankan

### Untuk Pertama Kali (Setelah Clone)

```
1. Halaman 🔬 Pemodelan
   → Upload dataset (min. 9 gambar/kelas, disarankan ≥ 50/kelas)
   → Pahami visualisasi preprocessing (Langkah 2)
   → Pilih mode 39 Fitur → Jalankan Ekstraksi → baca grafik IG
   → Langkah 4 Tab A — latih SEMUA variasi parameter sekaligus:
       LR: pilih semua max_iter [100,300,500,700,1000]   → LR1–LR5 (1 klik)
       SVM: pilih semua kernel [linear,rbf,poly]          → SVM1–SVM3 (1 klik)
       DT: pilih semua max_depth [3,5,7,9,11]            → DT1–DT5 (1 klik)
       RF: pilih semua n_estimators [100..500]           → RF1–RF5 (1 klik)
       GB: pilih semua n_est [100,200,300] x lr [0.01,0.1,1] → GB1–GB9 (1 klik)
   → Langkah 4 Tab B — latih kombinasi stacking:
       LRGB1: LR terbaik + GB terbaik
       SVMGB1: SVM terbaik + GB terbaik
       DTGB1: DT terbaik + GB terbaik
       RFGB1: RF terbaik + GB terbaik
   → Catat ID model dengan akurasi tertinggi

2. Halaman 🎯 Implementasi
   → Pilih model dari dropdown (misal: LRGB1 dengan akurasi tertinggi)
   → Upload gambar uji: 50 Renggang + 50 Sedang + 50 Padat
   → Klik Uji Model → pantau progress bar
   → Catat tabel pivot + breakdown per kelas
   → Ganti model, uji lagi → pivot table bertambah kolom
   → Screenshot tabel untuk bahan laporan skripsi
```

---

## 7. Activity Diagram — Panduan untuk Pembuat Diagram

Bagian ini ditulis khusus untuk memudahkan pembuatan **Activity Diagram UML** pada laporan skripsi. Sistem memiliki **2 alur utama** yang masing-masing dapat dibuat sebagai diagram terpisah.

---

### Diagram 1 — Alur Pemodelan (Training)

```
[START]
    │
    ▼
[Upload Gambar Dataset (JPG/JPEG per 3 kelas)]
    │
    ├─── [Validasi: jumlah gambar per kelas ≥ 9?] ──── TIDAK ──▶ [Tampil Pesan Error] ──▶ (kembali upload)
    │                   │ YA
    ▼                   ▼
[Tampil Ringkasan Dataset]
    │
    ▼
[Lanjut ke Langkah 2: Visualisasi Preprocessing]
    │
    ▼
[Resize gambar contoh ke 224×224]
    │
    ▼
[Gaussian Blur (5×5)]
    │
    ▼
[HSV Thresholding (deteksi warna hijau gulma)]
    │
    ▼
[Morphological Closing (isi celah pada mask)]
    │
    ▼
[Tampil 4 tahap preprocessing secara visual]
    │
    ▼
[Lanjut ke Langkah 3: Pilih Mode Fitur]
    │
    ├─── [Mode 39 Fitur?] ─── YA ──▶ [Ekstraksi 39 Fitur (GLCM + RGB + HSV + Hu)]
    │                                          │
    │                                  [Information Gain → pilih 14 fitur terbaik]
    │                                          │
    ├─── [Mode 19 Fitur?] ─── YA ──▶ [Ekstraksi 19 Fitur (RGB + HSV + Hu)]
    │
    ▼
[StandardScaler (normalisasi z-score)]
    │
    ▼
[Stratified Split 80:10:10 (Train / Validasi / Test)]
    │
    ▼
[Tampil Tabel Fitur + Grafik Information Gain (jika mode 39)]
    │
    ▼
[Lanjut ke Langkah 4: Pelatihan Dinamis]
    │
    ├─── [Pilih Tab A: Algoritma Tunggal]
    │         │
    │         ▼
    │    [Pilih Algoritma (LR / SVM / DT / RF / GB)]
    │         │
    │         ▼
    │    [Pilih satu atau banyak nilai parameter (multiselect)]
    │         │
    │         ▼
    │    [Klik Latih → Iterasi setiap kombinasi parameter]
    │         │
    │         ▼
    │    [Training model dengan kombinasi parameter saat ini]
    │         │
    │         ▼
    │    [Evaluasi pada data Test → Accuracy, Precision, Recall, F1]
    │         │
    │         ▼
    │    [Simpan sebagai TRAINED_{ID}.joblib]
    │         │
    │         ├─── [Accuracy lebih baik dari best sebelumnya?]
    │         │         │ YA
    │         │         ▼
    │         │    [Hapus file lama → Simpan sebagai Best Per Algo]
    │         │         │ TIDAK
    │         │         ▼
    │         │    [Hapus file model yang baru (bukan best)]
    │         │
    │         ▼
    │    [Update Tabel Rekap + Sidebar]
    │         │
    │         ├─── [Masih ada kombinasi parameter tersisa?] ── YA ──▶ (ulangi training)
    │         │                   │ TIDAK
    │         ▼                   ▼
    │    [Selesai — semua kombinasi terlatih]
    │
    ├─── [Pilih Tab B: Kombinasi Stacking]
    │         │
    │         ├─── [Best baseline algo & Best GB sudah ada?] ── TIDAK ──▶ [Tampil 🔒 Belum siap]
    │         │                   │ YA
    │         ▼                   ▼
    │    [Klik Latih Kombinasi (misal: LRGB1)]
    │         │
    │         ▼
    │    [Bangun StackingClassifier:
    │     - Base: estimator algo terbaik + param terbaik
    │     - Meta: GradientBoosting terbaik
    │     - cv=5, passthrough=True]
    │         │
    │         ▼
    │    [Training StackingClassifier]
    │         │
    │         ▼
    │    [Evaluasi pada data Test]
    │         │
    │         ▼
    │    [Simpan sebagai TRAINED_LRGB1.joblib (ID tetap)]
    │         │
    │         ▼
    │    [Tombol berubah menjadi badge hasil — tidak bisa dilatih ulang]
    │
    ▼
[END]
```

---

### Diagram 2 — Alur Implementasi (Pengujian)

```
[START]
    │
    ▼
[Buka Halaman Implementasi]
    │
    ▼
[Pilih Model dari Dropdown (model tersimpan)]
    │
    ├─── [Dropdown kosong?] ── YA ──▶ [Tampil pesan: kembali ke Pemodelan] ──▶ [END]
    │              │ TIDAK
    ▼              ▼
[Tampil Info Card: algoritma, parameter, metrik model]
    │
    ▼
[Upload Gambar Uji per Kelas (maks. 50/kelas × 3 kelas)]
    │
    ├─── [Tidak ada gambar yang diupload?] ── YA ──▶ [Tampil panduan upload] ──▶ (kembali)
    │              │ TIDAK
    ▼              ▼
[Tampil Ringkasan Upload per Kelas]
    │
    ▼
[Klik "Uji Model"]
    │
    ▼
[Iterasi setiap gambar (dengan progress bar)]
    │
    ▼
[Preprocessing gambar:
 Resize → Blur → HSV → Closing → Segmented]
    │
    ▼
[Ekstraksi fitur (sesuai mode model yang dipilih: 19 atau 39)]
    │
    ▼
[Normalisasi dengan Scaler tersimpan]
    │
    ▼
[Prediksi kelas dengan model tersimpan]
    │
    ▼
[Bandingkan prediksi vs label aktual (dari zona upload)]
    │
    ├─── [Prediksi == Aktual?] ── YA ──▶ [Hasil: TRUE]
    │                           TIDAK ──▶ [Hasil: FALSE]
    │
    ▼
[Update progress bar]
    │
    ├─── [Masih ada gambar tersisa?] ── YA ──▶ (ulangi dari preprocessing)
    │              │ TIDAK
    ▼              ▼
[Hitung: TRUE count, FALSE count, Presisi %, class breakdown per kelas]
    │
    ▼
[Tampil Tabel Pivot:
 - Kolom: Gambar, Aktual, Prediksi Model X, Hasil Model X
 - Baris: setiap gambar uji + baris summary]
    │
    ▼
[Tampil Summary: Benar Renggang / Sedang / Padat]
    │
    ▼
[Tampil Narasi Kesimpulan Otomatis]
    │
    ├─── [Ganti model & uji lagi?] ── YA ──▶ (kembali ke Pilih Model)
    │              │ TIDAK             (kolom baru ditambah ke pivot table)
    ▼              ▼
[END]
```

---

### Node Reference untuk Activity Diagram

**Action Nodes (aktivitas/aksi):**
1. Upload Gambar Dataset (per 3 kelas)
2. Resize Gambar ke 224×224
3. Gaussian Blur (5×5)
4. HSV Thresholding
5. Morphological Closing
6. Ekstraksi 19 Fitur (RGB + HSV + Hu Moments)
7. Ekstraksi 39 Fitur (GLCM + RGB + HSV + Hu Moments)
8. Information Gain — Seleksi 14 Fitur Terbaik
9. StandardScaler — Normalisasi Data
10. Stratified Split 80:10:10
11. Training Model dengan Parameter Terpilih
12. Evaluasi pada Data Test
13. Simpan Model (TRAINED_{ID}.joblib)
14. Update Tabel Rekap & Sidebar
15. Bangun StackingClassifier (base + meta-learner)
16. Upload Gambar Uji per Kelas (3 zona, maks. 50/kelas)
17. Preprocessing Gambar Uji
18. Ekstraksi Fitur Gambar Uji
19. Prediksi Kelas dengan Model Tersimpan
20. Tampil Tabel Pivot + Summary
21. Tampil Narasi Kesimpulan

**Decision Nodes (percabangan ya/tidak):**
1. Apakah jumlah gambar per kelas ≥ 9?
2. Apakah mode fitur = 39?
3. Apakah akurasi model baru lebih baik dari best sebelumnya?
4. Apakah masih ada kombinasi parameter tersisa?
5. Apakah prasyarat stacking terpenuhi (best algo & best GB ada)?
6. Apakah model sudah pernah dilatih (ID tetap sudah ada)?
7. Apakah dropdown model kosong?
8. Apakah ada gambar yang diupload?
9. Apakah prediksi == label aktual?
10. Apakah masih ada gambar tersisa untuk diproses?
11. Apakah ingin ganti model dan uji lagi?

---

## 8. Penjelasan Istilah Teknis

| Istilah | Penjelasan Sederhana |
|---------|---------------------|
| **Preprocessing** | Serangkaian langkah pengolahan gambar sebelum fitur diekstrak |
| **Gaussian Blur** | Menghaluskan gambar untuk mengurangi gangguan/noise |
| **HSV** | Model warna berbasis Hue (warna), Saturation (kepekatan), Value (kecerahan) |
| **Thresholding** | Memilah piksel berdasarkan rentang nilai tertentu |
| **Segmentasi** | Proses memisahkan area gulma dari latar belakang |
| **Morphological Closing** | Mengisi lubang kecil pada area yang tersegmentasi |
| **GLCM** | Gray-Level Co-occurrence Matrix — mengukur tekstur gambar |
| **Fitur** | Nilai numerik yang merepresentasikan karakteristik satu gambar |
| **Information Gain** | Mengukur seberapa besar fitur membantu memisahkan kelas |
| **SelectKBest** | Memilih K fitur dengan skor IG tertinggi |
| **StandardScaler** | Normalisasi data ke skala z-score |
| **Stratified Split** | Pembagian data yang mempertahankan proporsi tiap kelas |
| **ID Model Sekuensial** | Penomoran unik per algoritma: LR1, LR2, SVM1, GB3, dst. |
| **Best Per Algo** | 1 model terbaik per jenis algoritma yang tersimpan otomatis |
| **Ground Truth** | Label aktual/kelas nyata suatu gambar |
| **Presisi (Implementasi)** | Jumlah prediksi benar / total gambar uji × 100% |
| **Accuracy** | Persentase prediksi benar dari total prediksi (data test training) |
| **Precision** | Dari yang diprediksi kelas X, berapa persen yang benar-benar kelas X |
| **Recall** | Dari yang sebenarnya kelas X, berapa persen yang berhasil terdeteksi |
| **F1-Score** | Rata-rata harmonis Precision dan Recall |
| **Confusion Matrix** | Tabel yang menunjukkan prediksi benar vs salah per kelas |
| **Overfitting** | Model terlalu hafal data training, performa buruk di data baru |
| **Stacking Ensemble** | Menggabungkan dua model: base estimator + meta-learner |
| **Base Estimator** | Model pertama dalam stacking (LR/SVM/DT/RF) |
| **Meta Learner** | Model kedua yang belajar dari output base estimator (GB) |
| **Gradient Boosting** | Ensemble bertahap — tiap pohon memperbaiki error sebelumnya |
| **Random Forest** | Ensemble pohon independen yang dipilih secara acak |
| **n_estimators** | Jumlah pohon dalam model ensemble |
| **learning_rate** | Seberapa besar kontribusi tiap pohon baru dalam Gradient Boosting |
| **max_depth** | Kedalaman maksimum sebuah pohon keputusan |
| **max_iter** | Jumlah iterasi maksimum dalam Logistic Regression |
| **kernel** | Fungsi transformasi ruang fitur pada SVM (linear/rbf/poly) |
| **Hu Moments** | 7 deskriptor bentuk objek invariant terhadap rotasi, skala, translasi |
| **Multiselect** | Komponen UI yang memungkinkan memilih banyak nilai sekaligus |
| **Pivot Table** | Tabel yang mengorganisasi data dalam format baris × kolom terstruktur |

---

## 9. Troubleshooting (Masalah & Solusi)

### ❌ Dropdown model kosong di halaman Implementasi
**Penyebab:** Belum ada model yang tersimpan.  
**Solusi:** Buka **🔬 Pemodelan** → latih minimal satu model di Langkah 4 → kembali ke Implementasi.

---

### ❌ "Setiap kelas membutuhkan minimal 9 gambar"
**Penyebab:** Jumlah gambar kurang dari 9 per kelas.  
**Solusi:** Tambah gambar hingga **≥ 9 per kelas** di semua kelas, lalu ulangi dari Langkah 1.

---

### ❌ Tombol 🔒 Belum siap di Tab Kombinasi
**Penyebab:** Model baseline atau model GB belum ada sebagai best model.  
**Solusi:** Kembali ke Tab A, latih algoritma baseline yang diperlukan (LR/SVM/DT/RF) **dan** latih GB terlebih dahulu.

---

### ❌ Kombinasi stacking sudah ada (tidak bisa dilatih ulang)
**Penyebab:** LRGB1 / SVMGB1 / DTGB1 / RFGB1 sudah pernah dilatih — ID tetap, tidak bisa diulang.  
**Solusi:** Ini by design. Jika ingin melatih ulang, hapus semua riwayat dengan tombol "🗑️ Hapus Semua Riwayat Pelatihan" dan mulai dari awal.

---

### ❌ Model tidak menggantikan best setelah training
**Penyebab:** Model baru akurasinya lebih rendah dari yang sudah tersimpan.  
**Solusi:** Coba parameter lain atau algoritma berbeda.

---

### ❌ "Tidak ada fitur yang berhasil diekstrak"
**Penyebab:** Gambar tidak dapat dibaca atau bukan JPEG valid.  
**Solusi:** Pastikan file berformat JPG/JPEG asli (bukan PNG diganti ekstensi). Upload ulang dengan gambar berbeda.

---

### ❌ Riwayat pelatihan/pengujian hilang setelah refresh browser
**Penyebab:** Tabel rekap disimpan di session state (memori tab browser), bukan di file permanen.  
**Solusi:** Simpan screenshot sebelum refresh. File model (`.joblib`) tetap ada di disk.

---

### ❌ Training sangat lambat
**Penyebab:** Normal untuk Random Forest (n_estimators=500) dan Gradient Boosting, terutama saat multiselect banyak kombinasi sekaligus.  
**Solusi:** Tunggu hingga selesai. Progress muncul di notifikasi per model.

---

### ❌ Error "ModuleNotFoundError"
```bash
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

---

### ❌ Port 8501 sudah digunakan
```bash
streamlit run app.py --server.port 8502
```

---

## 10. Struktur File Sistem

```
weed-density-app/
├── requirements.txt              # Daftar library + versi minimum
├── runtime.txt                   # Versi Python (3.10)
├── .gitignore                    # File yang tidak di-upload ke GitHub
├── README.md                     # Dokumentasi singkat proyek
├── PANDUAN_PENGGUNAAN.md         # 📖 File ini
└── streamlit_app/
    ├── app.py                    # 🎨 Antarmuka utama (2 halaman: Pemodelan & Implementasi)
    ├── predict.py                # 🧠 Training dinamis, stacking, inference, save/load
    ├── preprocessing.py          # 🖼️ Pipeline preprocessing gambar
    ├── feature_extraction.py     # 🔬 Ekstraksi 19 atau 39 fitur
    ├── Data_ekstraksi_Fitur_Gulma.csv  # 📊 Dataset 2.097 sampel
    └── models/                   # 💾 Folder model (dibuat otomatis saat training)
        ├── TRAINED_LR1.joblib    # Best LR
        ├── TRAINED_SVM1.joblib   # Best SVM
        ├── TRAINED_DT1.joblib    # Best DT
        ├── TRAINED_RF1.joblib    # Best RF
        ├── TRAINED_GB1.joblib    # Best GB
        ├── TRAINED_LRGB1.joblib  # LR + GB Stacking (ID tetap)
        ├── TRAINED_SVMGB1.joblib # SVM + GB Stacking (ID tetap)
        ├── TRAINED_DTGB1.joblib  # DT + GB Stacking (ID tetap)
        └── TRAINED_RFGB1.joblib  # RF + GB Stacking (ID tetap)
```

### Alur Pipeline Lengkap

```
Upload Gambar (JPG/JPEG, min. 9/kelas)
    ↓
Resize 224×224 → Gaussian Blur (5×5) → HSV Segmentation → Morphological Closing
    ↓
Ekstraksi Fitur:
  Mode 19: RGB mean/std + HSV mean/std + Hu Moments = 19 fitur
  Mode 39: GLCM (4 sudut × 5 prop) + RGB + HSV + Hu = 39 fitur
    ↓
(Mode 39 saja) Information Gain → pilih 14 fitur terbaik
    ↓
StandardScaler (normalisasi z-score)
    ↓
Stratified Split: 80% Train | 10% Validasi | 10% Test
    ↓
Tab A — Training Dinamis (multiselect parameter, ID sekuensial)
  LR1..LRn / SVM1..SVMn / DT1..DTn / RF1..RFn / GB1..GBn
  Auto-save: best per algoritma → models/TRAINED_<ID>.joblib
    ↓
Tab B — Stacking Ensemble (ID tetap)
  LRGB1 = LR_best + GB_best (StackingClassifier, cv=5, passthrough=True)
  SVMGB1 = SVM_best + GB_best
  DTGB1 = DT_best + GB_best
  RFGB1 = RF_best + GB_best
    ↓
Halaman Implementasi:
  Pilih model → Upload per kelas (maks. 50/kelas × 3 kelas)
  → Progress bar → Prediksi semua gambar
  → Pivot table (Gambar | Aktual | Prediksi | Hasil per model)
  → Summary: TRUE/FALSE/Presisi + breakdown per kelas
  → Narasi kesimpulan otomatis
```

---

*Panduan ini ditulis untuk Tugas Akhir Kholifah Dina — Universitas Telkom Purwokerto*  
*Sistem dikembangkan dengan Python, Streamlit, scikit-learn, dan OpenCV*
