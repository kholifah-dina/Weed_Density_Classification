# 📖 Panduan Penggunaan Sistem Klasifikasi Kepadatan Gulma

**Sistem:** Weed Density Classification System  
**Versi:** 4.0 (2 Halaman: Pemodelan & Implementasi — ID Model Sekuensial, Top 5 Auto-Save)  
**Pemilik:** Kholifah Dina — Tugas Akhir, Universitas Telkom Purwokerto  

---

## Daftar Isi

1. [Persyaratan Sistem](#1-persyaratan-sistem)
2. [Cara Menjalankan Aplikasi](#2-cara-menjalankan-aplikasi)
3. [Navigasi Sistem](#3-navigasi-sistem)
4. [Halaman 1 — Pemodelan](#4-halaman-1--pemodelan)
5. [Halaman 2 — Implementasi](#5-halaman-2--implementasi)
6. [Alur Penggunaan yang Disarankan](#6-alur-penggunaan-yang-disarankan)
7. [Penjelasan Istilah Teknis](#7-penjelasan-istilah-teknis)
8. [Troubleshooting (Masalah & Solusi)](#8-troubleshooting-masalah--solusi)
9. [Struktur File Sistem](#9-struktur-file-sistem)

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

> 🌙 Sistem mendukung **Light Mode** dan **Dark Mode** secara otomatis — semua elemen teks, tabel, dan grafik menyesuaikan tema yang aktif.

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
    🏆 LR2  🏆 SVM1  🏆 GB3  (contoh)
```

> **Tidak ada login/role.** Semua fitur langsung bisa diakses oleh siapapun.

---

## 4. Halaman 1 — Pemodelan

### Tujuan
Melatih model klasifikasi gulma dari dataset gambar secara bertahap dan edukatif. Setiap percobaan training diberi **ID unik** dan dicatat di **tabel rekap real-time**. Lima model terbaik disimpan otomatis untuk digunakan di halaman Implementasi.

### Langkah-langkah

---

#### 📁 Langkah 1 — Upload Dataset

1. Buka halaman **🔬 Pemodelan** dari sidebar
2. Perhatikan info box: `📎 Format JPG/JPEG · Maks. 190 gambar/kelas · Min. 9 gambar/kelas`
3. Upload gambar untuk **3 kelas** menggunakan area drag-and-drop:
   - 🟢 **Renggang** — gambar lahan dengan gulma jarang
   - 🟡 **Sedang** — gambar lahan dengan gulma sedang
   - 🔴 **Padat** — gambar lahan dengan gulma padat
4. Syarat upload:
   - Format: **JPG atau JPEG** saja (bukan PNG)
   - **Minimal 9 gambar per kelas** (syarat wajib untuk stratified split 80:10:10)
   - **Maksimal 190 gambar per kelas**
5. Tabel ringkasan jumlah gambar per kelas otomatis muncul
6. Klik **"Lanjut ke Preprocessing →"**

> 💡 **Tips:** Semakin banyak gambar per kelas, semakin akurat model. Disarankan minimal 50 gambar/kelas.

---

#### 🖼️ Langkah 2 — Visualisasi Preprocessing

Sistem menampilkan **4 tahap preprocessing** otomatis dari 1 gambar contoh kelas Padat:

| Tahap | Nama | Penjelasan |
|-------|------|-----------|
| ① | Resize 224×224 | Gambar diubah ukuran menjadi 224×224 piksel sebagai standar input |
| ② | Gaussian Blur (5×5) | Menghaluskan gambar dan meredam noise agar segmentasi lebih akurat |
| ③ | HSV Thresholding | Mengisolasi piksel hijau (H:25–75°, S:40–255, V:50–255) sebagai area gulma |
| ④ | Morphological Closing | Mengisi celah kecil pada mask agar area gulma terdeteksi utuh |

Baca penjelasan setiap tahap, lalu klik **"Lanjut ke Ekstraksi Fitur →"**

---

#### 🧬 Langkah 3 — Ekstraksi Fitur

Pilih salah satu mode fitur:

| Mode | Jumlah Fitur | Seleksi | Input ke Model |
|------|-------------|---------|---------------|
| **19 Fitur** | RGB (6) + HSV (6) + Hu Moments (7) | Tidak ada | 19 fitur langsung |
| **39 Fitur** | GLCM (20) + RGB + HSV + Hu (19) | Information Gain | 14 fitur terbaik |

**Detail Mode 19 Fitur:**
- RGB mean & std (6 fitur): rata-rata dan variasi kanal R, G, B
- HSV mean & std (6 fitur): rata-rata dan variasi Hue, Saturation, Value
- Hu Moments (7 fitur): deskriptor bentuk invariant terhadap rotasi dan skala
- Semua 19 fitur langsung digunakan tanpa seleksi

**Detail Mode 39 Fitur:**
- GLCM: 5 properti tekstur × 4 sudut (0°, 45°, 90°, 135°) = 20 fitur
- Ditambah 19 fitur warna dan bentuk
- **Information Gain** otomatis memilih **14 fitur terbaik** dari 39 fitur

Klik **"Jalankan Ekstraksi Fitur →"** (atau "Jalankan Ekstraksi & Information Gain →" untuk mode 39).

**Output yang ditampilkan setelah ekstraksi berhasil:**

**A. Info Pembagian Data**  
Chip berwarna menampilkan jumlah sampel Train / Validasi / Test hasil stratified split 80:10:10.

**B. Tabel Contoh Hasil Ekstraksi**  
3 sampel per kelas (9 baris total) — setiap baris = satu gambar sebagai vektor numerik.

**C. Mode 39 Fitur — Grafik & Tabel Information Gain:**
- **Bar chart horizontal** — 39 fitur diurutkan berdasarkan IG Score (hijau = dipilih, abu = tidak)
- **Tabel ranking 39 fitur** — baris hijau = fitur terpilih, baris abu = tidak terpilih
- **Tabel 14 Fitur Terpilih** — dengan gradient warna berdasarkan IG Score

**D. Mode 19 Fitur — Tabel Daftar Fitur:**
- Tabel berisi: No, Nama Fitur, Kelompok, Deskripsi
- Warna per kelompok: 🟩 Hijau = RGB · 🟦 Biru = HSV · 🟧 Oranye = Hu Moments

---

#### 🤖 Langkah 4 — Pelatihan Dinamis

Ini adalah inti fitur baru sistem. Setiap klik **Latih** menghasilkan satu model dengan **ID unik**.

**Cara Penggunaan:**

1. Pilih algoritma dari dropdown:

| Algoritma | Kode | Karakteristik |
|-----------|------|--------------|
| Logistic Regression | LR | Model linear sederhana, sangat cepat |
| SVM | SVM | Efektif untuk data high-dimensional |
| Decision Tree | DT | Mudah diinterpretasi, cepat dilatih |
| Random Forest | RF | Ensemble pohon, robust terhadap overfitting |
| **Gradient Boosting** | **GB** | **Model utama penelitian — akurasi tinggi** |

2. Pilih **nilai parameter** (dropdown, nilai fixed sesuai skripsi):

| Algoritma | Parameter | Nilai Tersedia |
|-----------|-----------|---------------|
| Logistic Regression | `max_iter` | 100, 300, 500, 700, 1000 |
| SVM | `kernel` | linear, rbf, poly |
| Decision Tree | `max_depth` | 3, 5, 7, 9, 11 |
| Random Forest | `n_estimators` | 100, 200, 300, 400, 500 |
| Gradient Boosting | `n_estimators` | 100, 200, 300 |
| Gradient Boosting | `learning_rate` | 0.01, 0.1, 1 |

3. Lihat preview ID model berikutnya (contoh: **LR1**, **SVM1**, **GB3**)
4. Klik tombol **"🚀 Latih [ID] ([parameter])"**
5. Tunggu proses selesai (beberapa detik hingga beberapa menit)
6. Sistem menampilkan notifikasi sukses: apakah model masuk **Top 5** atau tidak

**Tabel Rekap Real-time:**

Setiap training menambah satu baris ke tabel di bawah form:

| Model ID | Algoritma | Parameter | Accuracy | Precision | Recall | F1-Score | Waktu (s) | Status |
|----------|-----------|-----------|----------|-----------|--------|----------|-----------|--------|
| LR1 | Logistic Regression | max_iter=100 | 0.8500 | 0.8612 | 0.8500 | 0.8421 | 1.23 | 🏆 Top 5 |
| LR2 | Logistic Regression | max_iter=300 | 0.8800 | 0.8901 | 0.8800 | 0.8750 | 1.45 | 🏆 Top 5 |
| SVM1 | SVM | kernel=rbf | 0.9000 | 0.9123 | 0.9000 | 0.8967 | 2.30 | 🏆 Top 5 |

- Baris **hijau tebal** = masuk Top 5 (tersimpan di disk)
- Badge "🏆 Terbaik saat ini" muncul di bawah tabel secara otomatis

**Auto-Save Top 5:**
- Setiap selesai training, sistem mengurutkan semua riwayat berdasarkan Accuracy
- **5 model tertinggi** disimpan sebagai `TRAINED_<ID>.joblib` di folder `models/`
- Model yang tidak masuk Top 5 dihapus otomatis dari disk
- Sidebar menampilkan daftar Top 5 yang aktif

**Tombol di bawah:**
- **"🔄 Mulai Ulang dari Langkah 1"** — reset langkah 1-4, riwayat pelatihan tetap
- **"🗑️ Hapus Semua Riwayat Pelatihan"** — hapus semua riwayat + file model di disk

> 📌 Anda bisa melatih hingga **23 model** (5 LR + 3 SVM + 5 DT + 5 RF + 3×3 GB = 23 kombinasi) dan sistem akan otomatis menjaga hanya **5 terbaik** yang tersimpan.

---

## 5. Halaman 2 — Implementasi

### Tujuan
Mengevaluasi model terpilih menggunakan gambar uji nyata dengan perbandingan label aktual (ground truth) secara manual — sesuai format dokumentasi skripsi (mirip tabel Excel pengujian).

### Cara Penggunaan

---

#### 🤖 Pilih Model

1. Buka halaman **🎯 Implementasi** dari sidebar
2. Dropdown **"Model (dari Top 5 hasil pemodelan)"** menampilkan hanya 5 model terbaik
3. Pilih model yang ingin diuji (contoh: LR2, SVM1, GB3)
4. Info card model muncul otomatis:
   - Nama algoritma, parameter, mode fitur
   - Metrik training: Accuracy, Precision, Recall, F1-Score

> ⚠️ Jika dropdown kosong, berarti belum ada model yang dilatih. Kembali ke halaman **🔬 Pemodelan** dan latih minimal satu model.

---

#### 📤 Upload Gambar Uji

1. Upload hingga **10 gambar** sekaligus (JPG/JPEG)
2. Jika upload lebih dari 10, hanya 10 gambar pertama yang diproses (ada peringatan)
3. Gambar ditampilkan dalam **grid thumbnail** (maks. 5 kolom)

---

#### 🏷️ Tentukan Label Aktual

Di bawah setiap thumbnail gambar, tersedia **dropdown label aktual**:

```
[thumbnail]    [thumbnail]    [thumbnail]
Label #1        Label #2        Label #3
[Renggang ▼]   [Sedang   ▼]   [Padat    ▼]
```

- Pilih label sesuai dengan yang Anda identifikasi secara **visual** dari gambar tersebut
- Ini adalah **Ground Truth** yang akan dibandingkan dengan prediksi model

---

#### 🔍 Uji Model & Lihat Hasil

1. Klik **"🔍 Uji Model"**
2. Sistem memproses setiap gambar: preprocessing → ekstraksi fitur → prediksi
3. **Tabel hasil** muncul sesuai format dokumentasi skripsi:

| No | Gambar | Label Aktual | Prediksi | Hasil |
|----|--------|-------------|----------|-------|
| 1 | gambar1.jpg | R | R | **TRUE** ← teks hijau |
| 2 | gambar2.jpg | R | S | **FALSE** ← teks merah |
| 3 | gambar3.jpg | S | S | **TRUE** ← teks hijau |
| 4 | gambar4.jpg | P | P | **TRUE** ← teks hijau |

- **TRUE** (teks hijau) = prediksi cocok dengan label aktual
- **FALSE** (teks merah) = prediksi tidak cocok

4. **Summary otomatis** di bawah tabel:

| Jumlah TRUE | Jumlah FALSE | Presisinya |
|-------------|--------------|------------|
| 7 | 3 | 70% |

---

#### 📋 Rekap Multi-Model

Setiap kali Anda **mengganti model dan menguji lagi**, sistem menambah baris ke tabel rekap:

| Model | Algoritma | Parameter | Jml Gambar | TRUE | FALSE | Presisi |
|-------|-----------|-----------|------------|------|-------|---------|
| LR1 | Logistic Regression | max_iter=100 | 10 | 7 | 3 | 70% |
| SVM3 | SVM | kernel=linear | 10 | 9 | 1 | 90% |
| GB1 | Gradient Boosting | n_est=200, lr=0.1 | 10 | 8 | 2 | 80% |
| **Rata-rata** | | | | | | **80%** |

---

#### 💬 Narasi Kesimpulan Otomatis

Sistem memberikan kesimpulan berdasarkan **rata-rata presisi** semua model yang diuji:

| Rata-rata Presisi | Kesimpulan |
|-------------------|-----------|
| ≥ 90% | "Model menunjukkan keandalan **sangat tinggi**..." |
| 70–89% | "Model menunjukkan keandalan **cukup baik**..." |
| < 70% | "Diperlukan evaluasi lebih lanjut..." |

**Tombol:** "🗑️ Hapus Riwayat Pengujian" — menghapus semua hasil uji di sesi ini.

---

## 6. Alur Penggunaan yang Disarankan

### Untuk Pertama Kali (Setelah Clone)

```
1. Halaman 🔬 Pemodelan
   → Upload dataset (min. 9 gambar/kelas, disarankan ≥ 50/kelas)
   → Pahami visualisasi preprocessing (Langkah 2)
   → Pilih mode 39 Fitur → Jalankan Ekstraksi → baca grafik IG
   → Langkah 4: latih SEMUA variasi parameter untuk setiap algoritma:
       LR: max_iter = 100, 300, 500, 700, 1000  → LR1 – LR5
       SVM: kernel = linear, rbf, poly           → SVM1 – SVM3
       DT: max_depth = 3, 5, 7, 9, 11           → DT1 – DT5
       RF: n_estimators = 100..500              → RF1 – RF5
       GB: kombinasi n_est × learning_rate      → GB1 – GB9
   → Pantau tabel rekap, identifikasi Top 5 terbaik

2. Halaman 🎯 Implementasi
   → Pilih model dari Top 5 (misal: GB3 dengan akurasi tertinggi)
   → Upload 10 gambar uji yang belum pernah digunakan saat training
   → Isi label aktual sesuai identifikasi visual
   → Klik Uji Model → catat hasil tabel
   → Ulangi untuk model lain dari Top 5
   → Bandingkan presisi antar model di tabel rekap
   → Gunakan narasi kesimpulan untuk laporan skripsi
```

### Untuk Sesi Lanjutan (Model Sudah Ada)
Selama session browser belum di-refresh, tabel rekap dan riwayat pengujian masih tersedia.  
Jika refresh, latih ulang di Pemodelan untuk mengisi kembali Top 5.

### Untuk Presentasi / Sidang
1. Jalankan aplikasi sebelum presentasi
2. Lakukan training semua variasi di Pemodelan — tunjukkan tabel rekap real-time
3. Lakukan pengujian di Implementasi — tunjukkan tabel TRUE/FALSE dan ringkasan presisi
4. Screenshot tabel rekap multi-model + narasi kesimpulan untuk bahan laporan

---

## 7. Penjelasan Istilah Teknis

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
| **Information Gain (LAN)** | Mengukur seberapa besar fitur membantu memisahkan kelas |
| **SelectKBest** | Memilih K fitur dengan skor IG tertinggi |
| **StandardScaler** | Normalisasi data ke skala z-score |
| **Stratified Split** | Pembagian data yang mempertahankan proporsi tiap kelas |
| **ID Model Sekuensial** | Penomoran unik per algoritma: LR1, LR2, SVM1, GB3, dst. |
| **Top 5** | 5 model dengan Accuracy tertinggi dari semua percobaan training |
| **Ground Truth** | Label aktual yang ditentukan secara manual oleh pengguna |
| **Presisi (Implementasi)** | Jumlah prediksi benar / total gambar uji × 100% |
| **Accuracy** | Persentase prediksi benar dari total prediksi (data test training) |
| **Precision** | Dari yang diprediksi kelas X, berapa persen yang benar-benar kelas X |
| **Recall** | Dari yang sebenarnya kelas X, berapa persen yang berhasil terdeteksi |
| **F1-Score** | Rata-rata harmonis Precision dan Recall |
| **Confusion Matrix** | Tabel yang menunjukkan prediksi benar vs salah per kelas |
| **Overfitting** | Model terlalu hafal data training, performa buruk di data baru |
| **Gradient Boosting** | Ensemble bertahap — tiap pohon memperbaiki error sebelumnya |
| **Random Forest** | Ensemble pohon independen yang dipilih secara acak |
| **n_estimators** | Jumlah pohon dalam model ensemble |
| **learning_rate** | Seberapa besar kontribusi tiap pohon baru dalam Gradient Boosting |
| **max_depth** | Kedalaman maksimum sebuah pohon keputusan |
| **max_iter** | Jumlah iterasi maksimum dalam Logistic Regression |
| **kernel** | Fungsi transformasi ruang fitur pada SVM (linear/rbf/poly) |
| **Hu Moments** | 7 deskriptor bentuk objek invariant terhadap rotasi, skala, translasi |

---

## 8. Troubleshooting (Masalah & Solusi)

### ❌ Dropdown model kosong di halaman Implementasi
**Penyebab:** Belum ada model Top 5 yang tersimpan.  
**Solusi:** Buka halaman **🔬 Pemodelan** → lakukan training minimal satu model di Langkah 4 → pastikan masuk Top 5 → kembali ke Implementasi.

---

### ❌ "Setiap kelas membutuhkan minimal 9 gambar"
**Penyebab:** Jumlah gambar kurang dari 9 per kelas. Stratified split 80:10:10 membutuhkan minimal 9 agar setiap subset mendapat minimal 1 sampel per kelas.  
**Solusi:** Tambah gambar hingga **≥ 9 per kelas**, lalu ulangi dari Langkah 1.

---

### ❌ Model tidak muncul di Top 5 setelah training
**Penyebab:** Model yang baru dilatih akurasinya lebih rendah dari 5 model yang sudah tersimpan.  
**Solusi:** Coba algoritma atau parameter lain, atau hapus riwayat (tombol "🗑️ Hapus Semua Riwayat") dan mulai ulang.

---

### ❌ "Tidak ada fitur yang berhasil diekstrak"
**Penyebab:** Gambar tidak dapat dibaca atau bukan JPEG valid.  
**Solusi:** Pastikan file berformat JPG/JPEG asli (bukan PNG diganti ekstensi). Upload ulang dengan gambar berbeda.

---

### ❌ Riwayat pelatihan/pengujian hilang setelah refresh browser
**Penyebab:** Tabel rekap disimpan di session state (memori tab browser), bukan di file permanen.  
**Solusi:** Simpan screenshot sebelum refresh. File model Top 5 (`.joblib`) tetap ada di disk — Anda hanya perlu training ulang agar tabel rekap terisi kembali.

---

### ❌ Training sangat lambat
**Penyebab:** Normal untuk Random Forest (n_estimators=500) dan Gradient Boosting.  
**Solusi:** Tunggu hingga selesai. Untuk percobaan awal gunakan Decision Tree atau LR yang lebih cepat.

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

### ❌ Error install opencv
```bash
pip install opencv-python-headless
```

---

### ❌ Teks tabel tidak terbaca di Dark Mode
**Solusi:** Pastikan sudah `git pull origin main` untuk mendapat versi terbaru.

---

## 9. Struktur File Sistem

```
weed-density-app/
├── requirements.txt              # Daftar library + versi minimum
├── runtime.txt                   # Versi Python (3.10)
├── .gitignore                    # File yang tidak di-upload ke GitHub
├── README.md                     # Dokumentasi singkat proyek
├── PANDUAN_PENGGUNAAN.md         # 📖 File ini
└── streamlit_app/
    ├── app.py                    # 🎨 Antarmuka utama (2 halaman: Pemodelan & Implementasi)
    ├── predict.py                # 🧠 Training dinamis, inference, save/load, Top 5 management
    ├── preprocessing.py          # 🖼️ Pipeline preprocessing gambar
    ├── feature_extraction.py     # 🔬 Ekstraksi 19 atau 39 fitur
    ├── Data_ekstraksi_Fitur_Gulma.csv  # 📊 Dataset 2.097 sampel
    └── models/                   # 💾 Folder model (dibuat otomatis saat training)
        └── TRAINED_<ID>.joblib   # Contoh: TRAINED_LR1.joblib, TRAINED_GB3.joblib
```

### Isi setiap file `TRAINED_<ID>.joblib`

| Key | Isi |
|-----|-----|
| `model` | Objek classifier yang sudah dilatih |
| `scaler` | StandardScaler yang sudah di-fit pada data training |
| `selector` | SelectKBest untuk seleksi IG (hanya jika mode 39 fitur) |
| `feature_mode` | `'19'` atau `'39'` |
| `features_used` | Daftar nama fitur yang digunakan model |
| `n_features` | Jumlah fitur input model (19 atau 14) |
| `metrics` | Accuracy, Precision, Recall, F1-Score, Val Accuracy, Execution Time |
| `confusion_matrix` | Matriks konfusi + label kelas |
| `split_info` | Jumlah sampel train / validasi / test |
| `model_id` | ID unik: LR1, SVM3, GB2, dst. |
| `algo_full` | Nama lengkap algoritma |
| `param_str` | Konfigurasi parameter: "max_iter=300", "kernel=rbf", dst. |

---

### Alur Pipeline Lengkap

```
Upload Gambar (JPG/JPEG)
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
Split stratified: 80% Train | 10% Validasi | 10% Test  (min. 9 sampel/kelas)
    ↓
Training Dinamis — ID Unik per percobaan (LR1, LR2, SVM1, GB3, ...)
    ↓
Evaluasi pada Test Split → Accuracy, Precision, Recall, F1-Score
    ↓
Auto-Save Top 5 → models/TRAINED_<ID>.joblib
    ↓
Halaman Implementasi:
  Pilih model → Upload gambar uji → Label aktual manual
  → Prediksi → Tabel TRUE/FALSE → Presisi → Kesimpulan
```

---

*Panduan ini ditulis untuk Tugas Akhir Kholifah Dina — Universitas Telkom Purwokerto*  
*Sistem dikembangkan dengan Python, Streamlit, scikit-learn, dan OpenCV*
