# 📖 Panduan Penggunaan Sistem Klasifikasi Kepadatan Gulma

**Sistem:** Weed Density Classification System  
**Versi:** 2.0 (Refactored — 4 Menu Sidebar)  
**Pemilik:** Kholifah Dina — Tugas Akhir, Universitas Telkom Purwokerto  

---

## Daftar Isi

1. [Persyaratan Sistem](#1-persyaratan-sistem)
2. [Cara Menjalankan Aplikasi](#2-cara-menjalankan-aplikasi)
3. [Navigasi Sistem](#3-navigasi-sistem)
4. [Menu 1 — Alur Pelatihan](#4-menu-1--alur-pelatihan)
5. [Menu 2 — Pengujian Gambar](#5-menu-2--pengujian-gambar)
6. [Menu 3 — Eksperimen Parameter](#6-menu-3--eksperimen-parameter)
7. [Menu 4 — Dashboard Hasil](#7-menu-4--dashboard-hasil)
8. [Alur Penggunaan yang Disarankan](#8-alur-penggunaan-yang-disarankan)
9. [Penjelasan Istilah Teknis](#9-penjelasan-istilah-teknis)
10. [Troubleshooting (Masalah & Solusi)](#10-troubleshooting-masalah--solusi)
11. [Struktur File Sistem](#11-struktur-file-sistem)

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

Aplikasi akan terbuka otomatis di browser di alamat: **http://localhost:8501**

> ⚠️ **Catatan Penting:** File model (`DT.joblib`, `RF.joblib`, dll.) **tidak disertakan** di GitHub karena ukurannya besar. Setelah clone, Anda **harus melakukan training** terlebih dahulu melalui Menu Alur Pelatihan sebelum bisa melakukan pengujian.

---

## 3. Navigasi Sistem

Sistem menggunakan **sidebar di sebelah kiri** untuk navigasi. Klik salah satu dari 4 menu:

```
🌿 Kepadatan Gulma
├── 📚 Alur Pelatihan     ← Training model langkah demi langkah
├── 🎯 Pengujian Gambar   ← Uji satu gambar, lihat prediksi
├── 🔬 Eksperimen Parameter ← Tuning parameter, analisis performa
└── 📊 Dashboard Hasil    ← Rekap semua hasil & laporan
```

> **Tidak ada login/role.** Semua fitur langsung bisa diakses oleh siapapun.

---

## 4. Menu 1 — Alur Pelatihan

### Tujuan
Melatih model klasifikasi dari dataset gambar gulma. Proses bersifat **edukatif** — setiap tahap divisualisasikan dengan penjelasan.

### Langkah-langkah

---

#### 📁 Langkah 1 — Upload Dataset

1. Buka menu **📚 Alur Pelatihan**
2. Upload gambar untuk **3 kelas**:
   - 🟢 **Renggang** — gambar lahan dengan gulma jarang
   - 🟡 **Sedang** — gambar lahan dengan gulma sedang
   - 🔴 **Padat** — gambar lahan dengan gulma padat
3. Format gambar: **JPG atau JPEG** — maksimum **190 gambar per kelas**
4. Pastikan minimal **1 gambar per kelas** sebelum melanjutkan
5. Setelah semua kelas terisi, klik **"Lanjut ke Preprocessing →"**

> 💡 **Tips:** Semakin banyak gambar per kelas, semakin akurat model yang dihasilkan. Disarankan minimal 50 gambar per kelas.

---

#### 🖼️ Langkah 2 — Visualisasi Preprocessing

Sistem secara otomatis menampilkan 4 tahap preprocessing menggunakan 1 gambar contoh dari kelas Padat:

| Tahap | Nama | Penjelasan |
|-------|------|-----------|
| ① | Resize 224×224 | Gambar diubah ukuran menjadi 224×224 piksel sebagai standar input |
| ② | Gaussian Blur (5×5) | Menghaluskan gambar dan meredam noise agar segmentasi lebih akurat |
| ③ | HSV Thresholding | Mengisolasi piksel berwarna hijau (H:25–75°) sebagai area gulma |
| ④ | Morphological Closing | Mengisi celah kecil pada mask agar area gulma terdeteksi utuh |

Baca dan pahami setiap penjelasan tahap, lalu klik **"Lanjut ke Ekstraksi Fitur →"**

---

#### 🧬 Langkah 3 — Ekstraksi Fitur

Pilih jumlah fitur yang akan digunakan:

**Opsi A — 19 Fitur (tanpa GLCM):**
- RGB mean & std (6 fitur): rata-rata dan variasi warna merah, hijau, biru
- HSV mean & std (6 fitur): rata-rata dan variasi hue, saturation, value
- Hu Moments (7 fitur): deskriptor bentuk yang invariant terhadap rotasi/skala
- Semua 19 fitur langsung digunakan tanpa seleksi

**Opsi B — 39 Fitur (dengan GLCM) ← Disarankan:**
- GLCM: 5 properti tekstur × 4 sudut = 20 fitur tambahan
- + 19 fitur warna/bentuk di atas
- **Information Gain (LAN)** otomatis memilih **14 fitur terbaik** dari 39 fitur

Setelah memilih, klik tombol jalankan. Sistem akan:
1. Mengekstrak fitur dari SEMUA gambar yang diupload
2. (Jika 39 fitur) Menjalankan Information Gain dan menampilkan **tabel ranking fitur**
3. Menampilkan informasi pembagian data: Train / Val / Test

**Membaca Tabel Ranking Information Gain:**
- Baris hijau = fitur yang **terpilih** (14 teratas)
- Baris abu-abu = fitur yang **tidak dipilih** (IG Score rendah)
- Semakin tinggi IG Score, semakin informatif fitur tersebut

---

#### 🤖 Langkah 4 — Pilih & Latih Model

1. Pilih salah satu algoritma dari dropdown:

| Algoritma | Singkatan | Karakteristik |
|-----------|-----------|--------------|
| Decision Tree | DT | Mudah diinterpretasi, cepat |
| Logistic Regression | LR | Model linear sederhana, sangat cepat |
| Support Vector Machine | SVM | Efektif untuk data high-dimensional |
| Random Forest | RF | Ensemble 500 pohon, akurasi tinggi |
| **Gradient Boosting** | **GB** | **Model utama penelitian, akurasi tertinggi** |

2. Klik **"🚀 Latih Model [NAMA]"**
3. Tunggu proses training selesai (beberapa detik hingga beberapa menit tergantung algoritma)
4. Sistem menampilkan hasil:
   - Badge akurasi (Accuracy, Precision, Recall, F1-Score, Val Accuracy, Execution Time)
   - Confusion Matrix (visualisasi prediksi benar vs salah)

5. **Model tersimpan otomatis** sebagai `models/[SINGKATAN].joblib` (contoh: `models/GB.joblib`)

**Ingin melatih model lain?** Cukup ubah pilihan di dropdown dan klik Latih lagi — tanpa perlu upload ulang gambar.

**Ingin mulai ulang dari awal?** Klik tombol **"🔄 Mulai Ulang dari Langkah 1"**

---

### ⚠️ Hal Penting tentang Training

- Data dibagi secara **stratified 80:10:10** (80% train, 10% validasi, 10% test)
- Model dievaluasi menggunakan **data test yang tidak pernah dilihat model** selama training
- Semua model menggunakan `class_weight='balanced'` untuk mengatasi ketidakseimbangan kelas
- Gradient Boosting menggunakan `sample_weight` karena tidak mendukung `class_weight` di constructor

---

## 5. Menu 2 — Pengujian Gambar

### Tujuan
Menguji satu gambar secara interaktif menggunakan model yang sudah dilatih.

### Cara Penggunaan

1. **Upload** satu gambar (JPG/JPEG) melalui file uploader
2. Sistem otomatis menampilkan **4 tahap preprocessing** pada gambar tersebut
3. **Pilih model** dari dropdown (hanya model yang sudah dilatih yang muncul)
4. Sistem menampilkan informasi model: mode fitur, fitur yang digunakan
5. Hasil prediksi muncul dalam **kartu besar** berwarna:
   - 🟢 Hijau = Renggang
   - 🟡 Kuning = Sedang
   - 🔴 Merah = Padat
6. Di bawah kartu prediksi ditampilkan:
   - **Metrik evaluasi model** (dari data test training, bukan gambar ini)
   - **Confusion Matrix** model tersebut
7. Ganti model di dropdown untuk **membandingkan hasil** pada gambar yang sama

### Expander "Bandingkan Prediksi Semua Model"
Klik untuk melihat prediksi dari semua model yang tersedia sekaligus dalam satu tabel.

### ⚠️ Catatan
- Confusion Matrix yang ditampilkan adalah dari **test split training** (bukan dari gambar ini)
- Metrik evaluasi bukan dihitung dari gambar ini, melainkan tersimpan di file `.joblib`

---

## 6. Menu 3 — Eksperimen Parameter

### Tujuan
Menganalisis pengaruh perubahan parameter terhadap performa klasifikasi. Cocok untuk kebutuhan skripsi (analisis mendalam).

### Sumber Data
Eksperimen menggunakan **dataset CSV penuh** (`Data_ekstraksi_Fitur_Gulma.csv` — 2.097 sampel) sehingga tidak perlu upload gambar ulang.

### Cara Penggunaan

1. **Pilih Mode Fitur:**
   - `19 Fitur (tanpa GLCM)` — semua 19 fitur langsung
   - `39 Fitur (dengan GLCM + IG)` — 39 fitur → dipilih 14 terbaik via Information Gain

2. **Pilih Model** dari dropdown

3. **Pilih nilai parameter** yang ingin diuji (bisa pilih lebih dari satu):

| Model | Parameter | Pilihan Default |
|-------|-----------|-----------------|
| Logistic Regression | max_iter | [100, 300, 500, 700, 1000] |
| SVM | kernel | [linear, rbf, poly] |
| Decision Tree | max_depth | [3, 5, 7, 9, 11] |
| Random Forest | n_estimators | [100, 200, 300, 400, 500] |
| Gradient Boosting | n_estimators + learning_rate | kombinasi keduanya |

4. Klik **"🚀 Jalankan Eksperimen"**

5. Hasil yang ditampilkan:
   - **Tabel metrik** semua konfigurasi parameter (baris terbaik disorot hijau)
   - **Grafik** metrik vs parameter (line chart / bar chart)
   - **Confusion Matrix** untuk konfigurasi parameter terbaik
   - **Classification Report** detail (precision, recall, F1 per kelas)

6. Klik **Jalankan Eksperimen** lagi untuk menambah hasil (hasil **terakumulasi**)

7. Klik **"🗑️ Hapus Riwayat Eksperimen Model Ini"** untuk membersihkan hasil model tersebut

### Tips Penggunaan Eksperimen
- Jalankan eksperimen untuk **semua model** agar bisa dibandingkan di Dashboard
- Coba berbagai kombinasi parameter untuk menemukan konfigurasi terbaik
- Hasil semua eksperimen tersimpan di session dan bisa dilihat di Dashboard Hasil
- Jika browser di-refresh, hasil eksperimen hilang (simpan screenshot dahulu jika perlu)

---

## 7. Menu 4 — Dashboard Hasil

### Tujuan
Rekap menyeluruh seluruh hasil sistem dalam satu halaman — cocok untuk laporan, presentasi, atau sidang skripsi.

### Konten Dashboard

#### Seksi 1 — Rekap Preprocessing
- Infografis 4 tahap preprocessing dengan penjelasan teks
- Visualisasi gambar contoh (jika training pernah dilakukan di sesi ini)

#### Seksi 2 — Perbandingan Semua Model Terlatih
- **Tabel perbandingan** akurasi, precision, recall, F1, val accuracy, execution time
- Baris **model terbaik** disorot hijau
- **Badge** menampilkan model terbaik dan akurasinya
- **Bar chart interaktif** perbandingan metrik semua model
- **Confusion Matrix** semua model (klik expander untuk melihat)

#### Seksi 3 — Rekap Semua Eksperimen
- Tabel gabungan semua hasil eksperimen parameter dari sesi ini
- **Line chart** per model: Accuracy & F1-Score vs Parameter

#### Seksi 4 — Penjelasan Sistem (untuk Pembaca Umum)
Klik masing-masing expander untuk membaca penjelasan:
- Apa itu Klasifikasi Kepadatan Gulma?
- Apa itu HSV Thresholding?
- Apa itu GLCM dan Information Gain?
- Apa itu Gradient Boosting?

---

## 8. Alur Penggunaan yang Disarankan

### Untuk Pertama Kali (Setelah Clone)

```
1. Menu Alur Pelatihan
   → Upload dataset (≥ 50 gambar per kelas)
   → Pahami visualisasi preprocessing
   → Pilih 39 fitur (dengan GLCM)
   → Latih SEMUA 5 model satu per satu
   → Semua model tersimpan di models/

2. Menu Pengujian Gambar
   → Upload 1 gambar baru
   → Coba semua model, bandingkan hasilnya

3. Menu Eksperimen Parameter
   → Jalankan eksperimen untuk setiap model
   → Analisis pengaruh parameter terhadap akurasi

4. Menu Dashboard Hasil
   → Lihat rekap semua hasil
   → Gunakan untuk laporan/presentasi
```

### Untuk Sesi Lanjutan (Model Sudah Ada)
Model tersimpan di file `.joblib` — tidak perlu training ulang.  
Langsung bisa ke **Menu Pengujian Gambar** atau **Eksperimen Parameter**.

### Untuk Presentasi / Sidang
Buka **Menu Dashboard Hasil** — semua informasi sudah terekap di sana.

---

## 9. Penjelasan Istilah Teknis

| Istilah | Penjelasan Sederhana |
|---------|---------------------|
| **Preprocessing** | Serangkaian langkah pengolahan gambar sebelum fitur diekstrak |
| **Gaussian Blur** | Teknik menghaluskan gambar untuk mengurangi gangguan/noise |
| **HSV** | Model warna berbasis Hue (warna), Saturation (kecerahan), Value (kecemerlangan) |
| **Thresholding** | Memilah piksel berdasarkan rentang nilai tertentu |
| **Segmentasi** | Proses memisahkan area gulma dari latar belakang |
| **Morphological Closing** | Mengisi lubang kecil pada area yang tersegmentasi |
| **GLCM** | Gray-Level Co-occurrence Matrix — matriks untuk mengukur tekstur gambar |
| **Fitur** | Nilai numerik yang merepresentasikan karakteristik gambar |
| **Information Gain (LAN)** | Metode memilih fitur paling informatif berdasarkan seberapa besar ia membantu klasifikasi |
| **SelectKBest** | Algoritma seleksi fitur yang memilih K fitur dengan skor tertinggi |
| **StandardScaler** | Normalisasi data agar semua fitur berada pada skala yang sama |
| **Stratified Split** | Pembagian data yang mempertahankan proporsi kelas |
| **Accuracy** | Persentase prediksi benar dari total prediksi |
| **Precision** | Dari yang diprediksi kelas X, berapa persen yang benar-benar kelas X |
| **Recall** | Dari yang sebenarnya kelas X, berapa persen yang berhasil terdeteksi |
| **F1-Score** | Rata-rata harmonis Precision dan Recall |
| **Val Accuracy** | Akurasi pada data validasi (digunakan untuk monitor overfitting) |
| **Confusion Matrix** | Tabel yang menunjukkan prediksi benar vs salah per kelas |
| **Overfitting** | Model terlalu hafal data training, performa buruk di data baru |
| **class_weight='balanced'** | Memberi bobot lebih pada kelas minoritas untuk mengatasi ketidakseimbangan dataset |
| **sample_weight** | Bobot per sampel, alternatif class_weight untuk GradientBoosting |
| **Ensemble** | Menggabungkan banyak model untuk hasil yang lebih akurat |
| **Gradient Boosting** | Model ensemble yang membangun pohon secara bertahap, tiap pohon memperbaiki error pohon sebelumnya |
| **Random Forest** | Model ensemble dari banyak pohon keputusan yang dibuat secara acak |
| **n_estimators** | Jumlah pohon dalam ensemble |
| **learning_rate** | Seberapa besar kontribusi tiap pohon baru dalam Gradient Boosting |
| **max_depth** | Kedalaman maksimum sebuah pohon keputusan |

---

## 10. Troubleshooting (Masalah & Solusi)

### ❌ "Belum ada model yang tersedia"
**Penyebab:** File `.joblib` model belum ada (pertama kali setelah clone, atau model terhapus).  
**Solusi:** Buka menu **Alur Pelatihan** dan latih minimal satu model.

---

### ❌ "File Data_ekstraksi_Fitur_Gulma.csv tidak ditemukan"
**Penyebab:** File CSV tidak ada di folder `streamlit_app/`.  
**Solusi:** Pastikan file `Data_ekstraksi_Fitur_Gulma.csv` berada di dalam folder `streamlit_app/`.

---

### ❌ Error saat training: "Tidak ada fitur yang berhasil diekstrak"
**Penyebab:** Gambar yang diupload tidak dapat dibaca atau bukan format JPEG yang valid.  
**Solusi:** 
1. Pastikan file berformat JPG/JPEG (bukan PNG yang diganti ekstensinya)
2. Coba upload ulang dengan gambar yang berbeda
3. Pastikan gambar tidak korup

---

### ❌ Training sangat lambat (lebih dari 10 menit)
**Penyebab:** Dataset terlalu besar atau model Random Forest/Gradient Boosting membutuhkan waktu lama.  
**Solusi:**
- Kurangi jumlah gambar per kelas (sementara)
- Random Forest (500 pohon) dan Gradient Boosting (300 iterasi) memang lebih lambat dari DT/LR/SVM
- Ini normal — tunggu hingga selesai

---

### ❌ Prediksi salah / tidak sesuai ekspektasi
**Penyebab:** Model dilatih dengan dataset yang kurang representatif.  
**Solusi:**
1. Tambah jumlah gambar training per kelas
2. Pastikan gambar training bervariasi (berbagai kondisi pencahayaan, sudut, dll.)
3. Coba model lain — Gradient Boosting biasanya paling akurat
4. Periksa apakah gambar uji kondisinya mirip dengan gambar training

---

### ❌ Refresh browser menghilangkan hasil eksperimen
**Penyebab:** Hasil eksperimen disimpan di session state (memori browser), bukan di file.  
**Solusi:** Simpan screenshot hasil eksperimen sebelum refresh, atau jangan refresh selama sesi eksperimen.

---

### ❌ Error "ImportError" atau "ModuleNotFoundError"
**Penyebab:** Library belum terinstall.  
**Solusi:**
```bash
pip install -r requirements.txt
```

---

### ❌ Port 8501 sudah digunakan
**Penyebab:** Ada instance Streamlit lain yang berjalan.  
**Solusi:**
```bash
streamlit run app.py --server.port 8502
```

---

## 11. Struktur File Sistem

```
weed-density-app/
├── requirements.txt              # Daftar library + versi minimum
├── runtime.txt                   # Versi Python (3.10)
├── .gitignore                    # File yang tidak di-upload ke GitHub
├── README.md                     # Dokumentasi singkat proyek
├── PANDUAN_PENGGUNAAN.md         # 📖 File ini
└── streamlit_app/
    ├── app.py                    # 🎨 Antarmuka utama (4 menu sidebar)
    ├── predict.py                # 🧠 Logika training, inference, save/load model
    ├── preprocessing.py          # 🖼️ Pipeline preprocessing gambar
    ├── feature_extraction.py     # 🔬 Ekstraksi 19 atau 39 fitur
    ├── Data_ekstraksi_Fitur_Gulma.csv  # 📊 Dataset 2.097 sampel (untuk Eksperimen)
    └── models/                   # 💾 Folder model (dibuat otomatis)
        ├── DT.joblib             # Model Decision Tree (dibuat setelah training)
        ├── LR.joblib             # Model Logistic Regression
        ├── SVM.joblib            # Model SVM
        ├── RF.joblib             # Model Random Forest
        └── GB.joblib             # Model Gradient Boosting
```

### Isi setiap file `.joblib`
Setiap file model menyimpan semua informasi yang dibutuhkan:
- `model` — objek classifier yang sudah dilatih
- `scaler` — StandardScaler yang sudah di-fit pada data training
- `selector` — SelectKBest (hanya ada jika mode 39 fitur)
- `feature_mode` — '19' atau '39'
- `features_used` — nama fitur yang digunakan model
- `metrics` — Accuracy, Precision, Recall, F1, Val Accuracy, Execution Time
- `confusion_matrix` — matriks dan label kelas
- `split_info` — jumlah sampel train/val/test

---

## Informasi Teknis Pipeline ML

### Alur Lengkap

```
Upload Gambar
    ↓
Resize 224×224 → Gaussian Blur (5×5) → HSV Segmentation → Morphological Closing
    ↓
Ekstrak Fitur:
  Mode 19: RGB mean/std + HSV mean/std + Hu Moments
  Mode 39: GLCM (4 sudut × 5 prop) + RGB + HSV + Hu = 39 fitur
    ↓
(Mode 39 saja) Information Gain → pilih 14 fitur terbaik
    ↓
StandardScaler (normalize z-score)
    ↓
Split: 80% Train | 10% Validasi | 10% Test (stratified)
    ↓
Training Model (DT / LR / SVM / RF / GB)
    ↓
Evaluasi pada Test Split → Metrik & Confusion Matrix
    ↓
Simpan ke models/[NAMA].joblib
```

### Konfigurasi Model Default (untuk Training di Menu Alur Pelatihan)

| Model | Parameter Utama |
|-------|----------------|
| Decision Tree | max_depth=3, class_weight='balanced' |
| Logistic Regression | solver='lbfgs', max_iter=300, class_weight='balanced' |
| SVM | kernel='rbf', C=5, gamma=0.01, class_weight='balanced' |
| Random Forest | n_estimators=500, class_weight='balanced' |
| Gradient Boosting | n_estimators=300, learning_rate=0.1, sample_weight (balanced) |

---

*Panduan ini ditulis untuk Tugas Akhir Kholifah Dina — Universitas Telkom Purwokerto*  
*Sistem dikembangkan dengan Python, Streamlit, scikit-learn, dan OpenCV*
