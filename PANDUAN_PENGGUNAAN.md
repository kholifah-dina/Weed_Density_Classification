# 📖 Panduan Penggunaan Sistem Klasifikasi Kepadatan Gulma

**Sistem:** Weed Density Classification System  
**Versi:** 3.0 (4 Menu Sidebar + Visualisasi Ekstraksi Fitur)  
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

> 🌙 Sistem mendukung **Light Mode** dan **Dark Mode** secara otomatis — semua elemen teks dan tabel menyesuaikan tema yang aktif.

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
├── 📚 Alur Pelatihan       ← Training model langkah demi langkah
├── 🎯 Pengujian Gambar     ← Uji satu gambar, lihat prediksi
├── 🔬 Eksperimen Parameter ← Tuning parameter, analisis performa
└── 📊 Dashboard Hasil      ← Rekap semua hasil & laporan
```

> **Tidak ada login/role.** Semua fitur langsung bisa diakses oleh siapapun.

---

## 4. Menu 1 — Alur Pelatihan

### Tujuan
Melatih model klasifikasi dari dataset gambar gulma. Proses bersifat **edukatif** — setiap tahap divisualisasikan dengan penjelasan lengkap.

### Langkah-langkah

---

#### 📁 Langkah 1 — Upload Dataset

1. Buka menu **📚 Alur Pelatihan**
2. Perhatikan info box di bagian atas: `📎 Format JPG/JPEG · Maks. 190 gambar/kelas · Min. 9 gambar/kelas`
3. Upload gambar untuk **3 kelas** menggunakan area drag-and-drop:
   - 🟢 **Renggang** — gambar lahan dengan gulma jarang
   - 🟡 **Sedang** — gambar lahan dengan gulma sedang
   - 🔴 **Padat** — gambar lahan dengan gulma padat
4. Syarat upload:
   - Format: **JPG atau JPEG** saja
   - **Minimal 9 gambar per kelas** (syarat wajib untuk stratified split 80:10:10)
   - **Maksimal 190 gambar per kelas**
5. Tabel ringkasan jumlah gambar per kelas otomatis muncul
6. Setelah semua kelas terisi dengan jumlah yang cukup, klik **"Lanjut ke Preprocessing →"**

> 💡 **Tips:** Semakin banyak gambar per kelas, semakin akurat model yang dihasilkan. Disarankan minimal 50 gambar per kelas untuk hasil yang baik.

---

#### 🖼️ Langkah 2 — Visualisasi Preprocessing

Sistem secara otomatis menampilkan **4 tahap preprocessing** menggunakan 1 gambar contoh dari kelas Padat:

| Tahap | Nama | Penjelasan |
|-------|------|-----------|
| ① | Resize 224×224 | Gambar diubah ukuran menjadi 224×224 piksel sebagai standar input |
| ② | Gaussian Blur (5×5) | Menghaluskan gambar dan meredam noise agar segmentasi lebih akurat |
| ③ | HSV Thresholding | Mengisolasi piksel berwarna hijau (H:25–75°, S:40–255, V:50–255) sebagai area gulma |
| ④ | Morphological Closing | Mengisi celah kecil pada mask agar area gulma terdeteksi utuh |

Baca dan pahami setiap penjelasan tahap, lalu klik **"Lanjut ke Ekstraksi Fitur →"**

---

#### 🧬 Langkah 3 — Ekstraksi Fitur

Pilih salah satu mode fitur:

| Mode | Jumlah Fitur | Seleksi | Input ke Model |
|------|-------------|---------|---------------|
| **19 Fitur** | RGB (6) + HSV (6) + Hu Moments (7) | Tidak ada | 19 fitur langsung |
| **39 Fitur** | GLCM (20) + RGB + HSV + Hu (19) | Information Gain | 14 fitur terbaik |

**Detail Mode 19 Fitur:**
- RGB mean & std (6 fitur): rata-rata dan variasi warna merah, hijau, biru
- HSV mean & std (6 fitur): rata-rata dan variasi hue, saturation, value
- Hu Moments (7 fitur): deskriptor bentuk yang invariant terhadap rotasi dan skala
- Semua 19 fitur langsung digunakan tanpa proses seleksi

**Detail Mode 39 Fitur:**
- GLCM: 5 properti tekstur (contrast, dissimilarity, homogeneity, energy, correlation) × 4 sudut (0°, 45°, 90°, 135°) = 20 fitur
- Ditambah 19 fitur warna dan bentuk di atas
- **Information Gain (LAN)** otomatis memilih **14 fitur terbaik** dari total 39 fitur

Setelah memilih mode, klik tombol **"Jalankan Ekstraksi Fitur →"** atau **"Jalankan Ekstraksi & Information Gain →"**.

**Output yang ditampilkan setelah ekstraksi berhasil:**

**A. Tabel Pembagian Data**
Menampilkan jumlah sampel Train / Validasi / Test hasil stratified split 80:10:10.

**B. Tabel Contoh Hasil Ekstraksi Fitur**
Menampilkan 3 sampel per kelas (total 9 baris) dalam bentuk vektor numerik — setiap baris adalah satu gambar yang direpresentasikan sebagai angka-angka fitur.

**C. Untuk Mode 39 Fitur — Grafik & Tabel Information Gain:**
- **Grafik bar chart horizontal** — semua 39 fitur diurutkan berdasarkan IG Score:
  - Batang **hijau** = fitur yang dipilih (14 fitur)
  - Batang **abu-abu** = fitur yang tidak dipilih (IG Score lebih rendah)
- **Tabel ranking semua 39 fitur** — diurutkan dari IG Score tertinggi:
  - Baris hijau tebal = fitur terpilih
  - Baris abu-abu = fitur tidak terpilih
- **Tabel 14 Fitur Terpilih** — daftar bersih fitur yang akan masuk ke model, dengan gradient warna hijau berdasarkan IG Score

**D. Untuk Mode 19 Fitur — Tabel Daftar 19 Fitur:**
- Tabel lengkap berisi: No, Nama Fitur, Kelompok, Deskripsi
- Warna per kelompok: 🟩 Hijau = RGB · 🟦 Biru = HSV · 🟧 Oranye = Hu Moments

> 📌 14 fitur terpilih bersifat **dinamis** — ditentukan otomatis oleh Information Gain berdasarkan dataset yang digunakan. Fitur berbeda dapat terpilih pada dataset yang berbeda.

---

#### 🤖 Langkah 4 — Pilih & Latih Model

1. Pilih salah satu algoritma dari dropdown:

| Algoritma | Singkatan | Karakteristik |
|-----------|-----------|--------------|
| Decision Tree | DT | Mudah diinterpretasi, cepat dilatih |
| Logistic Regression | LR | Model linear sederhana, baseline yang baik |
| Support Vector Machine | SVM | Efektif untuk data high-dimensional |
| Random Forest | RF | Ensemble 500 pohon, robust terhadap overfitting |
| **Gradient Boosting** | **GB** | **Model utama penelitian — akurasi tertinggi** |

2. Klik **"🚀 Latih Model [NAMA]"**
3. Tunggu proses selesai (beberapa detik hingga beberapa menit tergantung algoritma)
4. Sistem menampilkan hasil training:
   - Badge besar: nama model + Accuracy test
   - 6 metrik: Accuracy, Precision, Recall, F1-Score, Val Accuracy, Execution Time
   - Confusion Matrix interaktif dengan caption penjelasan
5. **Model tersimpan otomatis** ke `models/[SINGKATAN].joblib` (contoh: `models/GB.joblib`)

**Ingin melatih model lain?** Ubah pilihan di dropdown dan klik Latih lagi — tidak perlu upload ulang gambar.

**Ingin mulai ulang dari awal?** Klik tombol **"🔄 Mulai Ulang dari Langkah 1"**

---

### ⚠️ Hal Penting tentang Training

- Data dibagi secara **stratified 80:10:10** (80% train, 10% validasi, 10% test)
- Membutuhkan **minimal 9 gambar per kelas** agar stratified split bisa berjalan
- Model dievaluasi menggunakan **data test yang tidak pernah dilihat model** selama training
- Semua model menggunakan `class_weight='balanced'` untuk mengatasi ketidakseimbangan kelas
- Gradient Boosting menggunakan `sample_weight` karena tidak mendukung `class_weight` di constructor

---

## 5. Menu 2 — Pengujian Gambar

### Tujuan
Menguji satu gambar secara interaktif menggunakan model yang sudah dilatih.

### Cara Penggunaan

1. **Upload** satu gambar (JPG/JPEG) melalui file uploader
2. Sistem otomatis menampilkan **4 tahap preprocessing** pada gambar tersebut beserta caption penjelasan
3. **Pilih model** dari dropdown (hanya model yang sudah dilatih yang muncul)
4. Sistem menampilkan informasi model: mode fitur yang digunakan, daftar fitur
5. **Tabel nilai fitur** — semua nilai numerik yang diekstrak dari gambar ditampilkan dalam 2 kolom
6. Hasil prediksi muncul dalam **kartu besar** berwarna:
   - 🟢 Hijau = Renggang
   - 🟡 Kuning/Oranye = Sedang
   - 🔴 Merah = Padat
7. Di bawah kartu prediksi ditampilkan:
   - **Metrik evaluasi model** (Accuracy, Precision, Recall, F1-Score, Val Accuracy) dari data test training
   - **Confusion Matrix** interaktif dengan caption penjelasan cara membaca
8. Ganti model di dropdown untuk **membandingkan prediksi** dari model berbeda pada gambar yang sama

### Expander "Bandingkan Prediksi Semua Model"
Klik untuk melihat prediksi semua model yang tersedia sekaligus dalam satu tabel (Model · Prediksi · Accuracy).

### ⚠️ Catatan
- Confusion Matrix yang ditampilkan adalah dari **test split training** (bukan evaluasi gambar yang diupload)
- Metrik evaluasi tersimpan di file `.joblib` saat training, bukan dihitung ulang dari gambar ini

---

## 6. Menu 3 — Eksperimen Parameter

### Tujuan
Menganalisis pengaruh perubahan parameter terhadap performa klasifikasi. Cocok untuk analisis mendalam pada skripsi.

### Sumber Data
Eksperimen menggunakan **dataset CSV penuh** (`Data_ekstraksi_Fitur_Gulma.csv` — 2.097 sampel) sehingga tidak perlu upload gambar ulang.

### Cara Penggunaan

1. **Pilih Mode Fitur:**
   - `19 Fitur (tanpa GLCM)` — semua 19 fitur langsung
   - `39 Fitur (dengan GLCM + IG)` — 39 fitur, dipilih 14 terbaik via Information Gain

2. **Pilih Model** dari dropdown

3. **Pilih nilai parameter** yang ingin diuji (bisa pilih lebih dari satu nilai sekaligus):

| Model | Parameter | Nilai yang Tersedia |
|-------|-----------|---------------------|
| Logistic Regression | max_iter | 100, 300, 500, 700, 1000 |
| SVM | kernel | linear, rbf, poly |
| Decision Tree | max_depth | 3, 5, 7, 9, 11 |
| Random Forest | n_estimators | 100, 200, 300, 400, 500 |
| Gradient Boosting | n_estimators + learning_rate | kombinasi keduanya |

4. Klik **"🚀 Jalankan Eksperimen"**

5. **Progress real-time** ditampilkan dalam status box:
   - `⏳ Melatih [model] — [parameter] = [nilai]` saat proses berjalan
   - `✅ Selesai — [parameter] = [nilai] | Accuracy: X.XXXX` setelah tiap iterasi
   - `✅ Semua eksperimen selesai!` ketika semua kombinasi selesai

6. **Hasil yang ditampilkan:**
   - **Tabel metrik** semua konfigurasi parameter (baris terbaik disorot hijau)
   - **Grafik** metrik vs parameter — line chart (atau bar chart untuk Gradient Boosting)
   - Caption penjelasan cara membaca grafik
   - **Confusion Matrix** untuk konfigurasi parameter terbaik
   - **Classification Report** detail per kelas (dalam expander)

7. Klik **Jalankan Eksperimen** lagi dengan nilai berbeda untuk menambah hasil (hasil **terakumulasi**)

8. Klik **"🗑️ Hapus Riwayat Eksperimen Model Ini"** untuk membersihkan hasil model tersebut

### Tips Penggunaan Eksperimen
- Jalankan eksperimen untuk **semua model** agar bisa dibandingkan di Dashboard
- Coba berbagai nilai parameter untuk menemukan konfigurasi terbaik
- Hasil semua eksperimen tersimpan di session dan muncul di Dashboard Hasil
- Jika browser di-refresh, hasil eksperimen hilang — simpan screenshot sebelum refresh

---

## 7. Menu 4 — Dashboard Hasil

### Tujuan
Rekap menyeluruh seluruh hasil sistem dalam satu halaman — cocok untuk laporan, presentasi, atau sidang skripsi.

### Konten Dashboard

#### Seksi 1 — Rekap Preprocessing
- Infografis 4 tahap preprocessing dengan nama dan penjelasan teks per tahap
- Visualisasi contoh hasil preprocessing (muncul jika training pernah dilakukan di sesi ini)

#### Seksi 2 — Perbandingan Semua Model Terlatih
- **Tabel perbandingan** Accuracy, Precision, Recall, F1-Score, Val Accuracy, Execution Time
- Baris **model terbaik** disorot hijau secara otomatis
- **Badge** menampilkan nama model terbaik dan akurasinya
- **Bar chart interaktif** perbandingan 4 metrik semua model dengan caption penjelasan
- **Confusion Matrix semua model** dalam expander (dengan caption per chart)

#### Seksi 3 — Rekap Semua Hasil Eksperimen Parameter
- Tabel gabungan seluruh hasil eksperimen dari sesi ini
- **Line chart** per model: tren Accuracy & F1-Score vs nilai parameter, dengan caption penjelasan

#### Seksi 4 — Penjelasan Sistem (untuk Pembaca Umum)
Klik masing-masing expander untuk membaca penjelasan non-teknis:
- Apa itu Klasifikasi Kepadatan Gulma?
- Apa itu HSV Thresholding?
- Apa itu GLCM dan Information Gain?
- Apa itu Gradient Boosting?

---

## 8. Alur Penggunaan yang Disarankan

### Untuk Pertama Kali (Setelah Clone)

```
1. Menu Alur Pelatihan
   → Upload dataset (min. 9 gambar/kelas, disarankan ≥ 50/kelas)
   → Pahami visualisasi preprocessing (Langkah 2)
   → Pilih mode fitur 39 Fitur (GLCM)
   → Klik Jalankan Ekstraksi — baca grafik IG dan tabel fitur terpilih
   → Latih SEMUA 5 model satu per satu (Langkah 4)
   → Semua model tersimpan di models/

2. Menu Pengujian Gambar
   → Upload 1 gambar baru yang belum pernah dipakai
   → Coba semua model, bandingkan prediksi dan metriknya

3. Menu Eksperimen Parameter
   → Jalankan eksperimen untuk setiap model dengan berbagai nilai parameter
   → Catat parameter terbaik per model

4. Menu Dashboard Hasil
   → Lihat rekap semua hasil dalam satu halaman
   → Gunakan untuk bahan laporan dan presentasi sidang
```

### Untuk Sesi Lanjutan (Model Sudah Ada)
Model tersimpan di file `.joblib` — **tidak perlu training ulang**.  
Langsung bisa ke **Menu Pengujian Gambar** atau **Eksperimen Parameter**.

### Untuk Presentasi / Sidang
Buka **Menu Dashboard Hasil** — semua informasi sudah terekap lengkap di sana.

---

## 9. Penjelasan Istilah Teknis

| Istilah | Penjelasan Sederhana |
|---------|---------------------|
| **Preprocessing** | Serangkaian langkah pengolahan gambar sebelum fitur diekstrak |
| **Gaussian Blur** | Teknik menghaluskan gambar untuk mengurangi gangguan/noise |
| **HSV** | Model warna berbasis Hue (warna), Saturation (kepekatan), Value (kecerahan) |
| **Thresholding** | Memilah piksel berdasarkan rentang nilai tertentu |
| **Segmentasi** | Proses memisahkan area gulma dari latar belakang gambar |
| **Morphological Closing** | Mengisi lubang kecil pada area yang tersegmentasi |
| **GLCM** | Gray-Level Co-occurrence Matrix — matriks untuk mengukur tekstur gambar |
| **Fitur** | Nilai numerik yang merepresentasikan karakteristik satu gambar |
| **Information Gain (LAN)** | Metode memilih fitur paling informatif — mengukur seberapa besar fitur membantu klasifikasi |
| **SelectKBest** | Algoritma seleksi fitur yang memilih K fitur dengan skor IG tertinggi |
| **StandardScaler** | Normalisasi data agar semua fitur berada pada skala yang sama (z-score) |
| **Stratified Split** | Pembagian data yang mempertahankan proporsi tiap kelas di setiap subset |
| **Accuracy** | Persentase prediksi benar dari total prediksi |
| **Precision** | Dari yang diprediksi kelas X, berapa persen yang benar-benar kelas X |
| **Recall** | Dari yang sebenarnya kelas X, berapa persen yang berhasil terdeteksi |
| **F1-Score** | Rata-rata harmonis Precision dan Recall — metrik seimbang |
| **Val Accuracy** | Akurasi pada data validasi — digunakan untuk memantau overfitting |
| **Confusion Matrix** | Tabel yang menunjukkan prediksi benar vs salah per kelas secara rinci |
| **Classification Report** | Laporan lengkap Precision, Recall, F1-Score per kelas |
| **Overfitting** | Model terlalu hafal data training sehingga performa buruk di data baru |
| **class_weight='balanced'** | Memberi bobot lebih pada kelas minoritas untuk dataset tidak seimbang |
| **sample_weight** | Bobot per sampel, digunakan pada GradientBoosting sebagai pengganti class_weight |
| **Ensemble** | Menggabungkan banyak model untuk menghasilkan prediksi lebih akurat |
| **Gradient Boosting** | Ensemble yang membangun pohon secara bertahap — tiap pohon memperbaiki error sebelumnya |
| **Random Forest** | Ensemble dari banyak pohon keputusan yang dibuat secara acak dan independen |
| **n_estimators** | Jumlah pohon dalam model ensemble |
| **learning_rate** | Seberapa besar kontribusi tiap pohon baru dalam Gradient Boosting |
| **max_depth** | Kedalaman maksimum sebuah pohon keputusan |
| **Hu Moments** | 7 deskriptor bentuk objek yang invariant terhadap rotasi, skala, dan translasi |

---

## 10. Troubleshooting (Masalah & Solusi)

### ❌ "Belum ada model yang tersedia"
**Penyebab:** File `.joblib` model belum ada — pertama kali setelah clone, atau model terhapus.  
**Solusi:** Buka menu **📚 Alur Pelatihan** dan latih minimal satu model.

---

### ❌ "Setiap kelas membutuhkan minimal 9 gambar"
**Penyebab:** Jumlah gambar yang diupload kurang dari 9 per kelas. Stratified split 80:10:10 membutuhkan minimal 9 gambar per kelas (27 total) agar setiap subset mendapat setidaknya 1 sampel per kelas.  
**Solusi:** Tambah gambar hingga setiap kelas memiliki minimal **9 gambar**, lalu ulangi dari Langkah 1.

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
3. Pastikan gambar tidak korup/rusak

---

### ❌ Training sangat lambat (lebih dari 5 menit)
**Penyebab:** Dataset besar atau model Random Forest/Gradient Boosting memerlukan waktu komputasi panjang.  
**Solusi:**
- Ini normal — Random Forest (500 pohon) dan Gradient Boosting (300 iterasi) memang lebih lambat dari DT/LR/SVM
- Tunggu hingga selesai — progress ditampilkan di spinner
- Jika ingin lebih cepat untuk percobaan awal, gunakan Decision Tree atau Logistic Regression terlebih dahulu

---

### ❌ Prediksi salah / tidak sesuai ekspektasi
**Penyebab:** Model dilatih dengan dataset yang kurang representatif atau terlalu sedikit.  
**Solusi:**
1. Tambah jumlah gambar training per kelas (disarankan ≥ 50 gambar/kelas)
2. Pastikan gambar training bervariasi (berbagai kondisi pencahayaan, sudut, jarak)
3. Coba model lain — Gradient Boosting biasanya paling akurat
4. Pastikan gambar uji kondisinya mirip dengan gambar training

---

### ❌ Refresh browser menghilangkan hasil eksperimen
**Penyebab:** Hasil eksperimen disimpan di session state (memori tab browser), bukan di file permanen.  
**Solusi:** Simpan screenshot hasil eksperimen sebelum refresh, atau jangan tutup/refresh halaman selama sesi eksperimen berjalan.

---

### ❌ Error "ImportError" atau "ModuleNotFoundError"
**Penyebab:** Library Python belum terinstall.  
**Solusi:**
```bash
pip install -r requirements.txt
```

---

### ❌ Port 8501 sudah digunakan
**Penyebab:** Ada instance Streamlit lain yang sedang berjalan.  
**Solusi:**
```bash
streamlit run app.py --server.port 8502
```

---

### ❌ Teks tabel tidak terbaca di Dark Mode
**Penyebab:** Menggunakan versi lama aplikasi sebelum perbaikan dark mode.  
**Solusi:** Pastikan sudah `git pull origin main` untuk mendapatkan versi terbaru dengan dukungan dark mode penuh.

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
        └── GB.joblib             # Model Gradient Boosting ← Model Utama Penelitian
```

### Isi setiap file `.joblib`
Setiap file model menyimpan semua informasi yang dibutuhkan untuk inference:

| Key | Isi |
|-----|-----|
| `model` | Objek classifier yang sudah dilatih |
| `scaler` | StandardScaler yang sudah di-fit pada data training |
| `selector` | SelectKBest untuk seleksi IG (hanya ada jika mode 39 fitur) |
| `feature_mode` | `'19'` atau `'39'` |
| `features_used` | Daftar nama fitur yang digunakan model |
| `n_features` | Jumlah fitur input model (19 atau 14) |
| `metrics` | Accuracy, Precision, Recall, F1, Val Accuracy, Execution Time |
| `confusion_matrix` | Matriks konfusi + label kelas |
| `split_info` | Jumlah sampel train / validasi / test |

---

## Informasi Teknis Pipeline ML

### Alur Lengkap

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
Training Model (DT / LR / SVM / RF / GB)
    ↓
Evaluasi pada Test Split → Metrik & Confusion Matrix
    ↓
Simpan ke models/[NAMA].joblib (berisi model + scaler + selector + metadata)
```

### Konfigurasi Model Default (untuk Menu Alur Pelatihan)

| Model | Parameter Utama |
|-------|----------------|
| Decision Tree | `max_depth=3`, `class_weight='balanced'` |
| Logistic Regression | `solver='lbfgs'`, `max_iter=300`, `class_weight='balanced'` |
| SVM | `kernel='rbf'`, `C=5`, `gamma=0.01`, `class_weight='balanced'` |
| Random Forest | `n_estimators=500`, `class_weight='balanced'` |
| Gradient Boosting | `n_estimators=300`, `learning_rate=0.1`, `sample_weight` (balanced) |

---

*Panduan ini ditulis untuk Tugas Akhir Kholifah Dina — Universitas Telkom Purwokerto*  
*Sistem dikembangkan dengan Python, Streamlit, scikit-learn, dan OpenCV*
