# Project Brief: Prediksi Customer Churn Telco

## 1. Ringkasan Proyek

**Judul:** Prediksi Customer Churn pada Perusahaan Telekomunikasi

**Latar belakang bisnis:** Sebuah perusahaan telko kehilangan sebagian pelanggan setiap bulan (churn). Merekrut pelanggan baru jauh lebih mahal daripada mempertahankan pelanggan lama, sehingga tim bisnis ingin tahu pelanggan mana yang berisiko berhenti berlangganan agar tim retensi bisa menghubungi mereka lebih dulu dengan penawaran khusus.

**Pertanyaan yang harus dijawab proyek ini:**

1. Faktor apa saja yang paling berhubungan dengan pelanggan berhenti berlangganan (churn)?
2. Bisakah dibuat model yang memprediksi probabilitas seorang pelanggan akan churn?
3. Segmen pelanggan mana yang paling berisiko, dan rekomendasi bisnis apa yang bisa diberikan ke tim retensi?

Proyek ini dipilih karena mencakup hampir seluruh siklus kerja Data Scientist dalam satu proyek: eksplorasi data, cleaning, feature engineering, modeling klasifikasi, evaluasi dengan metrik yang relevan secara bisnis, sampai menerjemahkan hasil model ke rekomendasi yang bisa dieksekusi tim non-teknis.

## 2. Dataset

- **Nama:** Telco Customer Churn
- **Sumber:** Kaggle — blastchar/telco-customer-churn (dataset sampel resmi dari IBM, dipakai luas untuk latihan)
- **Ukuran:** 7.043 baris (1 baris = 1 pelanggan), 21 kolom, format CSV, sudah cukup bersih untuk pemula — cocok agar fokus belajar alur kerja lengkap, bukan tersendat di data yang sangat kotor.

**Daftar kolom:**

| Kolom | Tipe | Keterangan |
|---|---|---|
| customerID | teks | ID unik pelanggan (dibuang saat modeling) |
| gender | kategori | Male / Female |
| SeniorCitizen | biner | 0/1, apakah lansia |
| Partner | kategori | Punya pasangan (Yes/No) |
| Dependents | kategori | Punya tanggungan (Yes/No) |
| tenure | numerik | Lama berlangganan (bulan) |
| PhoneService | kategori | Pakai layanan telepon |
| MultipleLines | kategori | Punya banyak saluran telepon |
| InternetService | kategori | DSL / Fiber optic / No |
| OnlineSecurity | kategori | Layanan keamanan online |
| OnlineBackup | kategori | Layanan backup online |
| DeviceProtection | kategori | Proteksi perangkat |
| TechSupport | kategori | Dukungan teknis |
| StreamingTV | kategori | Layanan streaming TV |
| StreamingMovies | kategori | Layanan streaming film |
| Contract | kategori | Month-to-month / One year / Two year |
| PaperlessBilling | kategori | Tagihan tanpa kertas |
| PaymentMethod | kategori | Metode pembayaran |
| MonthlyCharges | numerik | Tagihan bulanan (USD) |
| TotalCharges | numerik | Total tagihan sejak jadi pelanggan |
| Churn | target | Yes/No — pelanggan berhenti atau tidak |

> **Catatan:** kolom `TotalCharges` di dataset asli tersimpan sebagai teks dan punya sedikit baris kosong (pelanggan baru dengan tenure 0) — ini justru bagian bagus untuk latihan data cleaning nyata.

## 3. Ruang Lingkup Kerja per Tahap

### Tahap 1 — Business & Data Understanding
- Baca konteks dataset, tentukan target (`Churn`), tulis ulang pertanyaan bisnis dengan kata-kata sendiri.
- Cek struktur data: `.info()`, `.describe()`, jumlah missing value, tipe tiap kolom.

### Tahap 2 — Data Cleaning
- Ubah `TotalCharges` dari teks ke numerik, tangani baris kosong (biasanya pelanggan dengan tenure = 0).
- Encode kolom biner `SeniorCitizen` agar konsisten dengan kolom Yes/No lain.
- Cek dan hapus duplikasi `customerID` bila ada.

### Tahap 3 — Exploratory Data Analysis (EDA)
- Distribusi target: berapa persen pelanggan churn vs tidak (biasanya data ini imbalance, sekitar 26% churn).
- Bandingkan churn rate terhadap `Contract`, `tenure`, `MonthlyCharges`, `InternetService` — cari pola mana yang paling terlihat.
- Buat minimal 4–6 visualisasi (bar chart churn per kategori, histogram tenure, boxplot MonthlyCharges vs churn, correlation heatmap untuk kolom numerik).

### Tahap 4 — Feature Engineering
- One-hot encoding untuk kolom kategori.
- Buat fitur turunan, misal rasio `TotalCharges / tenure`, atau kelompok tenure menjadi bin (baru/menengah/lama).
- Tangani class imbalance dengan `class_weight` atau oversampling (SMOTE) — sekaligus latihan konsep penting ini.

### Tahap 5 — Modeling
- Baseline: Logistic Regression (mudah diinterpretasi, sering jadi model perbandingan wajib).
- Model lanjutan: Random Forest dan XGBoost/LightGBM — bandingkan performanya dengan baseline.
- Gunakan train/test split (mis. 80/20) dan cross-validation.

### Tahap 6 — Evaluasi & Interpretasi
- Hitung precision, recall, F1-score, ROC-AUC — bukan hanya accuracy karena datanya imbalance.
- Lihat feature importance untuk menjawab pertanyaan bisnis "faktor apa yang paling memengaruhi churn".

### Tahap 7 — Komunikasi Hasil
- Ringkas temuan dalam bahasa bisnis: 3–5 insight utama + rekomendasi aksi untuk tim retensi.

## 4. Deliverable Konkret

- [ ] Notebook (Jupyter/Colab) berisi seluruh alur: cleaning → EDA → modeling → evaluasi, dengan komentar yang menjelaskan setiap keputusan (bukan hanya kode).
- [ ] Minimal 4–6 visualisasi EDA yang menjawab pertanyaan bisnis di bagian 1.
- [ ] Perbandingan minimal 2 model (baseline vs model lanjutan) dalam satu tabel metrik.
- [ ] Ringkasan 1 halaman ("executive summary") berisi insight utama dan rekomendasi bisnis, ditulis untuk pembaca non-teknis.
- [ ] Repository GitHub dengan README yang menjelaskan latar belakang, cara menjalankan kode, dan hasil utama — ini yang akan dilihat perekrut.
- [ ] (Opsional, nilai plus) Model dibungkus jadi aplikasi sederhana dengan Streamlit/Gradio, atau REST API dengan FastAPI, agar bisa dicoba orang lain secara interaktif.

## 5. Target Metrik & Cara Evaluasi Keberhasilan

Karena kelas Churn tidak seimbang (sekitar 73% tidak churn vs 27% churn), accuracy saja bisa menyesatkan (model yang selalu menjawab "tidak churn" pun akan terlihat akurat 73%). Fokus pada:

| Metrik | Kenapa relevan di kasus ini |
|---|---|
| Recall (kelas Churn) | Prioritas utama bisnis: sebisa mungkin menangkap pelanggan yang benar-benar akan churn agar bisa ditindaklanjuti |
| Precision (kelas Churn) | Menjaga agar tim retensi tidak menghubungi terlalu banyak pelanggan yang sebenarnya tidak akan churn (biaya kampanye) |
| F1-score | Keseimbangan precision & recall dalam satu angka untuk perbandingan model |
| ROC-AUC | Melihat kemampuan model membedakan churn vs tidak di berbagai threshold |

**Target realistis untuk proyek latihan:** ROC-AUC di atas 0.80 dan F1-score kelas churn di atas 0.55 sudah tergolong bagus untuk dataset ini — jangan mengejar accuracy setinggi mungkin, karena itu justru tanda model bias ke kelas mayoritas.

## 6. Estimasi Timeline & Tips Pengerjaan

| Minggu | Fokus |
|---|---|
| 1 | Data understanding, cleaning, EDA + visualisasi |
| 2 | Feature engineering, baseline model (Logistic Regression) |
| 3 | Model lanjutan (Random Forest/XGBoost), tuning, evaluasi & perbandingan |
| 4 | Ringkasan insight, README, rapikan repo GitHub, (opsional) deploy sederhana |

**Total estimasi:** 3–4 minggu dengan intensitas 1–2 jam per hari — realistis untuk dikerjakan sambil bekerja/kuliah.

**Tips:**
- Jangan lompat ke model kompleks sebelum baseline sederhana selesai — baseline adalah pembanding wajib.
- Tulis alasan di balik setiap keputusan (kenapa pilih metrik ini, kenapa buang kolom itu) langsung di notebook — ini yang membedakan portofolio kuat dari sekadar kode yang jalan.
- Simpan versi awal notebook di Git sejak hari pertama, jangan hanya commit di akhir.

## 7. Ide Pengembangan Lanjutan (Stretch Goal)

- **Explainability:** pakai SHAP atau LIME untuk menjelaskan prediksi tiap pelanggan secara individual, bukan cuma feature importance global.
- **Segmentasi:** tambahkan clustering (K-Means) di atas fitur pelanggan untuk menemukan segmen risiko, digabung dengan hasil klasifikasi churn.
- **Deployment:** bungkus model jadi API dengan FastAPI + Docker, atau dashboard interaktif dengan Streamlit yang menampilkan skor risiko churn per pelanggan.
- **Cost-sensitive analysis:** hitung estimasi biaya kampanye retensi vs potensi revenue yang diselamatkan, untuk menentukan threshold probabilitas optimal (bukan default 0.5).
- **Terapkan ke domain sendiri:** setelah alur ini dikuasai, pola yang sama (klasifikasi risiko + EDA + business recommendation) bisa dipakai untuk proyek dengan data dari domain yang sudah dikuasai, mis. prediksi performa/risiko di operasional peternakan, sebagai proyek portofolio yang lebih personal dan menonjol.
