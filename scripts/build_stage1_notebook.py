"""
One-off helper script to generate notebooks/01_business_data_understanding.ipynb
for Tahap 1 (Business & Data Understanding).

Run with the project's venv:
    .venv/bin/python scripts/build_stage1_notebook.py
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

# ---------------------------------------------------------------------------
# 0. Judul
# ---------------------------------------------------------------------------
cells.append(md(
"""# Tahap 1 — Business & Data Understanding
### Proyek: Prediksi Customer Churn pada Perusahaan Telekomunikasi

Notebook ini adalah langkah pertama dari alur kerja data science untuk proyek
prediksi churn pelanggan Telco. Fokus tahap ini **bukan** membersihkan data
atau membuat model, melainkan:

1. Memahami konteks bisnis dan menuliskan ulang pertanyaan bisnis dengan
   kata-kata sendiri.
2. Menentukan target prediksi.
3. Memeriksa struktur data mentah: tipe kolom, statistik deskriptif, dan
   jumlah missing value — sebagai dasar untuk perencanaan cleaning di
   Tahap 2."""
))

# ---------------------------------------------------------------------------
# 1. Business understanding
# ---------------------------------------------------------------------------
cells.append(md(
"""## 1. Konteks Bisnis (ditulis ulang dengan kata sendiri)

Perusahaan telko ini kehilangan pelanggan setiap bulan (*churn*). Biaya untuk
mendapatkan pelanggan baru jauh lebih mahal dibanding mempertahankan
pelanggan yang sudah ada, jadi kalau tim retensi tahu **lebih awal** siapa
saja pelanggan yang kemungkinan besar akan berhenti, mereka bisa proaktif
menawarkan promo/insentif sebelum pelanggan itu benar-benar pergi.

Jadi masalah bisnisnya bukan sekadar "prediksi angka", tapi: **siapa yang
harus dihubungi duluan oleh tim retensi, dan kenapa mereka berisiko pergi?**

## 2. Pertanyaan Bisnis (versi saya sendiri)

| # | Pertanyaan asli di brief | Versi saya |
|---|---|---|
| 1 | Faktor apa yang paling berhubungan dengan churn? | Karakteristik/perilaku pelanggan seperti apa yang paling sering muncul pada pelanggan yang berhenti — apakah soal jenis kontrak, lama berlangganan, biaya bulanan, atau jenis layanan yang dipakai? |
| 2 | Bisakah dibuat model prediksi probabilitas churn? | Bisakah kita membuat sistem skor risiko (0–1) per pelanggan, bukan cuma label Yes/No, supaya tim retensi bisa memprioritaskan pelanggan dengan skor risiko tertinggi? |
| 3 | Segmen mana yang paling berisiko & rekomendasi apa? | Kalau kita kelompokkan pelanggan berdasarkan kombinasi kontrak/tenure/layanan, kelompok mana yang churn rate-nya jauh di atas rata-rata, dan tindakan konkret apa yang bisa diambil tim retensi untuk tiap kelompok itu? |

## 3. Menentukan Target

- **Kolom target:** `Churn` (Yes / No).
- **Jenis masalah:** klasifikasi biner.
- **Kelas positif (yang ingin ditangkap):** `Yes` — pelanggan yang churn.
  Ini penting karena metrik yang dipakai nanti (recall, precision, F1 di
  Tahap 6) akan dihitung terhadap kelas `Yes`, bukan `No`.
- **Unit analisis:** 1 baris = 1 pelanggan (`customerID` sebagai ID, dibuang
  saat modeling karena tidak punya nilai prediktif)."""
))

# ---------------------------------------------------------------------------
# 2. Setup
# ---------------------------------------------------------------------------
cells.append(md("## 4. Setup & Load Data"))

cells.append(code(
"""import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 25)
pd.set_option("display.width", 120)

DATA_PATH = "../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(DATA_PATH)

print(f"Jumlah baris : {df.shape[0]}")
print(f"Jumlah kolom : {df.shape[1]}")"""
))

cells.append(md(
"Sesuai brief: dataset seharusnya berisi **7.043 baris** dan **21 kolom** — "
"dicek langsung di atas untuk memastikan file yang di-load memang benar."
))

cells.append(md("### Sekilas isi data"))
cells.append(code("df.head()"))

# ---------------------------------------------------------------------------
# 3. Struktur data
# ---------------------------------------------------------------------------
cells.append(md(
"""## 5. Struktur Data

Cek tipe data tiap kolom dan jumlah non-null value dengan `.info()`."""
))
cells.append(code("df.info()"))

cells.append(md(
"""**Catatan awal dari `.info()`:**

- Semua kolom terbaca sebagai `object` (teks) kecuali `SeniorCitizen`
  (int64), `tenure` (int64), dan `MonthlyCharges` (float64).
- `TotalCharges` **seharusnya numerik** tapi terbaca sebagai `object` — ini
  konsisten dengan catatan di brief bahwa kolom ini tersimpan sebagai teks
  dan punya baris kosong. Akan diselidiki lebih lanjut di bawah, dan
  diperbaiki di **Tahap 2 (Data Cleaning)**.
- Tidak ada kolom yang menunjukkan non-null count lebih kecil dari jumlah
  baris total, artinya secara eksplisit (`NaN`) tidak ada missing value —
  tapi ini belum tentu berarti datanya benar-benar lengkap (lihat poin
  `TotalCharges` di atas)."""
))

cells.append(md("## 6. Statistik Deskriptif"))
cells.append(md("### Kolom numerik"))
cells.append(code("df.describe()"))

cells.append(md(
"""Observasi cepat:

- `SeniorCitizen` walau numerik (0/1) sebenarnya kategori biner, bukan
  kuantitas — perlu ditangani secara berbeda saat feature engineering.
- `tenure` berkisar 0–72 bulan, dengan sebagian pelanggan tenure = 0
  (pelanggan baru) — ini yang nanti berkaitan dengan baris kosong di
  `TotalCharges`.
- `MonthlyCharges` berkisar sekitar 18–119 USD, distribusinya perlu dicek
  lebih lanjut saat EDA (Tahap 3)."""
))

cells.append(md("### Kolom kategorikal"))
cells.append(code("df.describe(include='object')"))

cells.append(md(
"""Observasi cepat:

- `customerID` punya jumlah `unique` yang perlu disamakan dengan jumlah
  baris — kalau lebih kecil, berarti ada `customerID` duplikat (dicek di
  bagian 8).
- Kolom-kolom layanan (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`,
  `TechSupport`, `StreamingTV`, `StreamingMovies`) kemungkinan punya nilai
  ketiga selain Yes/No, misalnya `"No internet service"` — perlu dicek
  kategori uniknya."""
))

cells.append(code(
"""service_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines",
]
for col in service_cols:
    print(f"{col}: {sorted(df[col].unique())}")"""
))

# ---------------------------------------------------------------------------
# 4. Missing values
# ---------------------------------------------------------------------------
cells.append(md("## 7. Missing Value per Kolom"))
cells.append(code(
"""missing = pd.DataFrame({
    "n_missing": df.isnull().sum(),
    "pct_missing": (df.isnull().sum() / len(df) * 100).round(2),
})
missing.sort_values("n_missing", ascending=False)"""
))

cells.append(md(
"""Secara eksplisit **tidak ada `NaN`** di kolom manapun. Tapi seperti
disinggung di brief, `TotalCharges` disimpan sebagai teks dan punya
**baris kosong berupa string spasi (`" "`)**, bukan `NaN` — sehingga tidak
terdeteksi oleh `.isnull()`. Ini dicek langsung di bawah."""
))

cells.append(code(
"""# Coba ubah TotalCharges ke numerik; baris yang gagal dikonversi akan jadi NaN
total_charges_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
hidden_missing_mask = total_charges_numeric.isna()

print(f"Jumlah baris dengan TotalCharges tidak valid: {hidden_missing_mask.sum()}")
df.loc[hidden_missing_mask, ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]]"""
))

cells.append(md(
"""**Temuan penting:** seluruh baris dengan `TotalCharges` tidak valid ternyata
punya `tenure = 0`, yaitu pelanggan yang baru saja mendaftar sehingga belum
punya total tagihan. Ini masuk akal secara bisnis (bukan data error acak),
dan akan ditangani secara eksplisit di **Tahap 2** (opsi: isi dengan 0, atau
`MonthlyCharges * tenure`)."""
))

# ---------------------------------------------------------------------------
# 5. Duplicate check
# ---------------------------------------------------------------------------
cells.append(md("## 8. Cek Duplikasi `customerID`"))
cells.append(code(
"""n_rows = len(df)
n_unique_ids = df["customerID"].nunique()
n_duplicate_ids = df["customerID"].duplicated().sum()

print(f"Jumlah baris          : {n_rows}")
print(f"Jumlah customerID unik: {n_unique_ids}")
print(f"Jumlah ID duplikat    : {n_duplicate_ids}")"""
))

cells.append(md(
"Jika `n_duplicate_ids == 0`, berarti setiap baris memang mewakili satu "
"pelanggan unik dan tidak perlu deduplikasi di Tahap 2."
))

# ---------------------------------------------------------------------------
# 6. Target distribution
# ---------------------------------------------------------------------------
cells.append(md("## 9. Distribusi Target (`Churn`)"))
cells.append(code(
"""churn_dist = pd.DataFrame({
    "count": df["Churn"].value_counts(),
    "percentage": (df["Churn"].value_counts(normalize=True) * 100).round(2),
})
churn_dist"""
))

cells.append(md(
"""Sesuai perkiraan di brief, distribusi target **tidak seimbang** (sekitar
73% `No` vs 27% `Yes`). Konsekuensinya sudah jelas untuk tahap-tahap
berikutnya:

- **Tahap 4 (Feature Engineering):** perlu strategi menangani imbalance
  (`class_weight` atau SMOTE).
- **Tahap 6 (Evaluasi):** accuracy saja tidak cukup — fokus ke recall,
  precision, F1-score, dan ROC-AUC pada kelas `Yes` (churn)."""
))

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
cells.append(md(
"""## 10. Ringkasan Tahap 1

**Yang sudah dipastikan:**

- Dataset punya 7.043 baris × 21 kolom, 1 baris = 1 pelanggan.
- Target: kolom `Churn` (Yes/No), kelas positif = `Yes`, masalah klasifikasi
  biner dengan distribusi imbalance (~27% churn).
- Tidak ada `NaN` eksplisit, tapi `TotalCharges` punya 11 baris "kosong
  tersembunyi" (string spasi) yang semuanya berasal dari pelanggan dengan
  `tenure = 0`.
- `TotalCharges` perlu dikonversi dari teks ke numerik.
- `SeniorCitizen` disimpan sebagai 0/1, sementara kolom biner lain pakai
  Yes/No — perlu diseragamkan.
- Beberapa kolom layanan punya kategori ketiga (`"No internet service"` /
  `"No phone service"`) selain Yes/No.
- `customerID` dicek keunikannya (lihat hasil bagian 8) — dibuang saat
  modeling karena bukan fitur prediktif.

**Lanjut ke Tahap 2 — Data Cleaning:**

1. Konversi `TotalCharges` ke numerik, isi/atasi 11 baris kosong (tenure=0).
2. Seragamkan `SeniorCitizen` menjadi format Yes/No (atau sebaliknya).
3. Pastikan tidak ada duplikasi `customerID` (sudah dicek di atas, tinggal
   ditindaklanjuti kalau ternyata >0)."""
))

nb["cells"] = cells

with open("notebooks/01_business_data_understanding.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written to notebooks/01_business_data_understanding.ipynb")
