"""
One-off helper script to generate notebooks/02_data_cleaning.ipynb
for Tahap 2 (Data Cleaning).

Run with the project's venv:
    .venv/bin/python scripts/build_stage2_notebook.py
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
"""# Tahap 2 — Data Cleaning
### Proyek: Prediksi Customer Churn pada Perusahaan Telekomunikasi

Lanjutan dari `01_business_data_understanding.ipynb`. Tiga temuan dari
Tahap 1 yang jadi fokus pembersihan di sini:

1. `TotalCharges` tersimpan sebagai teks dan punya **11 baris "kosong
   tersembunyi"** (string spasi) — semuanya pelanggan dengan `tenure = 0`.
2. `SeniorCitizen` disimpan sebagai `0`/`1`, tidak konsisten dengan kolom
   biner lain yang memakai `Yes`/`No`.
3. `customerID` perlu dipastikan tidak ada duplikat (di Tahap 1 hasilnya 0
   duplikat, tapi tetap kita cek ulang & tangani di sini secara eksplisit
   supaya notebook ini berdiri sendiri).

Output notebook ini adalah dataset yang sudah dibersihkan, disimpan ke
`data/interim/` (belum final untuk modeling — feature engineering di
Tahap 4 masih akan menambah kolom baru)."""
))

# ---------------------------------------------------------------------------
# 1. Setup & load
# ---------------------------------------------------------------------------
cells.append(md("## 1. Setup & Load Data Mentah"))
cells.append(code(
"""import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 25)
pd.set_option("display.width", 120)

RAW_PATH = "../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(RAW_PATH)

print(f"Shape awal: {df.shape}")
df.head()"""
))

# ---------------------------------------------------------------------------
# 2. TotalCharges
# ---------------------------------------------------------------------------
cells.append(md(
"""## 2. Membersihkan `TotalCharges`

### 2.1 Konversi ke numerik

`TotalCharges` dibaca sebagai `object` karena ada baris dengan nilai berupa
string kosong (`" "`). `pd.to_numeric(..., errors="coerce")` akan mengubah
baris tersebut jadi `NaN` sehingga bisa ditangani secara eksplisit."""
))
cells.append(code(
"""df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

n_missing = df["TotalCharges"].isnull().sum()
print(f"Jumlah TotalCharges yang jadi NaN setelah konversi: {n_missing}")
df.loc[df["TotalCharges"].isnull(), ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]]"""
))

cells.append(md(
"""### 2.2 Menangani baris kosong

Seperti dikonfirmasi di Tahap 1, seluruh baris yang kosong punya
`tenure = 0` — artinya pelanggan yang baru mendaftar dan belum pernah
ditagih sama sekali. Jadi nilai yang paling masuk akal secara bisnis untuk
`TotalCharges` mereka adalah **0**, bukan rata-rata/median dari pelanggan
lain (yang justru akan menyesatkan karena tidak merepresentasikan kondisi
pelanggan baru).

Alternatif `MonthlyCharges * tenure` juga menghasilkan 0 karena
`tenure = 0`, jadi kedua pendekatan sepakat pada nilai yang sama."""
))
cells.append(code(
"""df["TotalCharges"] = df["TotalCharges"].fillna(0)

assert df["TotalCharges"].isnull().sum() == 0
print("TotalCharges sekarang numerik dan tidak ada missing value.")
df["TotalCharges"].describe()"""
))

# ---------------------------------------------------------------------------
# 3. SeniorCitizen
# ---------------------------------------------------------------------------
cells.append(md(
"""## 3. Menyeragamkan `SeniorCitizen`

Kolom ini berisi `0`/`1`, sementara kolom biner lain (`Partner`,
`Dependents`, `PhoneService`, dst.) memakai `Yes`/`No`. Supaya konsisten
dan lebih mudah dibaca saat EDA, `SeniorCitizen` diubah jadi `Yes`/`No`."""
))
cells.append(code(
"""print("Sebelum:", df["SeniorCitizen"].unique())

df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

print("Setelah :", df["SeniorCitizen"].unique())
df["SeniorCitizen"].value_counts()"""
))

# ---------------------------------------------------------------------------
# 4. Duplicate check
# ---------------------------------------------------------------------------
cells.append(md(
"""## 4. Cek & Hapus Duplikasi `customerID`

Dicek ulang secara eksplisit di notebook ini (bukan hanya mengandalkan hasil
Tahap 1), dan baris duplikat (jika ada) dibuang, menyisakan kemunculan
pertama saja."""
))
cells.append(code(
"""n_before = len(df)
n_duplicate_ids = df["customerID"].duplicated().sum()
print(f"Jumlah baris sebelum        : {n_before}")
print(f"Jumlah customerID duplikat  : {n_duplicate_ids}")

if n_duplicate_ids > 0:
    df = df.drop_duplicates(subset="customerID", keep="first").reset_index(drop=True)

n_after = len(df)
print(f"Jumlah baris setelah        : {n_after}")
assert df["customerID"].is_unique"""
))

# ---------------------------------------------------------------------------
# 5. Final check
# ---------------------------------------------------------------------------
cells.append(md(
"""## 5. Pemeriksaan Akhir

Pastikan tidak ada missing value tersisa dan tipe data sudah sesuai
ekspektasi sebelum data disimpan."""
))
cells.append(code("df.info()"))
cells.append(code(
"""total_missing = df.isnull().sum().sum()
print(f"Total missing value di seluruh dataset: {total_missing}")
assert total_missing == 0"""
))

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
cells.append(md(
"""## 6. Simpan Data Bersih

Disimpan ke `data/interim/` (bukan `data/processed/`) karena ini baru hasil
*cleaning*, belum melalui feature engineering (encoding, fitur turunan,
penanganan class imbalance) yang jadi bagian Tahap 4."""
))
cells.append(code(
"""OUTPUT_PATH = "../data/interim/telco_customer_churn_cleaned.csv"
df.to_csv(OUTPUT_PATH, index=False)
print(f"Data bersih disimpan ke: {OUTPUT_PATH}")
print(f"Shape akhir: {df.shape}")"""
))

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
cells.append(md(
"""## 7. Ringkasan Tahap 2

**Yang sudah dilakukan:**

- `TotalCharges` dikonversi dari teks ke numerik; 11 baris kosong (semua
  `tenure = 0`) diisi dengan `0`.
- `SeniorCitizen` diseragamkan dari `0`/`1` menjadi `No`/`Yes` agar
  konsisten dengan kolom biner lain.
- Duplikasi `customerID` dicek ulang dan ditangani (hasil: tidak ada
  duplikat, jadi jumlah baris tetap 7.043).
- Dataset bersih disimpan ke `data/interim/telco_customer_churn_cleaned.csv`.

**Lanjut ke Tahap 3 — Exploratory Data Analysis (EDA):**

1. Distribusi target `Churn` (persentase Yes vs No).
2. Bandingkan churn rate terhadap `Contract`, `tenure`, `MonthlyCharges`,
   `InternetService`.
3. Buat minimal 4–6 visualisasi (bar chart, histogram, boxplot, correlation
   heatmap)."""
))

nb["cells"] = cells

with open("notebooks/02_data_cleaning.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written to notebooks/02_data_cleaning.ipynb")
