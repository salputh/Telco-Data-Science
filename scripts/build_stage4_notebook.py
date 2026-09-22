"""
One-off helper script to generate notebooks/04_feature_engineering.ipynb
for Tahap 4 (Feature Engineering).

Run with the project's venv:
    .venv/bin/python scripts/build_stage4_notebook.py
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
"""# Tahap 4 — Feature Engineering
### Proyek: Prediksi Customer Churn pada Perusahaan Telekomunikasi

Lanjutan dari `03_eda.ipynb`. Tiga hal yang dikerjakan sesuai brief:

1. One-hot encoding kolom kategori.
2. Fitur turunan: rasio `TotalCharges / tenure` dan binning `tenure`
   (baru/menengah/lama).
3. Menangani class imbalance (`class_weight` **dan** SMOTE — dua-duanya
   disiapkan supaya bisa dibandingkan langsung di Tahap 5, sesuai saran
   brief untuk "latihan konsep penting ini").

### Catatan penting soal urutan train/test split

Brief menaruh train/test split resmi di Tahap 5. Tapi SMOTE **wajib**
di-fit hanya pada data training, tidak boleh pada seluruh dataset —
kalau di-fit sebelum split, oversampling bisa membuat baris sintetis yang
mirip data test bocor ke proses training (*data leakage*), dan skor model
di Tahap 6 jadi terlalu optimis / tidak valid.

Karena itu, split 80/20 dilakukan **di sini** (bukan didefinisikan ulang di
Tahap 5) sehingga SMOTE bisa didemonstrasikan dengan benar. Split yang sama
persis (`random_state=42`, stratified) akan dipakai lagi di Tahap 5 untuk
modeling & cross-validation — jadi tidak ada duplikasi keputusan, hanya
dipindah ke tempat yang teknisnya benar."""
))

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
cells.append(md("## 1. Setup & Load Data Bersih"))
cells.append(code(
"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

pd.set_option("display.max_columns", 40)

DATA_PATH = "../data/interim/telco_customer_churn_cleaned.csv"
OUT_DIR = "../data/processed"

df = pd.read_csv(DATA_PATH)
print(f"Shape awal: {df.shape}")
df.head()"""
))

# ---------------------------------------------------------------------------
# 2. Drop ID & encode target
# ---------------------------------------------------------------------------
cells.append(md(
"""## 2. Buang `customerID` & Encode Target

`customerID` bukan fitur prediktif (ID unik per baris), dibuang sesuai
catatan di brief. Target `Churn` diubah dari `Yes`/`No` ke `1`/`0` supaya
bisa dipakai langsung oleh model klasifikasi di Tahap 5."""
))
cells.append(code(
"""df = df.drop(columns=["customerID"])
df["Churn"] = (df["Churn"] == "Yes").astype(int)

print(df["Churn"].value_counts(normalize=True).round(4))"""
))

# ---------------------------------------------------------------------------
# 3. Derived features
# ---------------------------------------------------------------------------
cells.append(md(
"""## 3. Fitur Turunan

### 3.1 `avg_monthly_spend` = `TotalCharges / tenure`

Menangkap rata-rata pengeluaran bulanan riil pelanggan sepanjang masa
berlangganannya (beda dengan `MonthlyCharges` yang cuma tarif bulan
berjalan — bisa saja berubah kalau ada promo/upgrade paket). Untuk
`tenure = 0` (pembagian oleh nol), dipakai `MonthlyCharges` sebagai
pendekatan terbaik, karena itulah tagihan bulanan pertama yang akan mereka
bayar."""
))
cells.append(code(
"""df["avg_monthly_spend"] = np.where(
    df["tenure"] == 0,
    df["MonthlyCharges"],
    df["TotalCharges"] / df["tenure"],
)

df[["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend"]].head()"""
))

cells.append(md(
"""### 3.2 `tenure_group` — binning tenure jadi Baru/Menengah/Lama

Dari EDA (Tahap 3) terlihat churn menumpuk di tenure rendah. Bin dibuat per
kelipatan tahun supaya mudah diterjemahkan tim bisnis:

- **Baru**: 0–12 bulan (< 1 tahun)
- **Menengah**: 13–36 bulan (1–3 tahun)
- **Lama**: 37–72 bulan (> 3 tahun)"""
))
cells.append(code(
"""df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 36, 72],
    labels=["Baru", "Menengah", "Lama"],
)

tenure_group_churn = df.groupby("tenure_group", observed=True)["Churn"].mean() * 100
print("Churn rate per tenure_group (%):")
print(tenure_group_churn.round(2))"""
))

cells.append(md(
"Konsisten dengan histogram di Tahap 3 — churn rate kelompok **Baru** jauh "
"di atas **Menengah** dan **Lama**, mengonfirmasi bin ini punya sinyal yang "
"berguna buat model."
))

# ---------------------------------------------------------------------------
# 4. One-hot encoding
# ---------------------------------------------------------------------------
cells.append(md(
"""## 4. One-Hot Encoding Kolom Kategori

`pd.get_dummies(..., drop_first=True)` dipakai supaya tidak ada kolom
redundan (dummy variable trap) — untuk kolom biner seperti `gender`, cukup
satu kolom (`gender_Male`) yang sudah merepresentasikan keduanya."""
))
cells.append(code(
"""categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
print("Kolom kategori yang di-encode:")
print(categorical_cols)

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# get_dummies menghasilkan tipe bool untuk kolom baru -> jadikan 0/1 int
bool_cols = df_encoded.select_dtypes(include="bool").columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

print(f"\\nShape sebelum encoding: {df.shape}")
print(f"Shape sesudah encoding: {df_encoded.shape}")
df_encoded.head()"""
))

# ---------------------------------------------------------------------------
# 5. Train/test split
# ---------------------------------------------------------------------------
cells.append(md(
"""## 5. Train/Test Split (80/20, Stratified)

`stratify=y` memastikan proporsi churn (~27%) tetap sama persis di train
maupun test set — penting karena datanya imbalance, supaya test set tidak
kebetulan berisi churn rate yang menyimpang jauh dari populasi asli."""
))
cells.append(code(
"""X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")
print("\\nProporsi churn - train:")
print(y_train.value_counts(normalize=True).round(4))
print("\\nProporsi churn - test:")
print(y_test.value_counts(normalize=True).round(4))"""
))

# ---------------------------------------------------------------------------
# 6. Class imbalance
# ---------------------------------------------------------------------------
cells.append(md(
"""## 6. Menangani Class Imbalance

Dua pendekatan disiapkan supaya Tahap 5 bisa membandingkan langsung mana
yang memberi trade-off precision/recall terbaik untuk kasus ini.

### 6.1 Opsi A — `class_weight`

Tidak mengubah data sama sekali — hanya memberi bobot lebih besar ke kelas
minoritas (`Yes`/churn) saat model dilatih, supaya kesalahan pada kelas
churn "dihukum" lebih berat. Cocok dipakai langsung sebagai parameter
`LogisticRegression(class_weight=...)` atau `RandomForestClassifier(class_weight=...)`
di Tahap 5 — **tidak perlu file terpisah**, `X_train`/`y_train` asli tetap
dipakai."""
))
cells.append(code(
"""classes = np.array([0, 1])
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes.tolist(), weights.round(4)))

print("class_weight yang dihitung dari y_train:")
print(class_weight_dict)"""
))

cells.append(md(
"""### 6.2 Opsi B — SMOTE (Synthetic Minority Oversampling)

Membuat sampel sintetis kelas minoritas (churn) dengan interpolasi di
antara tetangga terdekatnya, sampai jumlah kelas `Yes` = jumlah kelas `No`
**di training set**. Di-fit **hanya pada `X_train`/`y_train`** — `X_test`
tidak disentuh sama sekali, supaya evaluasi di Tahap 6 tetap merepresentasikan
distribusi churn yang sesungguhnya di dunia nyata."""
))
cells.append(code(
"""smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Sebelum SMOTE - X_train: {X_train.shape}")
print(y_train.value_counts())
print(f"\\nSesudah SMOTE - X_train_smote: {X_train_smote.shape}")
print(y_train_smote.value_counts())"""
))

# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------
cells.append(md(
"""## 7. Simpan Hasil ke `data/processed/`

Disimpan sebagai beberapa file model-ready:

- `X_train.csv` / `y_train.csv` / `X_test.csv` / `y_test.csv` — split asli
  (imbalance tetap ~27%), dipakai untuk pendekatan `class_weight`.
- `X_train_smote.csv` / `y_train_smote.csv` — training set hasil SMOTE
  (50/50), `X_test`/`y_test` di atas tetap dipakai untuk evaluasi (tidak
  ada versi SMOTE untuk test set)."""
))
cells.append(code(
"""X_train.to_csv(f"{OUT_DIR}/X_train.csv", index=False)
X_test.to_csv(f"{OUT_DIR}/X_test.csv", index=False)
y_train.to_csv(f"{OUT_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{OUT_DIR}/y_test.csv", index=False)

X_train_smote.to_csv(f"{OUT_DIR}/X_train_smote.csv", index=False)
y_train_smote.to_csv(f"{OUT_DIR}/y_train_smote.csv", index=False)

print("Semua file tersimpan di", OUT_DIR)
import os
for f in sorted(os.listdir(OUT_DIR)):
    print(" -", f)"""
))

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
cells.append(md(
"""## 8. Ringkasan Tahap 4

**Yang sudah dilakukan:**

- `customerID` dibuang; target `Churn` diencode jadi `0`/`1`.
- 2 fitur turunan dibuat: `avg_monthly_spend` (TotalCharges/tenure, dengan
  penanganan tenure=0) dan `tenure_group` (Baru/Menengah/Lama).
- Semua kolom kategori di-one-hot-encode (`drop_first=True`), shape naik
  dari 20 kolom mentah jadi fitur biner yang siap dipakai model.
- Train/test split 80/20 (stratified, `random_state=42`) dilakukan di sini
  supaya SMOTE bisa diterapkan dengan benar (hanya di training set).
- Dua strategi class imbalance disiapkan: `class_weight` (dict siap pakai,
  tanpa ubah data) dan `SMOTE` (training set baru, 50/50).
- Semua output disimpan ke `data/processed/` sebagai file model-ready.

**Lanjut ke Tahap 5 — Modeling:**

1. Baseline: Logistic Regression, dicoba dengan `class_weight` dict di atas.
2. Model lanjutan: Random Forest & XGBoost/LightGBM, dibandingkan juga
   dengan versi yang dilatih di atas `X_train_smote`/`y_train_smote`.
3. Cross-validation pada training set untuk masing-masing strategi."""
))

nb["cells"] = cells

with open("notebooks/04_feature_engineering.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written to notebooks/04_feature_engineering.ipynb")
