"""
One-off helper script to generate notebooks/05_modeling.ipynb for Tahap 5
(Modeling).

Run with the project's venv:
    .venv/bin/python scripts/build_stage5_notebook.py
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
"""# Tahap 5 — Modeling
### Proyek: Prediksi Customer Churn pada Perusahaan Telekomunikasi

Lanjutan dari `04_feature_engineering.ipynb`. Yang dikerjakan sesuai brief:

1. Baseline: **Logistic Regression**.
2. Model lanjutan: **Random Forest** dan **XGBoost**.
3. Cross-validation pada training set untuk membandingkan model — evaluasi
   final di test set (confusion matrix, feature importance, dsb.) baru
   dilakukan di **Tahap 6**, notebook ini fokus ke *melatih & memilih*
   model.

Setiap algoritma dicoba dengan **dua strategi class imbalance** yang sudah
disiapkan di Tahap 4 (`class_weight` vs `SMOTE`), supaya kelihatan langsung
strategi mana yang lebih membantu masing-masing model — total **6 kombinasi
model** dibandingkan.

### Catatan penting: SMOTE di dalam cross-validation

Di Tahap 4, `X_train_smote`/`y_train_smote` sudah disimpan sebagai hasil
SMOTE pada **seluruh** training set. File itu valid untuk fit model final,
tapi **tidak boleh dipakai langsung untuk cross-validation** — kalau
oversampling dilakukan sebelum data dibagi ke fold CV, sampel sintetis di
satu fold bisa "meniru" tetangga yang kebetulan jatuh di fold lain,
sehingga skor CV jadi bias optimis (leakage antar-fold).

Solusi standarnya: bungkus SMOTE di dalam `imblearn.pipeline.Pipeline`
bersama classifier-nya, lalu jalankan cross-validation di atas
`X_train`/`y_train` **asli** (belum di-resample). Dengan begitu, resampling
otomatis dikerjakan ulang di dalam setiap fold, hanya pada bagian training
fold tersebut — validation fold tetap murni data asli."""
))

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
cells.append(md("## 1. Setup & Load Data Model-Ready"))
cells.append(code(
"""import pandas as pd
import numpy as np
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

DATA_DIR = "../data/processed"
MODELS_DIR = "../models"
REPORTS_DIR = "../reports"

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze("columns")
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze("columns")

print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")
print(f"Churn rate y_train: {y_train.mean():.4f}")"""
))

cells.append(md(
"""Catatan: notebook ini memakai `X_train`/`y_train` yang **belum** di-SMOTE
(imbalanced asli). Resampling untuk varian SMOTE dilakukan di dalam
pipeline, per fold, seperti dijelaskan di atas — bukan dari file
`X_train_smote.csv`."""
))

# ---------------------------------------------------------------------------
# 2. CV helper
# ---------------------------------------------------------------------------
cells.append(md(
"""## 2. Helper Cross-Validation

5-fold `StratifiedKFold` (menjaga proporsi churn di tiap fold) dengan 4
metrik: ROC-AUC (kemampuan membedakan churn di semua threshold — metrik
utama untuk memilih model), F1, Precision, Recall — semuanya dihitung untuk
kelas positif (`Churn = 1`), konsisten dengan fokus bisnis di brief."""
))
cells.append(code(
"""cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"roc_auc": "roc_auc", "f1": "f1", "precision": "precision", "recall": "recall"}


def run_cv(name, estimator, X=X_train, y=y_train):
    scores = cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=1)
    return {
        "model": name,
        "roc_auc_mean": scores["test_roc_auc"].mean(),
        "roc_auc_std": scores["test_roc_auc"].std(),
        "f1_mean": scores["test_f1"].mean(),
        "precision_mean": scores["test_precision"].mean(),
        "recall_mean": scores["test_recall"].mean(),
    }


cv_results = []"""
))

# ---------------------------------------------------------------------------
# 3. Baseline: Logistic Regression
# ---------------------------------------------------------------------------
cells.append(md(
"""## 3. Baseline — Logistic Regression

Fitur numerik di-scale (`StandardScaler`) karena Logistic Regression
sensitif terhadap skala fitur (mis. `TotalCharges` 0–8000 vs kolom
one-hot 0/1) — Random Forest & XGBoost tidak butuh ini karena mereka
berbasis split/threshold, bukan jarak/gradien linear."""
))
cells.append(code(
"""logreg_cw = ImbPipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
])
cv_results.append(run_cv("Logistic Regression (class_weight)", logreg_cw))

logreg_smote = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])
cv_results.append(run_cv("Logistic Regression (SMOTE)", logreg_smote))

pd.DataFrame(cv_results)"""
))

# ---------------------------------------------------------------------------
# 4. Random Forest
# ---------------------------------------------------------------------------
cells.append(md("## 4. Model Lanjutan — Random Forest"))
cells.append(code(
"""rf_cw = RandomForestClassifier(
    n_estimators=300, class_weight="balanced", random_state=42, max_depth=None,
)
cv_results.append(run_cv("Random Forest (class_weight)", rf_cw))

rf_smote = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
])
cv_results.append(run_cv("Random Forest (SMOTE)", rf_smote))

pd.DataFrame(cv_results)"""
))

# ---------------------------------------------------------------------------
# 5. XGBoost
# ---------------------------------------------------------------------------
cells.append(md(
"""## 5. Model Lanjutan — XGBoost

XGBoost punya parameter bawaan `scale_pos_weight` yang setara dengan
`class_weight` — rasio jumlah kelas negatif dibagi kelas positif, dihitung
langsung dari `y_train`."""
))
cells.append(code(
"""scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.4f}")

xgb_cw = XGBClassifier(
    n_estimators=300, random_state=42, eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
)
cv_results.append(run_cv("XGBoost (scale_pos_weight)", xgb_cw))

xgb_smote = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf", XGBClassifier(n_estimators=300, random_state=42, eval_metric="logloss")),
])
cv_results.append(run_cv("XGBoost (SMOTE)", xgb_smote))

pd.DataFrame(cv_results)"""
))

# ---------------------------------------------------------------------------
# 6. Comparison table
# ---------------------------------------------------------------------------
cells.append(md(
"""## 6. Tabel Perbandingan Cross-Validation

Diurutkan berdasarkan ROC-AUC (metrik utama pemilihan model — lihat brief
bagian 5: ROC-AUC & F1 kelas churn adalah target utama, bukan accuracy)."""
))
cells.append(code(
"""cv_summary = pd.DataFrame(cv_results).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
cv_summary = cv_summary.round(4)

os.makedirs(REPORTS_DIR, exist_ok=True)
cv_summary.to_csv(f"{REPORTS_DIR}/cv_model_comparison.csv", index=False)

cv_summary"""
))

cells.append(md(
"""**Cara membaca tabel ini:** `roc_auc_mean` menunjukkan model mana yang
paling baik membedakan pelanggan churn vs tidak secara umum;
`recall_mean` vs `precision_mean` menunjukkan trade-off yang relevan
secara bisnis — recall tinggi berarti model menangkap lebih banyak
pelanggan yang benar-benar churn (prioritas tim retensi), tapi biasanya
dengan konsekuensi precision lebih rendah (lebih banyak "false alarm").
Analisis lebih dalam soal trade-off ini, confusion matrix, dan feature
importance dilakukan di **Tahap 6**."""
))

# ---------------------------------------------------------------------------
# 7. Fit final models & save
# ---------------------------------------------------------------------------
cells.append(md(
"""## 7. Fit Model Final & Simpan ke `models/`

Semua 6 model di-fit ulang di atas **seluruh** `X_train`/`y_train` (bukan
cuma satu fold CV) dan disimpan dengan `joblib`, supaya Tahap 6 tinggal
memuat model yang sudah terlatih untuk evaluasi di `X_test` — tidak perlu
retrain dari nol."""
))
cells.append(code(
"""fitted_models = {
    "logreg_class_weight": logreg_cw,
    "logreg_smote": logreg_smote,
    "rf_class_weight": rf_cw,
    "rf_smote": rf_smote,
    "xgb_class_weight": xgb_cw,
    "xgb_smote": xgb_smote,
}

os.makedirs(MODELS_DIR, exist_ok=True)

for name, model in fitted_models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"{MODELS_DIR}/{name}.joblib")
    print(f"Saved: {MODELS_DIR}/{name}.joblib")"""
))

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
cells.append(md(
"""## 8. Ringkasan Tahap 5

**Yang sudah dilakukan:**

- 3 algoritma × 2 strategi class imbalance = **6 model** dilatih & dibandingkan:
  Logistic Regression, Random Forest, XGBoost — masing-masing dengan
  `class_weight`/`scale_pos_weight` dan dengan SMOTE.
- Evaluasi model memakai **5-fold cross-validation** (ROC-AUC, F1,
  Precision, Recall pada kelas churn), dengan SMOTE dijalankan **di dalam**
  pipeline per fold agar tidak bocor ke validation fold.
- Tabel perbandingan CV disimpan ke `reports/cv_model_comparison.csv`.
- Ke-6 model di-fit ulang di seluruh training set dan disimpan ke
  `models/*.joblib`.

**Lanjut ke Tahap 6 — Evaluasi & Interpretasi:**

1. Evaluasi model-model ini di `X_test`/`y_test` (data yang belum pernah
   dilihat sama sekali oleh proses training/CV manapun).
2. Confusion matrix, classification report, ROC curve.
3. Feature importance (Random Forest/XGBoost) & koefisien (Logistic
   Regression) untuk menjawab Pertanyaan Bisnis #1."""
))

nb["cells"] = cells

with open("notebooks/05_modeling.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written to notebooks/05_modeling.ipynb")
