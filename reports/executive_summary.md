# Ringkasan Eksekutif: Prediksi Customer Churn

**Proyek:** Prediksi Customer Churn — Perusahaan Telekomunikasi
**Tanggal:** 22 September 2026
**Data:** 7.043 pelanggan, periode data historis (IBM Telco Customer Churn dataset)

## Latar Belakang

Perusahaan kehilangan **26,5% pelanggan** setiap periode (*churn*). Karena biaya akuisisi pelanggan baru jauh lebih mahal daripada mempertahankan pelanggan lama, tim retensi butuh cara untuk mengetahui **siapa yang berisiko berhenti berlangganan sebelum mereka benar-benar pergi**, supaya bisa ditawari insentif lebih dulu.

Analisis ini menggabungkan eksplorasi data historis pelanggan dengan model prediksi risiko churn, untuk menjawab tiga pertanyaan: faktor apa yang paling berhubungan dengan churn, bisakah risiko itu diprediksi per pelanggan, dan segmen mana yang paling perlu diprioritaskan tim retensi.

## 5 Insight Utama

1. **Jenis kontrak adalah sinyal risiko terkuat.** Pelanggan dengan kontrak bulanan (*month-to-month*) churn di angka **42,7%** — jauh di atas kontrak 1 tahun (**11,3%**) dan kontrak 2 tahun (hanya **2,8%**). Kontrak jangka panjang terbukti menjadi "pengikat" loyalitas yang kuat.

2. **Risiko churn paling tinggi di awal masa berlangganan.** Pelanggan baru (0–12 bulan pertama) churn di angka **47,4%**, turun menjadi **25,5%** pada 13–36 bulan, dan hanya **11,9%** pada pelanggan yang sudah berlangganan lebih dari 3 tahun. Periode kritis untuk retensi ada di beberapa bulan pertama.

3. **Pelanggan fiber optic paling rentan di antara jenis layanan internet.** Churn rate pelanggan fiber optic mencapai **41,9%**, dibanding DSL (**19,0%**) dan pelanggan tanpa internet (**7,4%**) — mengindikasikan ada gap antara harga/kualitas yang dirasakan dengan ekspektasi pelanggan fiber.

4. **Dua sinyal tambahan yang memperkuat prediksi:** pembayaran via **electronic check** dan **tidak berlangganan layanan proteksi** (Online Security) keduanya berasosiasi dengan churn yang lebih tinggi — dua hal ini menambah presisi dalam menargetkan pelanggan berisiko di luar sekadar kontrak dan lama berlangganan.

5. **Model prediksi cukup andal untuk dipakai memprioritaskan kontak.** Model terbaik (Logistic Regression) berhasil menangkap **79% pelanggan yang benar-benar akan churn** (recall) dengan kemampuan membedakan yang baik (ROC-AUC 0,84). Namun dari pelanggan yang ditandai berisiko, hanya sekitar **separuh yang benar-benar akan churn** (precision 50%) — model ini cocok untuk **memprioritaskan daftar kontak**, bukan untuk keputusan final tanpa judgment tim retensi.

## Rekomendasi Aksi untuk Tim Retensi

| # | Aksi | Target Segmen |
|---|---|---|
| 1 | Tawarkan insentif migrasi ke kontrak 1–2 tahun (diskon, bonus kuota/layanan) | Pelanggan month-to-month, terutama yang sudah lewat 3 bulan pertama |
| 2 | Jalankan program onboarding/check-in aktif di 90 hari pertama, bukan menunggu tanda-tanda churn muncul | Pelanggan baru (tenure < 3 bulan) |
| 3 | Evaluasi ulang value proposition paket fiber optic; pertimbangkan bundling gratis/diskon Online Security & Tech Support | Pelanggan fiber optic dengan tagihan tinggi |
| 4 | Selidiki gesekan pada metode pembayaran electronic check; tawarkan insentif pindah ke autopay/kartu kredit | Pelanggan dengan payment method electronic check |
| 5 | Gunakan skor risiko model untuk menyusun daftar prioritas kontak tim retensi, bukan menghubungi seluruh basis pelanggan secara acak | Pelanggan dengan skor risiko tertinggi dari model |

## Keterbatasan & Langkah Lanjutan

- **Precision model masih ~50%** — separuh kontak berpotensi "meleset" dari target yang benar-benar akan churn. Untuk kampanye dengan biaya kontak tinggi, threshold keputusan model bisa disesuaikan (bukan default 50/50) berdasarkan estimasi biaya kampanye vs nilai pelanggan yang diselamatkan.
- Beberapa fitur (mis. tagihan bulanan) saling berkorelasi kuat dengan jenis layanan, sehingga arah pengaruhnya di model perlu dibaca hati-hati — insight di atas difokuskan pada pola yang **konsisten di berbagai sudut analisis**, bukan satu angka tunggal.
- Untuk memvalidasi dampak nyata dari rekomendasi (mis. insentif migrasi kontrak), disarankan uji coba terbatas (A/B test) sebelum rollout penuh ke seluruh basis pelanggan.
