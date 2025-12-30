# Tabular ML Pipeline - Portofolio Hybrid Software Engineer + AI/ML

# Updated: December 31, 2025

## Tujuan Utama Proyek

Ini adalah portofolio pribadi yang menonjolkan kemampuan sebagai Software Engineer yang masuk ke dunia AI/ML. Fokus utama:

- Kode bersih, modular, dan mudah dipelihara (software engineering best practices)
- Pipeline end-to-end untuk data tabular (Kaggle Playground / kompetisi)
- Config-driven menggunakan YAML (mudah eksperimen tanpa ubah kode)
- Performa komputasi tinggi (bukan hanya akurasi model)
- Reproducible, testable, dan scalable

## Prioritas Teknis yang Sudah Diputuskan

1. **Format data**: Gunakan Parquet (snappy compression) di folder data/processed/

   - Alasan: load 5–20x lebih cepat, ukuran file jauh lebih kecil, columnar storage
   - Ada script terpisah: scripts/convert_to_parquet.py untuk konversi sekali dari CSV

2. **Data loading**:

   - src/data_loader.py harus support otomatis CSV atau Parquet berdasarkan ekstensi file
   - Selective column loading opsional di masa depan

3. **Optimasi performa komputasi** (fokus utama):

   - Downcast dtype otomatis (float64 → float32, int64 → int32)
   - OneHotEncoder dengan sparse=True
   - Semua operasi di feature engineering HARUS vectorized (Pandas/NumPy), TIDAK boleh pakai loop atau apply/lambda lambat
   - Parallel processing: n_jobs=-1 di CV dan model yang support
   - Prioritas model cepat: LightGBM > CatBoost > XGBoost
   - Parameter efisien: max_bin, subsample, early_stopping, dll

4. **Code quality**:

   - Wajib pakai type hints bawaan Python (from typing import Dict, List, Tuple, Any, Optional)
   - Tidak perlu mypy dulu (cukup Pylance di VSCode untuk warning)
   - Style: PEP8, docstrings jelas, nama variabel deskriptif
   - Belum pakai linter eksternal (Ruff/Black) tapi siap ditambah nanti

5. **Notebooks**:

   - Hanya 3 notebook utama:
     - 01_eda.ipynb → EDA + data checking
     - 02_feature_engineering.ipynb → Eksperimen fitur + validasi visual
     - 03_experiment.ipynb → Baseline, tuning, model comparison
   - Folder archive/ untuk versi lama

6. **Testing**:

   - Unit tests dengan pytest di folder tests/
   - Sudah direncanakan: test_config_loader, test_data_loader (termasuk Parquet), test_preprocessor, test_feature_engineering, test_utils

7. **Struktur yang sudah final**:
   - Semua config di folder config/ (termasuk default.yaml)
   - local.yaml untuk override pribadi (di-.gitignore)
   - src/ sebagai Python package (dengan **init**.py)
   - experiments/ untuk output per run
   - main.py sebagai entry point dengan argparse dan timing total execution

## Gaya Coding yang Diharapkan

- Modular: setiap tahap pipeline di file terpisah di src/
- Configurable: hampir semua hal diatur lewat YAML
- Defensive: error handling yang baik, logging informatif
- Performant: hindari operasi lambat, prioritaskan vectorized dan memory-efficient
- Readable: docstring, type hints, comment secukupnya

## Hal yang Belum Dijalankan (Fase Lanjutan - Opsional)

- Hyperparameter tuning dengan Optuna
- Experiment tracking dengan MLflow
- Ensemble/stacking
- Caching preprocessor dengan joblib
- GPU support
- Docker deployment

Copilot, tolong bantu saya menulis kode yang sesuai dengan semua konteks di atas.
Prioritaskan kebersihan kode, performa komputasi, dan kemudahan eksperimen melalui config.
