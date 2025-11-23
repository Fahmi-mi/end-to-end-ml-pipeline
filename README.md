# ML Project Template

## 1. Setup & Clone

```bash
git clone https://github.com/Fahmi-mi/ml-base-template.git temp
cp -r temp/* .
rm -rf temp
```

Untuk menghapus semua file `.gitkeep` secara rekursif di seluruh direktori, jalankan perintah berikut:

```powershell
Get-ChildItem -Recurse -Filter ".gitkeep" | Remove-Item -Force
```

```bash
name . ".gitkeep" -type f -delete
```

## 2. MLflow Tracking

```python
mlflow.set_tracking_uri("file:///c:/Documents/Fahmi/ai/ml/nama-folder-root/logs")
```

```bash
mlflow ui --backend-store-uri file:///c:/Documents/Fahmi/ai/ml/nama-folder-root/logs
```

## 3. Panduan Exploratory Data Analysis (EDA)

1. **Analisis Missing Values**  
   Tampilkan jumlah dan persentase missing values per kolom, serta visualisasinya (misal dengan heatmap atau barplot).

2. **Analisis Distribusi Target**  
   Jika ada kolom target (misal Personality), tampilkan distribusinya (countplot/barplot).

3. **Outlier Analysis**  
   Deteksi dan visualisasi outlier pada fitur numerik, misal dengan boxplot.

4. **Analisis Hubungan Fitur dengan Target**

   - Numerical vs Target: Boxplot/violinplot setiap fitur numerik terhadap target.
   - Categorical vs Target: Crosstab/stacked barplot fitur kategorikal terhadap target.

5. **Pairplot/Scatterplot Matrix**  
   Visualisasi hubungan antar fitur numerik (menggunakan sns.pairplot).

6. **Analisis Multikolinearitas**  
   Tampilkan fitur-fitur yang sangat berkorelasi (multikolinearitas tinggi).

7. **Analisis Skewness & Kurtosis**  
   Tampilkan nilai skewness dan kurtosis untuk fitur numerik.

8. **Analisis Unik dan Frekuensi Kategori**  
   Tampilkan jumlah kategori unik dan frekuensi tertinggi pada fitur kategorikal.

9. **Analisis Nilai Konstanta**  
   Cek apakah ada kolom yang hanya berisi satu nilai (kurang informatif).
