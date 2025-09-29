# Paper Folder

Folder ini berisi template LaTeX untuk menulis paper analisis machine learning dari kompetisi Kaggle.

## Struktur Folder

```
paper/
├── main.tex                  # File utama LaTeX (all-in-one)
├── images/                   # Folder untuk menyimpan gambar/plots
├── create_placeholders.py    # Script untuk membuat contoh gambar
├── README.md                 # File ini
└── main.pdf                  # Output PDF (setelah compile)
```

## Cara Menggunakan

1. **Edit main.tex**: Ganti placeholder seperti `[Competition Name]`, `[Your Name]`, dll.

2. **Tambahkan gambar**: Simpan plots dan visualisasi di folder `images/`

   - Format yang disarankan: PNG atau PDF
   - Contoh: `missing_values.png`, `feature_importance.png`

3. **Compile LaTeX**:

   ```bash
   cd paper
   pdflatex main.tex
   ```

4. **Buat placeholder gambar** (opsional untuk testing):
   ```bash
   python create_placeholders.py
   ```

## Template yang Sudah Fixed

✅ **No Error**: Template sudah diperbaiki dan bisa di-compile
✅ **Simple**: Hanya menggunakan packages essential
✅ **Commented Images**: Gambar di-comment agar tidak error saat belum ada file
✅ **Clean Structure**: Struktur sederhana dan mudah digunakan

## Tips

- **Gambar dari Python**:
  ```python
  plt.savefig('paper/images/plot_name.png', dpi=300, bbox_inches='tight')
  ```
- **Uncomment gambar**: Setelah ada file gambar, uncomment bagian figure di main.tex
- **Update sesuai kompetisi**: Sesuaikan sections dengan karakteristik kompetisi
- **Version control**: Commit paper bersama code untuk tracking progress

## Template Sections

Template sudah include:

- Abstract & Keywords
- Introduction & Problem Statement
- Dataset Description & EDA
- Methodology & Preprocessing
- Experiments & Results
- Conclusion & Future Work

Tinggal isi dengan hasil analisis Anda!
