# 🧠 Analisis Benchmark GPU untuk Hasil Cost-to-Performance Terbaik

Proyek ini bertujuan untuk membantu pengguna dalam memilih GPU terbaik berdasarkan rasio harga terhadap performa (Cost-to-Performance Ratio). Analisis dilakukan melalui teknik clustering, regresi, serta eksplorasi visual menggunakan Streamlit sebagai antarmuka interaktif.

## 🔍 Fitur Utama

- 🧩 **Clustering GPU** berdasarkan fitur performa dan harga menggunakan KMeans.
- 📈 **Prediksi performa GPU** dengan model regresi (Linear, Random Forest, Gradient Boosting).
- 📊 **Visualisasi interaktif** tren performa, distribusi harga, dan brand GPU.
- 🧮 **Rasio Price-to-Performance (PPR)** otomatis dihitung dan disorot.
- 🧵 **Aplikasi Streamlit** yang intuitif dan mudah digunakan.
- 📄 **Laporan PDF otomatis** untuk hasil analisis GPU.

## 📁 Struktur Proyek
```bash
/Analisis-Benchmark-Gpu-Untuk-Hasil-Cost-To-Performance-Terbaik
│
├── gpuApp.py                # Aplikasi Streamlit utama untuk analisis GPU
├── requirements.txt         # Daftar dependensi yang dibutuhkan
├── README.md                # Dokumentasi proyek
├── gpu_data_wPerformance.csv # Dataset GPU dengan performa
├── gpu_database_2015-2025.csv # Database GPU lengkap 2015-2025
├── gpu_2015.txt             # Data benchmark GPU tahun 2015
├── gpu_2016.txt             # Data benchmark GPU tahun 2016
├── gpu_2017.txt             # Data benchmark GPU tahun 2017
├── gpu_2018.txt             # Data benchmark GPU tahun 2018
├── gpu_2019.txt             # Data benchmark GPU tahun 2019
├── gpu_2020.txt             # Data benchmark GPU tahun 2020
├── gpu_2021.txt             # Data benchmark GPU tahun 2021
├── gpu_2022.txt             # Data benchmark GPU tahun 2022
├── gpu_2023.txt             # Data benchmark GPU tahun 2023
├── gpu_2024.txt             # Data benchmark GPU tahun 2024
├── gpu_2025.txt             # Data benchmark GPU tahun 2025
├── txt_to_csv.py            # Script untuk konversi data TXT ke CSV
```


## ⚙️ Cara Menjalankan Aplikasi

1. **Clone repository ini:**

   ```bash
   git clone https://github.com/muliarhmh/Analisis-Benchmark-Gpu-Untuk-Hasil-Cost-To-Performance-Terbaik.git
   cd Analisis-Benchmark-Gpu-Untuk-Hasil-Cost-To-Performance-Terbaik

2. **Aktifkan environment dan instal dependensi:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

3. **Jalankan aplikasi Streamlit:**
    ```bash
    streamlit run gpuApp.py

Aplikasi akan berjalan di:

   ```bash
   http://localhost:8501
   ```

### Panduan Menjalankan dan Penggunaan Aplikasi

Untuk panduan lebih lanjut, silakan kunjungi [Panduan Penggunaan Aplikasi](https://drive.google.com/file/d/1oAToemSQgtEncGh9Yio1LMNNjboyFfGZ/view?usp=drivesdk).

## 📌 Teknologi yang Digunakan

* Python 3.10+
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit
* st-aggrid
* fpdf

## 🎯 Tujuan Proyek

Proyek ini bertujuan untuk membantu pengguna, terutama gamer dan kreator konten, dalam memilih GPU yang paling efisien berdasarkan data historis performa dan harga. Proyek ini juga bermanfaat bagi analis pasar teknologi dan pemilik bisnis PC rakitan.

## 📢 Kontribusi

Kontribusi sangat terbuka! Silakan ajukan pull request atau buat issue jika Anda menemukan bug atau ingin menambahkan fitur baru.

## 📜 Lisensi

Proyek ini dilisensikan di bawah MIT License.

---

**Developed with ❤️ by Kelompok 8 Data Science StartUp Business**

---
