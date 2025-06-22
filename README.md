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

## Panduan Menjalankan Aplikasi
https://drive.google.com/file/d/1oAToemSQgtEncGh9Yio1LMNNjboyFfGZ/view?usp=drivesdk

📌 Teknologi yang Digunakan

- Python 3.10+
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- st-aggrid
- fpdf

🎯 Tujuan Proyek
Membantu pengguna, terutama gamer dan kreator konten, untuk memilih GPU yang paling efisien berdasarkan data historis performa dan harga. Proyek ini juga bermanfaat bagi analis pasar teknologi dan pemilik bisnis PC rakitan.

📢 Kontribusi
Kontribusi sangat terbuka! Silakan ajukan pull request atau buat issue jika Anda menemukan bug atau ingin menambahkan fitur baru.

📜 Lisensi
Proyek ini dilisensikan di bawah MIT License.

Developed with ❤️ by Kelompok 8 Data Science StartUp Bussiness
