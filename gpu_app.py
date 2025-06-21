import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Fungsi load data dengan pembersihan dan standarisasi Brand
def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)
    
    if 'Memory' in df.columns:
        df['Memory'] = df['Memory'].str.extract(r'(\d+)').astype(float)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Performance'] = pd.to_numeric(df['Performance'], errors='coerce')
    df = df.dropna(subset=['Price', 'Performance'])
    
    # Standarisasi kolom Brand menjadi uppercase dan perbaiki typo umum
    if 'Brand' in df.columns:
        df['Brand'] = df['Brand'].str.strip().str.upper()

        # Perbaikan typo brand umum (tambahkan jika ada typo lain)
        df['Brand'] = df['Brand'].replace({
            'INVIDIA': 'NVIDIA',
            'NVIDA': 'NVIDIA',
            'NIVIDIA': 'NVIDIA',
            'INTEL CORP': 'INTEL',
            'INTEL CORPORATION': 'INTEL',
            'AMD INC': 'AMD',
            'ADVANCED MICRO DEVICES': 'AMD',
            # Bisa tambahkan typo lain di sini
        })
    else:
        df['Brand'] = 'UNKNOWN'
        
    return df

# Clustering dengan n cluster
def perform_clustering(df, n_clusters=3):
    X = df[['Price', 'Performance']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters
    return df, kmeans

# Fungsi menambahkan label deskriptif untuk cluster
def assign_cluster_labels(df):
    cluster_label_map = {
        0: "Entry-level (murah, performa standar)",
        1: "Mid-range (harga dan performa seimbang)",
        2: "High-end (harga tinggi, performa maksimal)"
    }
    df['Cluster Label'] = df['Cluster'].map(cluster_label_map).fillna('Unknown')
    return df

# Regresi linear untuk prediksi performa
def train_regression_model(df):
    X_price = df[['Price']]
    y_perf = df['Performance']
    regressor = LinearRegression()
    regressor.fit(X_price, y_perf)
    return regressor

# Prediksi performa berdasarkan harga
def predict_performance(regressor, input_price):
    pred = regressor.predict(pd.DataFrame({'Price': [input_price]}))
    return pred[0]

# Rekomendasi GPU berdasarkan harga dan filter brand serta tipe GPU
def recommend_gpu(df, input_price, brand_filter=None, gpu_type_filter=None, tolerance=0.2):
    lower = input_price * (1 - tolerance)
    upper = input_price * (1 + tolerance)
    candidate_gpus = df[(df['Price'] >= lower) & (df['Price'] <= upper)]

    if brand_filter:
        candidate_gpus = candidate_gpus[candidate_gpus['Brand'].str.contains(brand_filter.upper(), na=False)]
    if gpu_type_filter:
        candidate_gpus = candidate_gpus[candidate_gpus['GPU Type'].str.contains(gpu_type_filter, case=False, na=False)]

    if candidate_gpus.empty:
        return None
    best_gpu = candidate_gpus.sort_values(['Performance', 'Price'], ascending=[False, True]).iloc[0]
    return best_gpu

# Visualisasi scatter plot dan analisis cluster
def show_cluster_analysis(df):
    st.subheader("Visualisasi Scatter Plot Cluster GPU")
    fig, ax = plt.subplots(figsize=(8,6))
    palette = sns.color_palette("tab10", n_colors=df['Cluster'].nunique())
    sns.scatterplot(
        data=df, x='Price', y='Performance', hue='Cluster',
        palette=palette, ax=ax, alpha=0.7, edgecolor=None
    )
    ax.set_title("Scatter Plot Harga vs Performa GPU dengan Cluster")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Statistik Ringkas Per Cluster")
    cluster_stats = df.groupby(['Cluster', 'Cluster Label']).agg(
        Jumlah_GPU = ('Product Name', 'count'),
        Rata_Rata_Harga = ('Price', 'mean'),
        Rata_Rata_Performa = ('Performance', 'mean')
    ).reset_index()

    st.dataframe(cluster_stats.style.format({
        'Rata_Rata_Harga': '{:,.0f}',
        'Rata_Rata_Performa': '{:.2f}'
    }))

# Visualisasi clustering plus titik prediksi performa dari harga input
def show_clustering_and_prediction(df, kmeans_model, regressor, input_price):
    pred_perf = predict_performance(regressor, input_price)

    fig, ax = plt.subplots(figsize=(10,6))
    palette = sns.color_palette("tab10", n_colors=df['Cluster'].nunique())
    sns.scatterplot(data=df, x='Price', y='Performance', hue='Cluster',
                    palette=palette, ax=ax, alpha=0.7, edgecolor=None)

    ax.scatter(input_price, pred_perf, color='black', s=150, marker='X', label='Input Harga + Prediksi Performa')

    ax.set_title('GPU Clustering dan Prediksi Performa dari Harga')
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

# Main Streamlit app
def main():
    st.title("Prediksi dan Rekomendasi GPU berdasarkan Harga")
    csv_path = st.text_input("Masukkan path file CSV GPU:", "Gpu_Data_wPerformance.csv")

    if csv_path:
        df = load_and_clean_data(csv_path)

        if 'GPU Type' not in df.columns:
            df['GPU Type'] = 'Desktop'  # Default jika kolom tidak ada

        # Cluster menggunakan 3 cluster sesuai permintaan
        df, kmeans_model = perform_clustering(df, n_clusters=3)

        # Tambah label cluster deskriptif
        df = assign_cluster_labels(df)

        regressor = train_regression_model(df)

        # Tampilkan visualisasi dan analisis cluster
        show_cluster_analysis(df)

        # Filter opsi untuk Brand dan GPU Type
        brands = [''] + sorted(df['Brand'].dropna().unique().tolist())
        gpu_types = [''] + sorted(df['GPU Type'].dropna().unique().tolist())

        selected_brand = st.selectbox("Filter Brand (optional)", brands)
        selected_gpu_type = st.selectbox("Filter GPU Type (optional)", gpu_types)

        input_price = st.number_input("Masukkan harga GPU yang ingin diketahui performanya:", min_value=0.0, step=100000.0)

        if st.button("Prediksi dan Rekomendasi"):
            if input_price <= 0:
                st.error("Harga harus lebih dari 0")
            else:
                pred_perf = predict_performance(regressor, input_price)
                st.write(f"Prediksi performa GPU untuk harga sekitar {input_price}: {pred_perf:.2f}")

                best_gpu = recommend_gpu(df, input_price,
                                         selected_brand if selected_brand else None,
                                         selected_gpu_type if selected_gpu_type else None)
                if best_gpu is not None:
                    st.write("Rekomendasi GPU terbaik di kisaran harga tersebut:")
                    st.write(f"Nama Produk: {best_gpu.get('Product Name', 'Nama produk tidak tersedia')}")
                    st.write(f"Brand: {best_gpu['Brand']}")
                    st.write(f"Harga: {best_gpu['Price']}")
                    st.write(f"Performa: {best_gpu['Performance']}")
                    st.write(f"GPU Type: {best_gpu.get('GPU Type', 'N/A')}")
                    st.write(f"Kategori Cluster: {best_gpu.get('Cluster Label', 'Tidak diketahui')}")
                else:
                    st.warning("Tidak ditemukan GPU dengan kriteria yang sesuai.")

                # Tampilkan visualisasi clustering + prediksi performa dari harga
                show_clustering_and_prediction(df, kmeans_model, regressor, input_price)

if __name__ == "__main__":
    main()
