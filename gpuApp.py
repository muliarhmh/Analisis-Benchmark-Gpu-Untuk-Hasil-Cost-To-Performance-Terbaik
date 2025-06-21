# GPU Insight & Analytic App with Enhanced Features (Final Version)
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from st_aggrid import AgGrid, GridOptionsBuilder
from fpdf import FPDF
import base64

# Load dan bersihkan data
def load_and_clean_data(csv_file):
    df = pd.read_csv(csv_file)
    df['Memory'] = df['Memory'].str.extract(r'(\d+)').astype(float)
    df['Bus Width'] = df['Bus'].str.extract(r'x(\d+)').astype(float)
    df['GPU clock'] = df['GPU clock'].str.extract(r'(\d+)').astype(float)
    df['Memory clock'] = df['Memory clock'].str.extract(r'(\d+)').astype(float)
    df['Shaders / TMUs / ROPs'] = df['Shaders / TMUs / ROPs'].str.extract(r'(\d+)').astype(float)
    df['Released'] = pd.to_datetime(df['Released'], errors='coerce')
    df['Release Year'] = df['Released'].dt.year
    df['Brand'] = df['Brand'].str.upper().replace({
        'INVIDIA': 'NVIDIA', 'NVIDA': 'NVIDIA', 'NIVIDIA': 'NVIDIA',
        'INTEL CORP': 'INTEL', 'INTEL CORPORATION': 'INTEL',
        'AMD INC': 'AMD', 'ADVANCED MICRO DEVICES': 'AMD'
    })
    df['is_AMD'] = (df['Brand'] == 'AMD').astype(int)
    df['is_NVIDIA'] = (df['Brand'] == 'NVIDIA').astype(int)
    df['is_INTEL'] = (df['Brand'] == 'INTEL').astype(int)
    if 'GPU Type' not in df.columns:
        df['GPU Type'] = 'Desktop'
    df['PPR'] = df['Performance'] / df['Price']
    return df

# Clustering dan Labeling
def perform_clustering(df, n_clusters=3):
    df = df.dropna(subset=['Price', 'Performance'])
    X = df[['Price', 'Performance']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    return df, kmeans

def assign_cluster_labels(df):
    cluster_summary = df.groupby('Cluster')['Performance'].mean().sort_values()
    label_dict = {
        0: "Entry-level (murah, performa standar)",
        1: "Mid-range (harga & performa seimbang)",
        2: "High-end (harga tinggi, performa maksimal)"
    }
    label_map = {cluster: label_dict[i] for i, cluster in enumerate(cluster_summary.index)}
    df['Cluster Label'] = df['Cluster'].map(label_map).fillna('Unknown')
    return df

def train_regression_model(df):
    X = df[['Price']]
    y = df['Performance']
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_performance(regressor, input_price):
    return regressor.predict(pd.DataFrame({'Price': [input_price]}))[0]

def recommend_gpu(df, input_price, brand_filter=None, gpu_type_filter=None, tolerance=0.2):
    lower, upper = input_price * (1 - tolerance), input_price * (1 + tolerance)
    result = df[(df['Price'] >= lower) & (df['Price'] <= upper)]
    if brand_filter:
        result = result[result['Brand'].str.contains(brand_filter.upper(), na=False)]
    if gpu_type_filter:
        result = result[result['GPU Type'].str.contains(gpu_type_filter, case=False, na=False)]
    return result.sort_values(['Performance', 'Price'], ascending=[False, True]).iloc[0] if not result.empty else None

def recommend_best_value_gpu(df, input_price, tolerance=0.2):
    lower, upper = input_price * (1 - tolerance), input_price * (1 + tolerance)
    df_filtered = df[(df['Price'] >= lower) & (df['Price'] <= upper)].copy()
    if df_filtered.empty:
        return None
    return df_filtered.sort_values('PPR', ascending=False).iloc[0]

def export_recommendation_pdf(gpu_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Rekomendasi GPU", ln=True, align='C')
    for col in gpu_data.index:
        pdf.cell(200, 10, txt=f"{col}: {gpu_data[col]}", ln=True)
    pdf_output = pdf.output(dest='S').encode('latin1')
    b64 = base64.b64encode(pdf_output).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="rekomendasi_gpu.pdf">\U0001F4C4 Unduh PDF Rekomendasi</a>'
    st.markdown(href, unsafe_allow_html=True)

def show_cluster_analysis(df):
    st.subheader("\U0001F4CA Scatter Plot Harga vs Performa GPU")
    fig, ax = plt.subplots(figsize=(8,6))
    palette = sns.color_palette("tab10", n_colors=df['Cluster'].nunique())
    sns.scatterplot(data=df, x='Price', y='Performance', hue='Cluster', palette=palette, ax=ax, alpha=0.7)
    ax.set_title("GPU Clustering")
    st.pyplot(fig)

    st.subheader("\U0001F4C8 Statistik Ringkas per Cluster")
    cluster_stats = df.groupby(['Cluster', 'Cluster Label']).agg(
        Jumlah_GPU = ('Product Name', 'count'),
        Rata_Rata_Harga = ('Price', 'mean'),
        Rata_Rata_Performa = ('Performance', 'mean')
    ).reset_index()

    gb = GridOptionsBuilder.from_dataframe(cluster_stats)
    gb.configure_pagination()
    grid_options = gb.build()
    AgGrid(cluster_stats, gridOptions=grid_options)

    st.subheader("\U0001F4CA Distribusi Price-to-Performance Ratio (PPR)")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    label_map = {
        "Entry-level (murah, performa standar)": "Entry-level",
        "Mid-range (harga & performa seimbang)": "Mid-range",
        "High-end (harga tinggi, performa maksimal)": "High-end"
    }
    df['Short Cluster Label'] = df['Cluster Label'].map(label_map)
    sns.boxplot(data=df, x='Short Cluster Label', y='PPR', palette='Set2', ax=ax2)
    ax2.set_title("Distribusi PPR per Cluster")
    ax2.set_ylabel("Performance / Price")
    ax2.tick_params(axis='x', rotation=15)
    ax2.set_xticklabels(ax2.get_xticklabels(), ha='right')
    st.pyplot(fig2)

def show_advanced_analysis(df):
    st.subheader("\U0001F4C9 Analisis Regresi dan Tren Performa GPU")
    features = ['is_AMD', 'is_NVIDIA', 'is_INTEL', 'Memory', 'Bus Width', 'GPU clock',
                'Memory clock', 'Shaders / TMUs / ROPs', 'TDP', 'Price', 'Release Year']
    target = 'Performance'
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = ['Memory', 'Bus Width', 'GPU clock', 'Memory clock', 'Shaders / TMUs / ROPs', 'TDP', 'Price', 'Release Year']
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('bin', 'passthrough', ['is_AMD', 'is_NVIDIA', 'is_INTEL'])
    ])

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Linear Regression': LinearRegression()
    }

    for name, model in models.items():
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        st.markdown(f"**{name}**")
        st.write(f"- MSE: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"- R2 Score: {r2_score(y_test, y_pred):.2f}")

        if hasattr(model, 'feature_importances_'):
            pipeline.fit(X, y)
            importances = model.feature_importances_
            full_features = numeric_features + ['is_AMD', 'is_NVIDIA', 'is_INTEL']
            sorted_idx = np.argsort(importances)[::-1]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(np.array(full_features)[sorted_idx], importances[sorted_idx])
            ax.set_title(f"Feature Importance - {name}")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

    yearly_perf = df.groupby(['Release Year', 'Brand'])['Performance'].mean().reset_index()
    future_years = np.array([[2025], [2026], [2027]])
    colors = {'AMD': 'red', 'NVIDIA': 'green', 'INTEL': 'blue'}

    fig, ax = plt.subplots(figsize=(12, 6))
    for brand in yearly_perf['Brand'].unique():
        brand_data = yearly_perf[yearly_perf['Brand'] == brand]
        Xb = brand_data['Release Year'].values.reshape(-1, 1)
        yb = brand_data['Performance'].values
        model = LinearRegression().fit(Xb, yb)
        pred = model.predict(future_years)
        ax.plot(Xb.flatten(), yb, label=f'{brand} (actual)', color=colors.get(brand, 'gray'))
        ax.plot(future_years.flatten(), pred, '--', label=f'{brand} (predicted)', color=colors.get(brand, 'gray'))

    ax.set_title("Prediction Performance Trend by Brand")
    ax.set_xlabel("Years")
    ax.set_ylabel("Average Performance")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="GPU Recommender & Analytics", layout="wide")
    st.title("\U0001F3AE GPU Insight & Analytic App")

    uploaded_file = st.sidebar.file_uploader("\U0001F4C1 Upload file CSV", type="csv")
    csv_path = st.sidebar.text_input("\U0001F4C2 atau Path file CSV", "Gpu_Data_wPerformance.csv")
    if uploaded_file:
        df = load_and_clean_data(uploaded_file)
    elif csv_path:
        try:
            df = load_and_clean_data(csv_path)
        except FileNotFoundError:
            st.error("File tidak ditemukan. Cek kembali path file CSV.")
            return
    else:
        st.warning("Masukkan atau upload file CSV.")
        return

    df, kmeans_model = perform_clustering(df)
    df = assign_cluster_labels(df)
    regressor = train_regression_model(df)

    input_price = st.sidebar.number_input("\U0001F4B2 Harga GPU (USD)", min_value=0, step=1, format="%d")
    brand = st.sidebar.selectbox("\U0001F3F7️ Brand", [''] + sorted(df['Brand'].unique()))
    gtype = st.sidebar.selectbox("\U0001F5A5️ GPU Type", [''] + sorted(df['GPU Type'].unique()))

    if st.sidebar.button("\U0001F50D Prediksi & Rekomendasi"):
        if input_price > 0:
            perf = predict_performance(regressor, input_price)
            st.success(f"Performa diprediksi: {perf:.2f}")

            gpu = recommend_gpu(df, input_price, brand, gtype)
            if gpu is not None:
                st.markdown("### \U0001F3C6 GPU Rekomendasi")
                st.write(gpu[['Product Name', 'Brand', 'Price', 'Performance', 'GPU Type', 'Cluster Label']])
                export_recommendation_pdf(gpu)
            else:
                st.warning("\u26A0\ufe0f Tidak ditemukan GPU yang cocok.")

            best_value_gpu = recommend_best_value_gpu(df, input_price)
            if best_value_gpu is not None:
                st.markdown("### \U0001F4A1 GPU dengan Cost-to-Performance Terbaik")
                st.write(best_value_gpu[['Product Name', 'Brand', 'Price', 'Performance', 'PPR', 'GPU Type']])
            else:
                st.warning("\u26A0\ufe0f Tidak ditemukan GPU terbaik berdasarkan rasio harga-performa.")
        else:
            st.sidebar.error("Masukkan harga lebih dari 0.")

    show_cluster_analysis(df)

    if st.sidebar.button("\U0001F52C Analisis Lanjutan"):
        with st.expander("\U0001F4C9 Hasil Analisis Lanjut", expanded=True):
            show_advanced_analysis(df)

    st.markdown("---")
    st.markdown("Made with ❤️ by Kelompok 8")
    st.markdown("Anggota: Laras, Mia, Rafif, Arif")

if __name__ == '__main__':
    main()
