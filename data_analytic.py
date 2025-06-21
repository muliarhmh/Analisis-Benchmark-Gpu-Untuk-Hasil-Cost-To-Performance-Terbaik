import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Load the data
df = pd.read_csv('Gpu_Data_wPerformance.csv')

# Data Exploration
print(df.info())
print(df.describe())
print(df['Brand'].value_counts())

# Data Cleaning
# Ekstrak nilai numerik dari kolom string
df['Memory'] = df['Memory'].str.extract(r'(\d+)').astype(float)
df['Bus Width'] = df['Bus'].str.extract(r'x(\d+)').astype(float)
df['GPU clock'] = df['GPU clock'].str.extract(r'(\d+)').astype(float)
df['Memory clock'] = df['Memory clock'].str.extract(r'(\d+)').astype(float)
df['Shaders / TMUs / ROPs'] = df['Shaders / TMUs / ROPs'].str.extract(r'(\d+)').astype(float)

# Ubah kolom tanggal ke datetime dan ambil tahunnya
df['Released'] = pd.to_datetime(df['Released'], errors='coerce')
df['Release Year'] = df['Released'].dt.year

# Normalisasi brand (mengatasi 'Nvidia' dan 'NVIDIA')
df['Brand'] = df['Brand'].str.upper()

# Fitur binary brand
df['is_AMD'] = (df['Brand'] == 'AMD').astype(int)
df['is_NVIDIA'] = (df['Brand'] == 'NVIDIA').astype(int)
df['is_INTEL'] = (df['Brand'] == 'INTEL').astype(int)

# Memory Type (catatan: aslinya 'Memory' sudah jadi float, jadi info ini mungkin tidak tersedia)
df['Memory Type'] = 'Unknown'  # placeholder

# Pilih fitur dan target
features = ['is_AMD', 'is_NVIDIA', 'is_INTEL', 'Memory', 'Bus Width',
            'GPU clock', 'Memory clock', 'Shaders / TMUs / ROPs',
            'TDP', 'Price', 'Release Year']
target = 'Performance'

# Hapus baris dengan nilai null
df = df.dropna(subset=features + [target])

# Pisah data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline preprocessing
numeric_features = ['Memory', 'Bus Width', 'GPU clock', 'Memory clock',
                    'Shaders / TMUs / ROPs', 'TDP', 'Price', 'Release Year']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Gunakan 'passthrough' untuk kolom is_AMD, is_NVIDIA, is_INTEL karena sudah numerik
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('bin', 'passthrough', ['is_AMD', 'is_NVIDIA', 'is_INTEL'])
    ])

# Model-model yang akan dicoba
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Linear Regression': LinearRegression()
}

# Evaluasi
results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2}

    # Feature importance (khusus untuk tree model)
    if hasattr(model, 'feature_importances_'):
        pipeline.fit(X, y)
        importances = model.feature_importances_
        full_features = numeric_features + ['is_AMD', 'is_NVIDIA', 'is_INTEL']
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title(f"{name} Feature Importance")
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [full_features[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()

# Cetak hasil
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  MSE: {metrics['MSE']:.2f}")
    print(f"  R2: {metrics['R2']:.2f}")
    print()

# Bandingkan brand
brand_perf = df.groupby('Brand')['Performance'].agg(['mean', 'median', 'std'])
print("\nBrand Performance Comparison:")
print(brand_perf)

# Visualisasi: Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x='Brand', y='Performance', data=df)
plt.title('Performance Distribution by Brand')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Tren performa tahunan
plt.figure(figsize=(12, 6))
sns.lineplot(x='Release Year', y='Performance', hue='Brand', data=df, errorbar=None)

plt.title('Performance Trend by Brand Over Years')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
from sklearn.linear_model import LinearRegression

# Ambil data rata-rata performa per tahun untuk masing-masing brand
yearly_perf = df.groupby(['Release Year', 'Brand'])['Performance'].mean().reset_index()

# Tahun prediksi
future_years = np.array([[2025], [2026], [2027]])

plt.figure(figsize=(12, 6))
colors = {'AMD': 'red', 'NVIDIA': 'green', 'INTEL': 'blue'}

# Loop untuk masing-masing brand
for brand in yearly_perf['Brand'].unique():
    brand_data = yearly_perf[yearly_perf['Brand'] == brand]
    
    # Training model regresi linier
    X = brand_data['Release Year'].values.reshape(-1, 1)
    y = brand_data['Performance'].values
    model = LinearRegression().fit(X, y)
    
    # Prediksi untuk tahun 2025–2027
    pred = model.predict(future_years)
    
    # Gabungkan data aktual + prediksi
    plt.plot(X.flatten(), y, label=f'{brand} (actual)', color=colors[brand])
    plt.plot(future_years.flatten(), pred, '--', label=f'{brand} (predicted)', color=colors[brand])

# Finalisasi grafik
plt.title("Prediction Performance Trend by Brand")
plt.xlabel("Years")
plt.ylabel("Average Performance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
