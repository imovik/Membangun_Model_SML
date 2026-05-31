import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import dagshub

#   Inisialisasi DagsHub 
dagshub.init(repo_owner='imovik', repo_name='MSML-Model-Tracking', mlflow=True)

# Memuat Data Preprocessed
print("Memuat data bersih...")
df = pd.read_csv('data_preprocessed.zip')
# Memisahkan fitur dan target
X = df.drop('% Silica Concentrate', axis=1)
y = df['% Silica Concentrate']

# Membagi data training dan testing (80% latih, 20% tes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Memulai Eksperimen MLflow
mlflow.set_experiment("Prediksi_Silica_Tambang")

with mlflow.start_run():
    print("Melatih model Random Forest...")
    # Menentukan Hyperparameter
    params = {
        "n_estimators": 50, 
        "max_depth": 10,
        "random_state": 42
    }
    
    # Log parameter secara manual
    mlflow.log_params(params)
    
    # Melatih model
    rf_model = RandomForestRegressor(**params)
    rf_model.fit(X_train, y_train)
    
    # Melakukan Prediksi
    y_pred = rf_model.predict(X_test)
    
    # Menghitung Metrik
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log Metrik secara manual
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    print(f"Model berhasil dilatih! MSE: {mse:.4f} | R2: {r2:.4f}")
    
    # Log Model ke MLflow
    mlflow.sklearn.log_model(rf_model, "random_forest_model")
    
    # --- MENAMBAHKAN 2 ARTEFAK TAMBAHAN (Syarat Advanced) ---
    print("Menyimpan artefak tambahan...")
    
    # Artefak 1: File requirements.txt
    mlflow.log_artifact("requirements.txt")
    
    # Artefak 2: Gambar Plot Feature Importance
    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    feat_importances.plot(kind='barh', color='teal')
    plt.title("Top 10 Feature Importances - Prediksi Silica")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    
    # Log gambar tersebut ke MLflow
    mlflow.log_artifact("feature_importance.png")
    
    print("Selesai! Silakan cek dashboard DagsHub Anda.")