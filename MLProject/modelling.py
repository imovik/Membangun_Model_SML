import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import dagshub

dagshub.init(repo_owner="imovik", repo_name="MSML-Model-Tracking", mlflow=True)


print("Memuat data bersih...")
df = pd.read_csv("data_preprocessed.zip")


X = df.drop("% Silica Concentrate", axis=1)
y = df["% Silica Concentrate"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


mlflow.set_experiment("Prediksi_Silica_Tambang")

with mlflow.start_run():
    print("Melatih model Random Forest (Baseline)...")

    params = {"n_estimators": 50, "max_depth": 10, "random_state": 42}

    mlflow.log_params(params)

    rf_model = RandomForestRegressor(**params)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    print(f"Model berhasil dilatih! MSE: {mse:.4f} | R2: {r2:.4f}")

    mlflow.sklearn.log_model(rf_model, "random_forest_model")

    print("Menyimpan artefak tambahan...")

    mlflow.log_artifact("requirements.txt")

    plt.figure(figsize=(10, 6))
    feat_importances = (
        pd.Series(rf_model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(10)
    )
    feat_importances.plot(kind="barh", color="teal")
    plt.title("Top 10 Feature Importances - Prediksi Silica")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_importance.png")

    #
    mlflow.log_artifact("feature_importance.png")

    print("Selesai! Silakan cek dashboard DagsHub Anda.")
