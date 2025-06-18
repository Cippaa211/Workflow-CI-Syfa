# modelling.py

import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ====================== MLflow Setup =======================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CI_Titanic_LogReg")

# ====================== Load Dataset =======================
df = pd.read_csv('dataset_preprocessing/dataset_preprocessed_train.csv')

# ================== Feature Engineering ====================
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# ====================== Label Encoding =====================
cat_cols = df.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=["Survived"])
y = df["Survived"]

# ====================== Feature Scaling =====================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ====================== Split Data ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= Hyperparameter Tuning ====================
param_grid = {
    "C": [0.01, 0.1, 1, 10, 50, 100],
    "solver": ["liblinear", "lbfgs"],
    "max_iter": [200, 500, 1000]
}

grid_search = GridSearchCV(
    LogisticRegression(), param_grid, cv=5, scoring='f1', verbose=0, n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# ===================== MLflow Logging =======================
with mlflow.start_run():  # ⛳️ Penting untuk menghindari run ID conflict di GitHub Actions

    # Logging parameter terbaik
    for param_name, value in best_params.items():
        mlflow.log_param(param_name, value)

    # Prediksi dan evaluasi
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc_score": roc_auc
    })

    # Simpan model ke MLflow artifact
    mlflow.sklearn.log_model(best_model, artifact_path="model")

    # Simpan model manual ke artifacts (untuk upload GitHub Actions)
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(best_model, "artifacts/model.pkl")

    print("[CI RUN] Model disimpan ke artifacts/model.pkl")
    print(f"[CI RUN] Akurasi: {acc:.4f} | F1 Score: {f1:.4f}")
