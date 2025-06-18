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

# Set tracking lokal
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CI_Titanic_LogReg")

# Load dan preprocessing data
df = pd.read_csv('dataset_preprocessing/dataset_preprocessed_train.csv')
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.drop(columns=["Survived"])
y = df["Survived"]

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Model & tuning
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"],
    "max_iter": [200, 500]
}
grid = GridSearchCV(LogisticRegression(), param_grid, scoring='f1', cv=5)
grid.fit(X_train, y_train)

model = grid.best_estimator_

# ❌ JANGAN panggil start_run() di sini, karena sudah dijalankan otomatis oleh mlflow CLI
# Logging
for k, v in grid.best_params_.items():
    mlflow.log_param(k, v)

y_pred = model.predict(X_test)
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "roc_auc_score": roc_auc_score(y_test, y_pred)
}
mlflow.log_metrics(metrics)

# Log model dan dump manual
mlflow.sklearn.log_model(model, artifact_path="model")
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/model.pkl")

print("[CI RUN] Model disimpan ke artifacts/model.pkl")
print(f"[CI RUN] Akurasi: {metrics['accuracy']:.4f} | F1 Score: {metrics['f1_score']:.4f}")
