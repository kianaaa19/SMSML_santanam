import os
import pandas as pd
from xgboost import XGBClassifier
import mlflow

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Bank_Churn_Optimization_Santanam")

    # Load data hasil preprocessing
    train = pd.read_csv("churn_bank_ABC_preprocessing/train_preprocessed.csv")
    test = pd.read_csv("churn_bank_ABC_preprocessing/test_preprocessed.csv")

    X_train, y_train = train.drop(columns=['churn']), train['churn']
    X_test, y_test = test.drop(columns=['churn']), test['churn']

    mlflow.autolog()
    print("[+] MLflow Autolog aktif.")

    with mlflow.start_run(run_name="XGBoost_Baseline_santanam"):
        # Model baseline standar tanpa tuning
        model = XGBClassifier(random_state=42, eval_metric='logloss')
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        print(f"[+] Eksekusi Baseline Sukses. Akurasi Lokal: {acc}")

if __name__ == "__main__":
    main()
