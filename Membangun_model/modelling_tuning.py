import os
import pandas as pd
from xgboost import XGBClassifier
import mlflow
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def main():
    DAGSHUB_USERNAME = "kianaaa19"
    DAGSHUB_REPO_NAME = "SMSML_santanam"
    token_dagshub = os.getenv('MLFLOW_TRACKING_PASSWORD', '')

    os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token_dagshub

    if token_dagshub:
        mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")
        print(f"[+] Sukses: Terkoneksi ke DagsHub Cloud ({DAGSHUB_USERNAME})")
    else:
        mlflow.set_tracking_uri("file:./mlruns")
        print("[Warning] Token kosong. Menggunakan tracking lokal.")

    mlflow.set_experiment("Bank_Churn_Optimization_santanam")

    train = pd.read_csv("churn_bank_ABC_preprocessing/train_preprocessed.csv")
    test = pd.read_csv("churn_bank_ABC_preprocessing/test_preprocessed.csv")
    X_train, y_train = train.drop(columns=['churn']), train['churn']
    X_test, y_test = test.drop(columns=['churn']), test['churn']

    tuning_params = [
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 150},
        {"max_depth": 5, "learning_rate": 0.1, "n_estimators": 200},
        {"max_depth": 7, "learning_rate": 0.2, "n_estimators": 100}
    ]

    for i, params in enumerate(tuning_params):
        run_name = f"XGBoost_Tuning_Hyperparam_{i+1}"

        with mlflow.start_run(run_name=run_name):
            model = XGBClassifier(
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                n_estimators=params["n_estimators"],
                random_state=42,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)

            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(6, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Churn', 'Churn'])
            disp.plot(cmap=plt.cm.Blues, ax=ax)
            plt.title(f"Confusion Matrix - {run_name}")

            nama_gambar_cm = f"confusion_matrix_tuning_{i+1}.png"
            plt.savefig(nama_gambar_cm)
            plt.close()

            mlflow.log_artifact(nama_gambar_cm, artifact_path="plots")

            if os.path.exists(nama_gambar_cm):
                os.remove(nama_gambar_cm)

            mlflow.log_param("max_depth", params["max_depth"])
            mlflow.log_param("learning_rate", params["learning_rate"])
            mlflow.log_param("n_estimators", params["n_estimators"])
            mlflow.log_metric("tuning_accuracy", acc)

            print(f"[+] {run_name} logged to DagsHub! Akurasi: {acc}")

if __name__ == "__main__":
    main()
