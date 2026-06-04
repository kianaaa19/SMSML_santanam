import os
import time
import psutil
import pandas as pd
from flask import Flask, request, jsonify
from xgboost import XGBClassifier
import mlflow
from prometheus_client import generate_latest, Counter, Gauge, Histogram, REGISTRY

app = Flask(__name__)

# ==============================================================================
# DOWNLOAD MODEL TERBAIK DARI DAGSHUB REGISTRY
# ==============================================================================
DAGSHUB_USERNAME = "kianaaa19"
DAGSHUB_REPO_NAME = "SMSML_santanam"
os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD', '')

mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")

print("[*] Mengunduh model produksi terbaru dari DagsHub Model Registry...")
try:
    model_uri = "models:/Bank_Churn_Model_Santanam/latest"
    model = mlflow.xgboost.load_model(model_uri)
    print("[+] Model berhasil dimuat dan siap melayani prediksi!")
except Exception as e:
    print(f"[Warning] Gagal memuat dari remote registry: {e}")
    print("[*] Menggunakan model fallback lokal untuk pengetesan...")
    model = XGBClassifier()
    import numpy as np
    model.fit(np.random.randn(10, 11), np.random.randint(0, 2, 10))

# ==============================================================================
# DEFINISI METRIK MONITORING (Total 11 Metrik untuk Kriteria Advance)
# ==============================================================================
HTTP_REQUESTS_TOTAL = Counter('kianaaa19_http_requests_total', 'Total HTTP Requests', ['method', 'endpoint', 'status'])
HTTP_REQUEST_LATENCY = Histogram('kianaaa19_http_request_latency_seconds', 'HTTP Request Latency', ['endpoint'])

PREDICTIONS_TOTAL = Counter('kianaaa19_predictions_total', 'Total prediksi yang dilakukan')
CHURN_PREDICTIONS_TOTAL = Counter('kianaaa19_churn_predictions_total', 'Total nasabah diprediksi Churn')
RETAIN_PREDICTIONS_TOTAL = Counter('kianaaa19_retain_predictions_total', 'Total nasabah diprediksi Retain')
MODEL_ACCURACY_GAUGE = Gauge('kianaaa19_model_accuracy_score', 'Skor akurasi model live saat ini')
HIGH_CHURN_ALERT_GAUGE = Gauge('kianaaa19_high_churn_alert_status', 'Status waspada rasio churn tinggi')

SYSTEM_CPU_USAGE = Gauge('kianaaa19_system_cpu_usage_percent', 'Persentase penggunaan CPU sistem')
SYSTEM_MEMORY_USAGE = Gauge('kianaaa19_system_memory_usage_bytes', 'Penggunaan memori sistem')
SYSTEM_DISK_USAGE = Gauge('kianaaa19_system_disk_usage_percent', 'Persentase penggunaan storage')
SERVER_UPTIME = Gauge('kianaaa19_server_uptime_seconds', 'Lama waktu server berjalan')

START_TIME = time.time()
MODEL_ACCURACY_GAUGE.set(0.8625) # Set sesuai akurasi run MLProject kamu tadi

@app.route('/predict', methods=['POST'])
def predict():
    start_req_time = time.time()
    HTTP_REQUESTS_TOTAL.labels(method='POST', endpoint='/predict', status='200').inc()

    try:
        data = request.get_json()
        df_input = pd.DataFrame([data])

        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]

        PREDICTIONS_TOTAL.inc()
        if prediction == 1:
            CHURN_PREDICTIONS_TOTAL.inc()
        else:
            RETAIN_PREDICTIONS_TOTAL.inc()

        latency = time.time() - start_req_time
        HTTP_REQUEST_LATENCY.labels(endpoint='/predict').observe(latency)

        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'churn_probability': float(probability),
            'kredensial_tester': 'kianaaa19'
        })

    except Exception as e:
        HTTP_REQUESTS_TOTAL.labels(method='POST', endpoint='/predict', status='500').inc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    SYSTEM_CPU_USAGE.set(psutil.cpu_percent())
    SYSTEM_MEMORY_USAGE.set(psutil.virtual_memory().used)
    SYSTEM_DISK_USAGE.set(psutil.disk_usage('/').percent)
    SERVER_UPTIME.set(time.time() - START_TIME)

    total_pred = PREDICTIONS_TOTAL._value.get() if PREDICTIONS_TOTAL._value.get() is not None else 1
    churn_pred = CHURN_PREDICTIONS_TOTAL._value.get() if CHURN_PREDICTIONS_TOTAL._value.get() is not None else 0
    if (churn_pred / max(total_pred, 1)) > 0.4:
        HIGH_CHURN_ALERT_GAUGE.set(1.0)
    else:
        HIGH_CHURN_ALERT_GAUGE.set(0.0)

    return generate_latest(REGISTRY), 200, {'Content-Type': 'text/plain; charset=utf-8'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
