import time
import random
import requests

url = "http://localhost:8000/predict"

fitur_names = [
    'credit_score', 'age', 'tenure', 'balance', 'products_number',
    'credit_card', 'active_member', 'estimated_salary',
    'country_Germany', 'country_Spain', 'gender_Male'
]

print("[*] Memulai pengiriman trafik simulasi... Tekan Ctrl+C untuk berhenti.")

while True:
    try:
        payload = {fitur: random.uniform(-2.0, 2.0) for fitur in fitur_names}

        # Buat anomali sesekali agar grafik naik turun menarik
        if random.random() > 0.6:
            payload['age'] = random.uniform(1.5, 3.5)
            payload['active_member'] = -1.0

        response = requests.post(url, json=payload)
        print(f"[Hit] Status: {response.status_code} -> Hasil Prediksi Churn: {response.json()['prediction']}")

        time.sleep(random.uniform(0.1, 0.6))

    except KeyboardInterrupt:
        print("\n[+] Trafik dihentikan.")
        break
    except Exception as e:
        print(f"[Error] Pastikan server exporter di port 8000 sudah up: {e}")
        time.sleep(2)
