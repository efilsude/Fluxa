import joblib
import pandas as pd
from catboost import CatBoostRegressor

# Modeller ai_ml klasöründe
ANOMALY_MODEL_PATH = "../ai_ml/isolation_forest_modeli.joblib"
RUL_MODEL_PATH = "../ai_ml/audentes_kestirimci_bakim.cbm"

try:
    anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
    rul_model = CatBoostRegressor()
    rul_model.load_model(RUL_MODEL_PATH)
    MODELS_LOADED = True
    print("✅ AI Modelleri başarıyla yüklendi!")
except Exception as e:
    print(f"⚠️ Modeller yüklenirken hata oluştu (Simülasyon çalışacak): {e}")
    MODELS_LOADED = False

def predict_machine_health(machine_id: str, current_sensor_data: dict) -> dict:
    """Makinenin anlık durumunu değerlendirir ve RUL tahmini yapar."""
    
    # Eğer modeller henüz klasöre atılmadıysa arayüz çökmesin diye statik veri dönüyoruz
    if not MODELS_LOADED or not current_sensor_data:
        if machine_id == "PR-A101":
            return {
                "probability": "Yüksek",
                "rul_days": 7,
                "message": "Önümüzdeki bakım ihtimali: Yüksek. 7 gün içinde kontrol önerilir."
            }
        else:
            return {
                "probability": "Düşük",
                "rul_days": 120,
                "message": "Sistem stabil çalışıyor. Herhangi bir anomali tespit edilmedi."
            }

    # Gerçek model entegrasyonu (Canlı veri geldiğinde çalışacak kısım)
    input_df = pd.DataFrame([current_sensor_data])
    anomaly_score = anomaly_model.predict(input_df)[0]
    predicted_rul = rul_model.predict(input_df)[0]

    if anomaly_score == -1:
        return {
            "probability": "Kritik",
            "rul_days": int(predicted_rul),
            "message": "Kritik anomali tespit edildi. Hızlı müdahale gerekli!"
        }
    else:
        return {
            "probability": "Düşük",
            "rul_days": int(predicted_rul),
            "message": "Sistem stabil çalışıyor."
        }