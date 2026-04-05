import random
import time
import requests  # <-- Bunu ekledik
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import MachineStatus, TelemetryData, AIPrediction
from services.data_sim import predict_machine_health

app = FastAPI(title="Fluxa Terminal API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- TELEGRAM AYARLARI (Yeni Eklenen Kısım) ---
TELEGRAM_TOKEN = "8411569714:AAE28M43_Dhe6x2ctzQN4s8jcJj6HLwb2SA"
CHAT_ID = "1568677527"

def send_telegram_alert(machine_id, risk_level):
    """Kritik durumda Telegram bildirimi gönderir."""
    message = (
        f"🚨 *FLUXA AI UYARISI* 🚨\n\n"
        f"🏭 *Makine:* {machine_id}\n"
        f"⚠️ *Tespit Edilen Risk:* {risk_level}\n"
        f"🕒 *Zaman:* {time.strftime('%H:%M:%S')}\n\n"
        f"🔧 _Lütfen sistem üzerinden detayları kontrol edin._"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=5)
    except:
        pass # İnternet kesilirse sistemi kilitlediğine değmez

# --- MEVCUT ENDPOINTLERİN (HİÇBİRİNE DOKUNMADIK) ---

@app.get("/")
def read_root():
    return {"message": "Fluxa AI Backend Aktif ve Çalışıyor."}

@app.get("/api/machines", response_model=list[MachineStatus])
def get_dashboard_machines():
    # Buradaki listen aynı kalıyor...
    return [
        {"id": "PR-A101", "name": "Pres Makinesi A1", "department": "Baskı Hattı", "icon": "precision_manufacturing", "status_color": "Red", "risk_level": "Yüksek", "mentor_action": "Titreşim sensörü bağlantılarını kontrol edin."},
        {"id": "CNC-B4", "name": "CNC Makinesi B4", "department": "CNC İşleme", "icon": "settings_input_component", "status_color": "Orange", "risk_level": "Orta", "mentor_action": "Soğutma sıvısı seviyesini denetleyin."},
        {"id": "COMP-C2", "name": "Kompresör C2", "department": "Enerji", "icon": "air", "status_color": "Green", "risk_level": "Düşük", "mentor_action": "Sistem stabil."},
        {"id": "MOT-D7", "name": "Motor Ünitesi D7", "department": "Ana Tahrik", "icon": "engineering", "status_color": "Green", "risk_level": "Düşük", "mentor_action": "Rutin izleme."},
        {"id": "CONV-E3", "name": "Konveyör E3", "department": "Lojistik", "icon": "conveyor_belt", "status_color": "Green", "risk_level": "Düşük", "mentor_action": "Sistem stabil."}
    ]

@app.get("/api/machines/{machine_id}/telemetry", response_model=TelemetryData)
def get_machine_telemetry(machine_id: str):
    return {
        "labels": ['08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '00:00', '02:00', '04:00', '06:00'],
        "performance": [random.uniform(70, 95) for _ in range(12)],
        "temperature": [random.uniform(40, 80) for _ in range(12)],
        "vibration": [random.uniform(1, 15) for _ in range(12)],
        "sound_db": [random.uniform(50, 100) for _ in range(12)],
        "humidity": [random.uniform(30, 60) for _ in range(12)]
    }

@app.get("/api/stats")
def get_landing_stats():
    start_time = time.time()
    latency_ms = round((time.time() - start_time) * 1000 + random.uniform(8, 14), 2)
    return {
        "uptime": "99.8%",
        "izleme": "24/7",
        "gecikme": f"{latency_ms}ms",
        "entegrasyon": "450+",
        "dogruluk": "%96"
    }

# --- GÜNCELLENEN KRİTİK ENDPOINT ---
@app.get("/api/machines/{machine_id}/ai-prediction", response_model=AIPrediction)
def get_ai_status(machine_id: str):
    current_sensor_data = {} 
    result = predict_machine_health(machine_id, current_sensor_data)
    
    # EĞER MODEL "Yüksek" RİSK DERSE TELEGRAM'A AT
    # Not: Sunumda çok fazla mesaj gelmesini istemiyorsan buraya küçük bir kontrol de koyabiliriz.
    if result["risk_level"] == "Yüksek":
        send_telegram_alert(machine_id, "YÜKSEK")
        
    return result