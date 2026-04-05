from pydantic import BaseModel
from typing import List, Optional

# Dashboard'daki her bir makine kartı için
class MachineStatus(BaseModel):
    id: str
    name: str
    department: str
    icon: str
    status_color: str 
    risk_level: str   
    mentor_action: Optional[str] = None

# Chart.js için Telemetri verisi
class TelemetryData(BaseModel):
    labels: List[str]
    performance: List[float]
    temperature: List[float]
    vibration: List[float]
    sound_db: List[float]
    humidity: List[float]

# RUL ve Yapay Zeka Tahmini için
class AIPrediction(BaseModel):
    probability: str 
    rul_days: int
    message: str