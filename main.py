"""
Green AI Solar - Backend FastAPI pour Render
=============================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os

# ============================================
# Configuration FastAPI
# ============================================

app = FastAPI(
    title="Green AI Solar API",
    description="API de prédiction d'efficacité de panneaux solaires",
    version="1.0.0",
)

# CORS - Autoriser GitHub Pages et localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://*.github.io",  # Toutes les GitHub Pages
        "*"  # En production, remplacer par ton URL exacte
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Modèles Pydantic
# ============================================

class SolarPanelInput(BaseModel):
    temperature: float = Field(default=25.0, ge=-20, le=60)
    irradiance: float = Field(default=500.0, ge=0, le=1200)
    humidity: float = Field(default=50.0, ge=0, le=100)
    panel_age: float = Field(default=10.0, ge=0, le=40)
    maintenance_count: float = Field(default=3.0, ge=0, le=20)
    soiling_ratio: float = Field(default=0.7, ge=0.3, le=1)
    voltage: float = Field(default=30.0, ge=0, le=100)
    current: float = Field(default=2.0, ge=0, le=10)
    module_temperature: float = Field(default=35.0, ge=0, le=80)
    cloud_coverage: float = Field(default=30.0, ge=0, le=100)
    wind_speed: float = Field(default=7.0, ge=0, le=20)
    pressure: float = Field(default=1013.0, ge=950, le=1050)

class PredictionResponse(BaseModel):
    efficiency: float
    efficiency_percent: float
    quality_label: str
    confidence: str

# ============================================
# Modèle ML
# ============================================

MODEL_PATH = "model/solar_efficiency_model.joblib"
model = None

FEATURE_NAMES = [
    'temperature', 'irradiance', 'humidity', 'panel_age',
    'maintenance_count', 'soiling_ratio', 'voltage', 'current',
    'module_temperature', 'cloud_coverage', 'wind_speed', 'pressure'
]

FEATURE_IMPORTANCE = {
    'irradiance': 0.669346,
    'soiling_ratio': 0.228650,
    'panel_age': 0.078271,
    'humidity': 0.010584,
    'module_temperature': 0.005262,
    'temperature': 0.003132,
    'current': 0.003024,
    'voltage': 0.000467,
    'wind_speed': 0.000427,
    'pressure': 0.000384,
    'cloud_coverage': 0.000230,
    'maintenance_count': 0.000221
}

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Modèle chargé depuis {MODEL_PATH}")
        return True
    print(f"⚠️ Modèle non trouvé, mode simulation")
    return False

def simulate_prediction(inputs: dict) -> float:
    efficiency = 0.3
    efficiency += (inputs['irradiance'] / 1000) * 0.35
    efficiency += inputs['soiling_ratio'] * 0.15
    efficiency -= (inputs['panel_age'] / 35) * 0.08
    if inputs['temperature'] > 25:
        efficiency -= ((inputs['temperature'] - 25) / 100) * 0.03
    efficiency -= (inputs['humidity'] / 100) * 0.02
    return max(0.1, min(0.85, efficiency))

def get_quality_label(efficiency: float) -> str:
    if efficiency < 0.3:
        return "Faible"
    elif efficiency < 0.5:
        return "Modérée"
    elif efficiency < 0.7:
        return "Bonne"
    return "Excellente"

# ============================================
# Endpoints
# ============================================

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    return {
        "message": "🌞 Green AI Solar API",
        "status": "online",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_efficiency(data: SolarPanelInput):
    try:
        inputs = data.model_dump()
        
        if model is not None:
            input_df = pd.DataFrame([inputs])[FEATURE_NAMES]
            efficiency = float(model.predict(input_df)[0])
        else:
            efficiency = simulate_prediction(inputs)
        
        efficiency = max(0.0, min(1.0, efficiency))
        
        return PredictionResponse(
            efficiency=round(efficiency, 4),
            efficiency_percent=round(efficiency * 100, 2),
            quality_label=get_quality_label(efficiency),
            confidence="high" if model is not None else "simulated"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feature-importance")
async def get_feature_importance():
    features = [
        {"name": name, "importance": round(imp * 100, 2)}
        for name, imp in sorted(FEATURE_IMPORTANCE.items(), key=lambda x: -x[1])
    ]
    return {"features": features}

@app.get("/model-info")
async def get_model_info():
    return {
        "model_type": "GradientBoostingRegressor",
        "r2_score": 0.814,
        "mse": 0.002,
        "features": FEATURE_NAMES
    }
