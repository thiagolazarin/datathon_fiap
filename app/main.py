import os, json, joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy import text
import numpy as np 
from src.utils import make_engine_from_env

# Carrega artefato uma única vez no startup
ARTIFACT_PATH = os.getenv("MODEL_ARTIFACT", "./artifacts/modelo_prec80.joblib")

app = FastAPI(title="Hiring Model API", version="1.0.0")
artifact: Dict[str, Any] = {}
model = None
feature_columns: List[str] = []
threshold: float = 0.5

class PredictPayload(BaseModel):
    # features em dicionário: {coluna: valor}
    features: Dict[str, Any] = Field(..., description="Mapa de features conforme feature_columns do artefato")
    codigo_profissional: Optional[int] = Field(None, description="Opcional, se disponível")

def _load_artifact():
    global artifact, model, feature_columns, threshold
    if not os.path.exists(ARTIFACT_PATH):
        raise FileNotFoundError(f"Artifact not found: {ARTIFACT_PATH}")
    artifact = joblib.load(ARTIFACT_PATH)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    threshold = float(artifact["threshold"])

@app.on_event("startup")
def startup_event():
    # Carregar modelo e artefatos quando a API iniciar
    try:
        _load_artifact()
        print(f"✅ Artefato carregado com sucesso: {ARTIFACT_PATH}")
    except Exception as e:
        print(f"❌ Falha ao carregar artefato: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

@app.get("/version")
def version():
    meta = artifact.get("metadata", {})
    return {
        "operating_mode": artifact.get("operating_mode"),
        "threshold": artifact.get("threshold"),
        "feature_columns": feature_columns,
        "metadata": meta,
        "artifact_path": ARTIFACT_PATH
    }

def _log_inference(payload: Dict[str, Any], score: float, decision: int, codigo_profissional: Optional[int]):
    try:
        eng = make_engine_from_env()
        with eng.begin() as c:
            c.execute(
                text("""INSERT INTO inference_log
                        (model_mode, model_threshold, model_created_at, model_path, score, decision, codigo_profissional, payload)
                        VALUES (:mode, :thr, :created, :path, :score, :dec, :cod, :payload)"""),
                dict(
                    mode=artifact.get("operating_mode"),
                    thr=float(artifact.get("threshold")),
                    created=artifact.get("metadata", {}).get("created_at"),
                    path=ARTIFACT_PATH,
                    score=float(score),
                    dec=int(decision),
                    cod=codigo_profissional,
                    payload=json.dumps(payload)
                )
            )
    except Exception:
        pass

@app.post("/predict")
def predict(req: PredictPayload):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    row = {col: req.features.get(col, None) for col in feature_columns}
    X = pd.DataFrame([row]).reindex(columns=feature_columns)

    # --- normaliza a saída do predict_proba para lidar com list/np.ndarray 1D/2D
    try:
        proba_raw = model.predict_proba(X)

        # para listas/tuplas -> vira np.array
        proba_arr = np.asarray(proba_raw)

        if proba_arr.ndim == 0:
            # escalar
            proba = float(proba_arr)
        elif proba_arr.ndim == 1:
            # vetor (pega o primeiro valor)
            proba = float(proba_arr.ravel()[0])
        else:
            # matriz; se tiver 2 colunas, usa a da classe positiva
            proba = float(proba_arr[0, 1] if proba_arr.shape[1] >= 2 else proba_arr.ravel()[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao gerar probabilidade: {e}")

    label = int(proba >= threshold)
    _log_inference(req.features, proba, label, req.codigo_profissional)

    return {
        "probabilidade_contratacao": proba,
        "aprovado_pelo_modelo": bool(label),
        "threshold": threshold,
        "operating_mode": artifact.get("operating_mode"),
        "codigo_profissional": req.codigo_profissional
    }