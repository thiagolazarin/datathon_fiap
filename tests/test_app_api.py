import json
import types
from fastapi.testclient import TestClient
from app.main import app, _load_artifact

def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_version_loads_artifact(monkeypatch):
    # mock do artefato em memória
    fake_artifact = {
        "model": types.SimpleNamespace(predict_proba=lambda X: [[0.2, 0.8]]),
        "feature_columns": ["tem_email","salario_valor"],
        "threshold": 0.5,
        "operating_mode": "prec80",
        "metadata": {"created_at":"2025-01-01T00:00:00Z"}
    }
    def fake_load():
        from app.main import globals
    # monkeypatch do joblib.load
    import app.main as m
    m.artifact = fake_artifact
    m.model = fake_artifact["model"]
    m.feature_columns = fake_artifact["feature_columns"]
    m.threshold = fake_artifact["threshold"]

    client = TestClient(app)
    r = client.get("/version")
    assert r.status_code == 200
    j = r.json()
    assert j["operating_mode"] == "prec80"
    assert j["threshold"] == 0.5
    assert "feature_columns" in j

def test_predict_ok(monkeypatch):
    # mock do artefato/modelo
    import app.main as m
    m.artifact = {
        "model": None,  # será injetado abaixo
        "feature_columns": ["tem_email","salario_valor"],
        "threshold": 0.6,
        "operating_mode": "prec80",
        "metadata": {}
    }
    class FakeModel:
        def predict_proba(self, X):
            # força um score de 0.7 => acima do threshold 0.6 => True
            return [[0.3, 0.7]]
    m.model = FakeModel()
    m.feature_columns = ["tem_email","salario_valor"]
    m.threshold = 0.6

    client = TestClient(app)
    payload = {
        "features": {"tem_email": 1, "salario_valor": 3000},
        "codigo_profissional": 123
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert j["aprovado_pelo_modelo"] is True
    assert "probabilidade_contratacao" in j
