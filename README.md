🧠 Projeto: Datathon MLET3 — Hiring Model (end-to-end)
🎯 Objetivo

Entregar um pipeline completo de Machine Learning para predição de contratação de candidatos, cobrindo:

✅ Ingestão & pré-processamento de applicants e prospects em PostgreSQL
✅ Feature engineering (candidatos) + labels (prospects)
✅ Tabela Gold (join de features + target)
✅ Treino, avaliação e serialização do modelo (joblib) com threshold calibrado por precisão mínima
✅ API FastAPI com /predict, /health, /version
✅ Logs de inferência no banco para auditoria
✅ Containerização (Docker/Compose) e execução local
✅ Testes (pytest) com cobertura
✅ Monitoramento simples de drift (baseline + rotina diária)

🧱 Stack

Python 3.11, FastAPI, scikit-learn, LightGBM
PostgreSQL 15 (via Docker)
SQLAlchemy (IO com DB)
pytest (+ opcionalmente pytest-cov)
Docker / docker compose

📦 Estrutura do repositório
.
├─ app/
│  └─ main.py                     # API FastAPI (/predict, /health, /version)
├─ src/
│  ├─ preprocessing/
│  │  ├─ applicants_ingest.py     # CLI: carrega ./data/applicants.json -> DB
│  │  └─ prospects_ingest.py      # CLI: carrega ./data/prospects.json  -> DB
│  ├─ feature_engineering/
│  │  ├─ applicants_features.py   # features a partir de applicants_raw (chunk + barra de progresso)
│  │  ├─ prospects_labels.py      # labels a partir de prospects_raw (chunk)
│  │  └─ gold.py                  # monta gold_applicants + COPY com progresso
│  ├─ training/
│  │  ├─ train.py                 # treina modelo + calibração + salva joblib
│  │  └─ evaluate.py              # métricas holdout e corte por precisão
│  ├─ monitoring/
│  │  ├─ record_baseline.py       # salva baseline de features (gold) em model_baseline
│  │  └─ monitor_daily.py         # volume últimas 24h e alertas simples de drift
│  └─ utils.py                    # helpers (DB engine, threshold_for_min_precision)
├─ artifacts/
│  └─ modelo_prec80.joblib        # artefato do modelo (gerado por treino)
├─ tests/
│  ├─ test_app_api.py             # testes da API
│  ├─ test_features.py            # testes de feature engineering (núcleo)
│  └─ test_utils.py               # testes utilitários
├─ data/                          # (opcional) JSONs de entrada para ingestão
├─ Dockerfile
├─ docker-compose.yml
├─ .env                           # variáveis de ambiente (DB / artefato)
├─ requirements.txt               # deps locais
├─ requirements-api.txt           # deps do container API
├─ pytest.ini, .coveragerc
└─ README.md

🔧 Configuração (local)
1. Crie o venv e instale deps
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
2. Crie o .env na raiz (exemplo):
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=postgres
POSTGRES_SCHEMA=public

# Artefato do modelo (usado pela API):
MODEL_ARTIFACT=./artifacts/modelo_prec80.joblib
3. Suba o Postgres via Docker
docker compose up -d db

📥 Ingestão de dados
Assumindo os arquivos:

./data/applicants.json
./data/prospects.json

# Applicants -> applicants_raw
python -m src.preprocessing.applicants_ingest --json ./data/applicants.json --table applicants_raw

# Prospects -> prospects_raw
python -m src.preprocessing.prospects_ingest --json ./data/prospects.json --table prospects_raw

🧪 Feature Engineering & Labels
# Applicants -> applicants_feat (com chunk e progresso)
python -c "from src.feature_engineering.applicants_features import build_and_write_applicants_feat_chunked; print(build_and_write_applicants_feat_chunked('applicants_raw','applicants_feat'))"

# Prospects -> prospects_labels (com chunk e progresso)
python -c "from src.feature_engineering.prospects_labels import build_and_write_prospects_labels_chunked; print(build_and_write_prospects_labels_chunked('prospects_raw','prospects_labels'))"

🥇 Tabela Gold
python -c "from src.feature_engineering.gold import build_and_write_gold; print(build_and_write_gold('applicants_feat','prospects_labels','gold_applicants'))"

🤖 Treinamento, Avaliação e Artefato
Treina LightGBM com calibração e calcula threshold para precisão mínima (padrão 0.80).
# Treino
python -m src.training.train

# Avaliação holdout + cortes (0.5 e corte por precisão mínima)
python -m src.training.evaluate

🌐 API (FastAPI)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
Exemplo de payload para /predict:

{
  "codigo_profissional": 31001,
  "features": {
    "tem_email": 1, "tem_telefone": 1, "tem_linkedin": 0, "tem_local": 1, "tem_objetivo": 1,
    "email_corporativo": 0, "salario_valor": 3000,
    "ingl_nenhum": 1, "ingl_basico": 0, "ingl_intermediario": 0, "ingl_avancado": 0, "ingl_outro": 0,
    "esp_nenhum": 1, "esp_basico": 0, "esp_intermediario": 0, "esp_avancado": 0, "esp_outro": 0,
    "outro_idioma_presente": 0,
    "esc_pos": 0, "esc_tecnologo": 0, "esc_medio": 0, "esc_superior_completo": 1, "esc_superior_incompleto": 0,
    "area_admin": 1, "area_ti": 0, "area_financeiro": 0,
    "titulo_admin": 1, "titulo_ti": 0, "titulo_dados_bi": 0, "titulo_financeiro": 0,
    "cert_mos_word": 0, "cert_mos_excel": 1, "cert_mos_outlook": 0, "cert_mos_powerpoint": 0, "cert_sap_fi": 0,
    "has_cert": 1,
    "cv_excel_avancado": 1, "cv_kpi": 0, "cv_controladoria": 0, "cv_contabil": 1, "cv_financeiro": 1, "cv_administrativo": 1,
    "cv_sap": 0, "cv_protheus": 0, "cv_navision": 0, "cv_tamanho_maior_1500": 0
  }
}

Resposta:

{
    "probabilidade_contratacao": 0.2090899291607077,
    "aprovado_pelo_modelo": false,
    "threshold": 0.4960699431287667,
    "operating_mode": "prec80",
    "codigo_profissional": 31001
}

🐳 Docker / Compose
Subir tudo
docker compose up -d --build


Serviços:

db: Postgres 15 (exposto em 5432)

api: FastAPI (exposto em 8000)

Variáveis: a API lê MODEL_ARTIFACT (no compose mapeado como /app/artifacts/modelo_prec80.joblib).
Volume: ./artifacts é montado em /app/artifacts:ro.

Testar a API no container
curl -s http://localhost:8000/health
curl -s http://localhost:8000/version | jq

Rodar scripts dentro do container API
# baseline (uma vez após ter a gold pronta)
docker compose run --rm api python -m src.monitoring.record_baseline

# monitor diário (imprime volume últimas 24h e alertas simples)
docker compose run --rm api python -m src.monitoring.monitor_daily


Esses comandos usam o mesmo PYTHONPATH=/app configurado no Dockerfile.

🧩 Features usadas no modelo

Conjunto final de colunas esperado (e retornado por /version):

tem_email, tem_telefone, tem_linkedin, tem_local, tem_objetivo, email_corporativo,
salario_valor, ingl_nenhum, ingl_basico, ingl_intermediario, ingl_avancado, ingl_outro,
esp_nenhum, esp_basico, esp_intermediario, esp_avancado, esp_outro, outro_idioma_presente,
esc_pos, esc_tecnologo, esc_medio, esc_superior_completo, esc_superior_incompleto,
area_admin, area_ti, area_financeiro, titulo_admin, titulo_ti,
titulo_dados_bi, titulo_financeiro, cert_mos_word, cert_mos_excel, cert_mos_outlook,
cert_mos_powerpoint, cert_sap_fi, has_cert, cv_excel_avancado, cv_kpi, cv_controladoria,
cv_contabil, cv_financeiro, cv_administrativo, cv_sap, cv_protheus, cv_navision, cv_tamanho_maior_1500

📊 Monitoramento simples (drift e volume)
1) Gravar baseline (a partir da gold)
python -m src.monitoring.record_baseline
# ou no container:
docker compose run --rm api python -m src.monitoring.record_baseline


Cria (se não existir) a tabela:

CREATE TABLE IF NOT EXISTS model_baseline (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  model_path TEXT,
  stats JSONB
);


E salva estatísticas:

binárias: taxa de 1 (rate1)

numéricas: média e desvio padrão (mean, std) — atualmente salario_valor

2) Checagem diária (últimas 24h)
python -m src.monitoring.monitor_daily
# ou
docker compose run --rm api python -m src.monitoring.monitor_daily


Mostra:

Volume por hora e avg_score

Alertas de drift por regra simples:

Binárias: |rate_now - rate_base| > 0.15

Numéricas: |mean_now - mean_base|/std_base > 3

Qualidade (opcional) se houver tabela feedback (join com inference_log)

Os logs de inferência são gravados pela API em inference_log com:
ts/created_at, model_mode, model_threshold, model_path, score, decision, codigo_profissional, payload.

✅ Testes & Cobertura
Rodar testes
pytest

Com cobertura (se tiver pytest-cov instalado)
pytest -q --cov=src --cov=app --cov-report=term-missing

Dicas de import para testes

Garanta app/__init__.py e src/__init__.py

No pytest.ini, inclua:

[pytest]
pythonpath = .

(ou crie tests/conftest.py inserindo a raiz no sys.path)

🧭 Troubleshooting

ModuleNotFoundError: No module named 'app'/'src' ao rodar pytest
→ Veja a dica de pythonpath acima.

Modelo não carregado na API
→ Confirme o caminho do artefato:

Local: MODEL_ARTIFACT=./artifacts/modelo_prec80.joblib

Docker: MODEL_ARTIFACT=/app/artifacts/modelo_prec80.joblib (e volume montado)

Log grava no Postgres do container, mas não no local
→ No Docker, a API aponta para POSTGRES_HOST=db (o container db).
Se quer gravar no seu Postgres local, rode a API localmente com POSTGRES_HOST=localhost.

UserWarning: X does not have valid feature names
→ Certifique-se de enviar todas as colunas de feature_columns (ordem não importa; a API reindexa).

🚀 Fluxo resumido (end-to-end)

Subir DB: docker compose up -d db
Ingest: applicants_ingest e prospects_ingest
Features/Labels: build_and_write_applicants_feat_chunked e build_and_write_prospects_labels_chunked
Gold: build_and_write_gold
Treino: python -m src.training.train

(Opcional) Avaliação: python -m src.training.evaluate

API local: uvicorn app.main:app --reload
ou Docker: docker compose up -d --build
Testar /version e /predict (Postman/curl)

Monitoramento:
record_baseline.py (uma vez pós-gold)
monitor_daily.py (rotina — pode cronar)