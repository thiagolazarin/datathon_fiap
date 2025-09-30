import json, os, pandas as pd, numpy as np
from sqlalchemy import text
from src.utils import make_engine_from_env

FEATURES = [
 'tem_email','tem_telefone','tem_linkedin','tem_local','tem_objetivo','email_corporativo',
 'salario_valor','ingl_nenhum','ingl_basico','ingl_intermediario','ingl_avancado','ingl_outro',
 'esp_nenhum','esp_basico','esp_intermediario','esp_avancado','esp_outro','outro_idioma_presente',
 'esc_pos','esc_tecnologo','esc_medio','esc_superior_completo','esc_superior_incompleto',
 'area_admin','area_ti','area_financeiro','titulo_admin','titulo_ti',
 'titulo_dados_bi','titulo_financeiro','cert_mos_word','cert_mos_excel','cert_mos_outlook',
 'cert_mos_powerpoint','cert_sap_fi','has_cert','cv_excel_avancado','cv_kpi','cv_controladoria',
 'cv_contabil','cv_financeiro','cv_administrativo','cv_sap','cv_protheus','cv_navision','cv_tamanho_maior_1500'
]

def main():
    eng = make_engine_from_env()
    with eng.begin() as c:
        df = pd.read_sql(text("SELECT * FROM gold_applicants"), c)

    df = df[FEATURES].copy()

    # detecta numéricas x binárias
    numeric = ["salario_valor"]
    binary  = [f for f in FEATURES if f not in numeric]

    stats = {}
    for col in binary:
        s = pd.to_numeric(df[col], errors="coerce")
        stats[col] = {"type":"binary", "rate1": float(np.nanmean(s==1))}
    for col in numeric:
        s = pd.to_numeric(df[col], errors="coerce")
        stats[col] = {"type":"numeric", "mean": float(np.nanmean(s)), "std": float(np.nanstd(s) or 1.0)}

    # guarda em tabela 1 linha por modelo (ou por caminho de artefato)
    payload = json.dumps(stats)
    eng = make_engine_from_env()
    with eng.begin() as c:
        c.execute(text("""
            CREATE TABLE IF NOT EXISTS model_baseline (
              id BIGSERIAL PRIMARY KEY,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              model_path TEXT,
              stats JSONB
            );
        """))
        c.execute(text("""
            INSERT INTO model_baseline (model_path, stats)
            VALUES (:path, CAST(:stats AS JSONB))
        """), {"path": os.getenv("MODEL_ARTIFACT", "./artifacts/modelo_prec80.joblib"),
               "stats": payload})
    print("✅ baseline salvo em model_baseline")

if __name__ == "__main__":
    main()
