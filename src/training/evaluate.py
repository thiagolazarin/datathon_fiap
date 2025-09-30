import os, joblib, numpy as np, pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)

from src.utils import make_engine_from_env

# Mesmas FEATURES usadas no treino
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

def _metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    return {
        "thr": float(thr),
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }

def main():
    load_dotenv()
    artifact_path = os.getenv("MODEL_ARTIFACT", "artifacts/modelo_prec80.joblib")
    art = joblib.load(artifact_path)
    model = art["model"]
    thr_art = float(art["threshold"])
    op_mode = art.get("operating_mode", "custom")

    engine = make_engine_from_env()
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT * FROM gold_applicants"), conn)

    df = df.dropna(subset=["target"]).copy()
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].round().clip(0, 1).astype(int)

    y = df["target"]
    X = df.reindex(columns=FEATURES)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    p_te = model.predict_proba(X_te)[:, 1]

    m05 = _metrics(y_te, p_te, 0.5)
    mart = _metrics(y_te, p_te, thr_art)

    def fmt(m):
        return (
            f"thr={m['thr']:.3f} | "
            f"Acc={m['acc']:.4f}  Prec={m['prec']:.4f}  Rec={m['rec']:.4f}  F1={m['f1']:.4f} | "
            f"ROC AUC={m['roc_auc']:.4f}  PR AUC={m['pr_auc']:.4f} | "
            f"CM={m['cm']}"
        )

    print("\n=== Avaliação (validação holdout) ===")
    print("Corte 0.5  ->", fmt(m05))
    print(f"Corte {op_mode} ->", fmt(mart))

    min_prec = float(os.getenv("MIN_PRECISAO", "0.80"))
    prec, rec, thr = precision_recall_curve(y_te, p_te)
    idx = np.where(prec[:-1] >= min_prec)[0]
    if len(idx):
        thr_p80 = float(thr[idx[0]])
        mp80 = _metrics(y_te, p_te, thr_p80)
        print(f"Corte p/ precisão >= {min_prec:.2f} (thr={thr_p80:.3f}) ->", fmt(mp80))
    else:
        print(f"Não houve threshold que atingisse precisão >= {min_prec:.2f}")

if __name__ == "__main__":
    main()
