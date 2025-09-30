import os, joblib, numpy as np, pandas as pd, sys, datetime as dt
from dotenv import load_dotenv
from sqlalchemy import text
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
import sklearn, lightgbm

from ..utils import threshold_for_min_precision, make_engine_from_env

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

def build_preprocessor():
    num_cols = ["salario_valor"]
    other_cols = [c for c in FEATURES if c not in num_cols]
    return ColumnTransformer(
        transformers=[
            ("sal", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("log1p", FunctionTransformer(np.log1p, validate=False)),
                ("scaler", StandardScaler()),
            ]), num_cols),
            ("resto", SimpleImputer(strategy="most_frequent"), other_cols),
        ],
        remainder="drop",
    )

def train_and_save(min_prec=0.80, artifact_path="artifacts/modelo_prec80.joblib"):
    load_dotenv()
    engine = make_engine_from_env()
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT * FROM gold_applicants"), conn)

    df = df.dropna()
    y = df["target"].astype(int)
    X = df.reindex(columns=FEATURES)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    pre = build_preprocessor()
    lgbm = LGBMClassifier(
        n_estimators=3000, learning_rate=0.03, num_leaves=31,
        min_child_samples=30, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.9, reg_lambda=3.0, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    base_pipe = Pipeline([("pre", pre), ("clf", lgbm)])
    cal = CalibratedClassifierCV(base_pipe, method="isotonic", cv=3)
    cal.fit(X_tr, y_tr)

    p_te = cal.predict_proba(X_te)[:, 1]
    thr = threshold_for_min_precision(y_te, p_te, min_prec)

    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    artifact = {
        "model": cal,
        "feature_columns": FEATURES,
        "threshold": float(thr),
        "operating_mode": f"prec{int(min_prec*100)}",
        "metadata": {
            "python": sys.version.split()[0],
            "sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "created_at": dt.datetime.utcnow().isoformat() + "Z",
        },
    }
    joblib.dump(artifact, artifact_path)
    print(f"✅ Artefato salvo em: {artifact_path} | threshold={thr:.3f}")

if __name__ == "__main__":
    min_prec = float(os.getenv("MIN_PRECISAO", "0.80"))
    path = os.getenv("MODEL_ARTIFACT", "artifacts/modelo_prec80.joblib")
    train_and_save(min_prec=min_prec, artifact_path=path)
