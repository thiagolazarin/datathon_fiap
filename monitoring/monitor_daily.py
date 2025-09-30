import os, json, pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
from src.utils import make_engine_from_env

def _get_time_col(conn):
    q = text("SELECT * FROM inference_log LIMIT 0")
    cols = set(conn.execute(q).keys())
    return "ts" if "ts" in cols else "created_at"

def main():
    eng = make_engine_from_env()
    with eng.begin() as c:
        tcol = _get_time_col(c)

        # janela: último dia
        c.execute(text("SET TIME ZONE 'UTC'"))
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)

        # volume/score
        inf = pd.read_sql(text(f"""
            SELECT date_trunc('hour', {tcol}) as hora, COUNT(*) n, AVG(score) avg_score
            FROM inference_log
            WHERE {tcol} >= now() - interval '24 hours'
            GROUP BY 1 ORDER BY 1
        """), c)
        print("== Volume últimas 24h ==")
        if inf.empty:
            print("Sem predições.")
        else:
            print(inf.to_string(index=False))

        # amostra de payloads de ontem para medir drift de features
        raw = pd.read_sql(text(f"""
            SELECT payload
            FROM inference_log
            WHERE {tcol} >= now() - interval '24 hours'
        """), c)
        if raw.empty:
            print("\n== Drift ==")
            print("Sem dados de ontem.")
            return

        # carrega baseline mais recente
        base = pd.read_sql(text("""
            SELECT stats FROM model_baseline ORDER BY created_at DESC LIMIT 1
        """), c)
    if base.empty:
        print("\n== Drift ==")
        print("Sem baseline salvo. Rode: python -m src.monitoring.record_baseline")
        return

    stats = base["stats"].iloc[0] if isinstance(base["stats"].iloc[0], dict) else json.loads(base["stats"].iloc[0])

    # reconstroi dataframe de features a partir dos payloads
    feats = []
    for j in raw["payload"]:
        d = j if isinstance(j, dict) else json.loads(j)
        feats.append(d)
    df = pd.DataFrame(feats)

    alerts = []
    for col, meta in stats.items():
        if col not in df.columns:
            alerts.append(f"[MISSING] {col} sumiu do serving"); continue
        s = pd.to_numeric(df[col], errors="coerce")

        if meta["type"] == "binary":
            rate = float(np.nanmean(s==1))
            drift = abs(rate - meta["rate1"])
            if drift > 0.15:
                alerts.append(f"[DRIFT BIN] {col}: base={meta['rate1']:.2f} | now={rate:.2f} | Δ={drift:.2f}")
        else:
            m_base, std = meta["mean"], (meta["std"] or 1.0)
            m_now = float(np.nanmean(s))
            z = abs(m_now - m_base) / std
            if z > 3:
                alerts.append(f"[DRIFT NUM] {col}: mean_base={m_base:.2f} | mean_now={m_now:.2f} | z={z:.2f}")

    print("\n== Drift ==")
    if alerts:
        eng = make_engine_from_env()
        with eng.begin() as c:
            for a in alerts:
                feat = a.split()[2].rstrip(':')
                c.execute(text("INSERT INTO drift_alerts (feature, alert) VALUES (:f,:a)"),
                        {"f": feat, "a": a})
    else:
        print("Sem alertas de drift com regras simples.")

if __name__ == "__main__":
    main()
