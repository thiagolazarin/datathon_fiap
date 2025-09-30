# monitoring/streamlit_app.py
import os, json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text
from dotenv import load_dotenv

from src.utils import make_engine_from_env

load_dotenv()

st.set_page_config(page_title="Monitoring - Model Drift", layout="wide")
st.title("📊 Monitoring — Drift & Volume")

def load_baseline(conn):
    base = pd.read_sql(text("SELECT stats, created_at FROM model_baseline ORDER BY created_at DESC LIMIT 1"), conn)
    if base.empty:
        return None, None
    raw_stats = base["stats"].iloc[0]
    stats = raw_stats if isinstance(raw_stats, dict) else json.loads(raw_stats)
    return stats, base["created_at"].iloc[0]

def get_time_col(conn):
    cols = set(conn.execute(text("SELECT * FROM inference_log LIMIT 0")).keys())
    return "ts" if "ts" in cols else "created_at"

def drift_alerts_from_payloads(df_payloads, baseline_stats):
    alerts = []
    feats = []
    for j in df_payloads["payload"]:
        d = j if isinstance(j, dict) else json.loads(j)
        feats.append(d)
    if not feats:
        return ["Sem payloads nas últimas 24h."]
    df = pd.DataFrame(feats)

    for col, meta in baseline_stats.items():
        if col not in df.columns:
            alerts.append(f"[MISSING] {col} sumiu do serving"); 
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if meta["type"] == "binary":
            rate = float(np.nanmean(s == 1))
            drift = abs(rate - meta["rate1"])
            if drift > 0.15:
                alerts.append(f"[DRIFT BIN] {col}: base={meta['rate1']:.2f} | now={rate:.2f} | Δ={drift:.2f}")
        else:
            m_base = float(meta["mean"])
            std = float(meta["std"] or 1.0)
            m_now = float(np.nanmean(s))
            z = abs(m_now - m_base) / (std if std > 0 else 1.0)
            if z > 3:
                alerts.append(f"[DRIFT NUM] {col}: mean_base={m_base:.2f} | mean_now={m_now:.2f} | z={z:.2f}")
    return alerts

#UI 
with st.sidebar:
    st.header("Config")
    st.caption("As credenciais de banco são lidas do .env ou variáveis do container (POSTGRES_*).")
    lookback_hours = st.slider("Janela (horas) para volume/score", 6, 48, 24, 1)

#queries 
eng = make_engine_from_env()
with eng.begin() as c:
    tcol = get_time_col(c)

    # volume & score por hora
    q_vol = text(f"""
        SELECT date_trunc('hour', {tcol}) AS hora,
               COUNT(*) AS n_preds,
               AVG(score) AS avg_score
        FROM inference_log
        WHERE {tcol} >= now() - interval :win
        GROUP BY 1
        ORDER BY 1
    """)
    inf = pd.read_sql(q_vol, c, params={"win": f"{lookback_hours} hours"})

    # payloads das últimas 24h (fixo p/ drift)
    q_payloads = text(f"""
        SELECT payload, {tcol}
        FROM inference_log
        WHERE {tcol} >= now() - interval '24 hours'
        ORDER BY {tcol} ASC
    """)
    payloads = pd.read_sql(q_payloads, c)

    baseline_stats, baseline_dt = load_baseline(c)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Predições no período", int(inf["n_preds"].sum()) if not inf.empty else 0)
with col2:
    st.metric("Média de score (janela)", f"{float(inf['avg_score'].mean()):.3f}" if not inf.empty else "—")
with col3:
    st.metric("Baseline registrada em", str(baseline_dt) if baseline_dt is not None else "—")

#graficos 
st.subheader("Volume por hora")
if inf.empty:
    st.info("Sem predições no período.")
else:
    vdf = inf[["hora", "n_preds"]].set_index("hora")
    st.line_chart(vdf, use_container_width=True)

st.subheader("Score médio por hora")
if inf.empty:
    st.info("Sem predições no período.")
else:
    sdf = inf[["hora", "avg_score"]].set_index("hora")
    st.line_chart(sdf, use_container_width=True)

#drift 
st.subheader("Alertas de Drift (últimas 24h)")
if baseline_stats is None:
    st.warning("Nenhuma baseline encontrada em model_baseline. Rode primeiro o script de baseline.")
else:
    if payloads.empty:
        st.info("Sem dados nas últimas 24h.")
    else:
        alerts = drift_alerts_from_payloads(payloads, baseline_stats)
        if alerts:
            st.error("Foram encontrados alertas:")
            st.dataframe(pd.DataFrame({"alerta": alerts}))
        else:
            st.success("Sem alertas de drift com as regras simples.")
