import io, math
import pandas as pd
from typing import Optional
from sqlalchemy import text
from ..utils import make_engine_from_env

# agrupando o que são aprovados e reprovados
APROVADOS = {
    "Aprovado","Contratado pela Decision","Contratado como Hunting","Proposta Aceita","Encaminhar Proposta",
}
REPROVADOS = {
    "Não Aprovado pelo Cliente","Não Aprovado pelo RH","Não Aprovado pelo Requisitante",
    "Recusado","Desistiu","Desistiu da Contratação","Sem interesse nesta vaga",
}

def _classificar(status: Optional[str]) -> Optional[float]:
    if status in APROVADOS: return 1.0
    if status in REPROVADOS: return 0.0
    return None

def rotulos_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Recebe um chunk de prospects_raw e devolve labels normalizados."""
    situacao = df_raw.get("situacao_candidado")
    if situacao is None:
        situacao = pd.Series([""] * len(df_raw))
    else:
        situacao = situacao.astype(str).str.strip()

    df = pd.DataFrame({
        "prospect_codigo": pd.to_numeric(df_raw.get("codigo"), errors="coerce"),
        "prospect_situacao_candidado": situacao,
    })
    df = df.dropna(subset=["prospect_codigo"]).copy()
    df["prospect_codigo"] = df["prospect_codigo"].astype("Int64")
    df["target"] = df["prospect_situacao_candidado"].apply(_classificar)
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(float)
    return df[["prospect_codigo","prospect_situacao_candidado","target"]]

def _print_percent(done: int, total: int, last_pct: int) -> int:
    pct = int((done / total) * 100) if total else 100
    if pct > last_pct:
        print(f"\rProgresso: {pct:3d}%", end="", flush=True)
    return pct

def build_and_write_prospects_labels(
    raw_table: str = "prospects_raw",
    labels_table: str = "prospects_labels",
    if_exists: str = "replace",
    read_chunk_rows: int = 50_000,
) -> int:
    """
    Lê prospects_raw em chunks, gera labels e grava em prospects_labels via COPY
    (psycopg2), exibindo progresso 1..100%.
    """
    eng = make_engine_from_env()

    # total para barra de progresso
    with eng.begin() as conn:
        total_raw = conn.execute(text(f"SELECT COUNT(*) FROM {raw_table}")).scalar() or 0
    if total_raw == 0:
        print(f"Nenhuma linha em '{raw_table}'.")
        return 0

    # granularidade ~1% por chunk
    read_chunk_rows = min(read_chunk_rows, max(1, math.ceil(total_raw / 100)))

    created = False
    inserted = 0
    processed = 0
    last_pct = -1

    print(f"Lendo {total_raw} linhas de '{raw_table}' em chunks de ~{read_chunk_rows}...")

    raw_conn = eng.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            # um pequeno ganho de desempenho na carga
            cur.execute("SET synchronous_commit = OFF;")

            # stream de leitura em chunks
            with eng.connect() as rconn:
                for df_raw in pd.read_sql(text(f"SELECT * FROM {raw_table}"), rconn, chunksize=read_chunk_rows):
                    processed += len(df_raw)
                    df_lbl = rotulos_from_raw(df_raw)
                    if df_lbl.empty:
                        last_pct = _print_percent(processed, total_raw, last_pct)
                        continue

                    if not created:
                        # criar tabela com schema correto
                        with eng.begin() as c2:
                            df_lbl.head(0).to_sql(labels_table, c2, if_exists=if_exists, index=False)
                        cols = ", ".join(f'"{c}"' for c in df_lbl.columns)
                        copy_sql = f"COPY {labels_table} ({cols}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')"
                        created = True

                    # COPY do chunk
                    buf = io.StringIO()
                    df_lbl.to_csv(buf, index=False)
                    buf.seek(0)
                    cur.copy_expert(copy_sql, buf)

                    inserted += len(df_lbl)
                    last_pct = _print_percent(processed, total_raw, last_pct)

            raw_conn.commit()
            print("\rProgresso: 100%")
    finally:
        raw_conn.close()

    print(f"\n✅ '{labels_table}' escrito com {inserted} linhas (a partir de {total_raw} brutas).")
    return inserted

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-table", default="prospects_raw")
    ap.add_argument("--labels-table", default="prospects_labels")
    ap.add_argument("--if-exists", default="replace", choices=["replace","append"])
    ap.add_argument("--chunk-rows", type=int, default=50_000)
    args = ap.parse_args()
    n = build_and_write_prospects_labels(args.raw_table, args.labels_table, args.if_exists, args.chunk_rows)
    print(f"Total inserido: {n}")
