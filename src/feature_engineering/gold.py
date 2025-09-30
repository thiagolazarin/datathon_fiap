import io
import math
import pandas as pd
from sqlalchemy import text
from ..utils import make_engine_from_env


def _print_percent(done: int, total: int, last_pct: int) -> int:
    pct = int((done / total) * 100) if total else 100
    if pct > last_pct:
        print(f"\rProgresso: {pct:3d}%", end="", flush=True)
    return pct


def write_gold_with_progress(
    df_gold: pd.DataFrame,
    table: str = "gold_applicants",
    if_exists: str = "replace",
    chunk_rows: int = 50_000,
) -> int:
    """
    Mantida sua função original (recebe um DataFrame completo).
    Usa COPY em chunks e imprime 1..100%.
    """
    if df_gold.empty:
        print(f"Nenhuma linha para inserir em {table}.")
        return 0

    eng = make_engine_from_env()

    # Para a barra ficar suave (~1% cada), adapta chunk_rows ao tamanho
    chunk_rows = min(chunk_rows, max(1, math.ceil(len(df_gold) / 100)))
    with eng.begin() as conn:
        df_gold.head(0).to_sql(table, conn, if_exists=if_exists, index=False)

    raw = eng.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute("SET synchronous_commit = OFF;")
            cols = ", ".join(f'"{c}"' for c in df_gold.columns)
            copy_sql = f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')"

            total = len(df_gold)
            inserted = 0
            last_pct = -1

            print(f"Carregando {total} linhas em '{table}' (chunks ~{chunk_rows})...")
            last_pct = _print_percent(0, total, last_pct)

            for start in range(0, total, chunk_rows):
                chunk = df_gold.iloc[start : start + chunk_rows]
                buf = io.StringIO()
                chunk.to_csv(buf, index=False)
                buf.seek(0)
                cur.copy_expert(copy_sql, buf)
                inserted = min(start + chunk_rows, total)
                last_pct = _print_percent(inserted, total, last_pct)

            raw.commit()
            print("\rProgresso: 100%")
    finally:
        raw.close()

    return len(df_gold)


def build_and_write_gold_streamed(
    applicants_feat_table: str = "applicants_feat",
    prospects_labels_table: str = "prospects_labels",
    gold_table: str = "gold_applicants",
    if_exists: str = "replace",
    chunk_rows: int = 100_000,
) -> int:
    """
    CONSTRUÇÃO STREAMING:
      - Conta linhas do JOIN (para a barra)
      - Cria a tabela destino com o schema correto (SELECT ... LIMIT 0)
      - Lê o JOIN em chunks e grava via COPY, mostrando progresso 1..100%

    Vantagens:
      - Não materializa o JOIN inteiro em memória
      - Usa COPY para desempenho
    """
    eng = make_engine_from_env()

    join_sql = f"""
    SELECT
        a.*,
        l.prospect_situacao_candidado AS status_label,
        l.target
    FROM {applicants_feat_table} AS a
    INNER JOIN {prospects_labels_table} AS l
        ON l.prospect_codigo = a.codigo_profissional
    """

    with eng.begin() as conn:
        total = conn.execute(text(f"SELECT COUNT(*) FROM ({join_sql}) AS q")).scalar() or 0
        if total == 0:
            df_head = pd.read_sql(text(join_sql + " LIMIT 0"), conn)
            df_head.to_sql(gold_table, conn, if_exists=if_exists, index=False)
            print(f"Nenhuma linha no JOIN. '{gold_table}' criada vazia.")
            return 0
        df_head = pd.read_sql(text(join_sql + " LIMIT 0"), conn)
        df_head.to_sql(gold_table, conn, if_exists=if_exists, index=False)

    chunk_rows = min(chunk_rows, max(1, math.ceil(total / 100)))

    raw = eng.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute("SET synchronous_commit = OFF;")
            cols = ", ".join(f'"{c}"' for c in df_head.columns)
            copy_sql = f"COPY {gold_table} ({cols}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')"

            done = 0
            last_pct = -1
            print(f"Construindo '{gold_table}' via JOIN em chunks de ~{chunk_rows} linhas (total={total})...")
            last_pct = _print_percent(0, total, last_pct)

            with eng.connect() as rconn:
                for df_chunk in pd.read_sql(text(join_sql), rconn, chunksize=chunk_rows):
                    buf = io.StringIO()
                    df_chunk.to_csv(buf, index=False)
                    buf.seek(0)
                    cur.copy_expert(copy_sql, buf)

                    done += len(df_chunk)
                    last_pct = _print_percent(done, total, last_pct)

            raw.commit()
            print("\rProgresso: 100%")
    finally:
        raw.close()
        
    with eng.begin() as conn:
        try:
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{gold_table}__cod ON {gold_table}(codigo_profissional)"))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{gold_table}__target ON {gold_table}(target)"))
        except Exception:
            pass
        conn.execute(text(f"ANALYZE {gold_table}"))
        n = conn.execute(text(f"SELECT COUNT(*) FROM {gold_table}")).scalar() or 0

    print(f"✅ '{gold_table}' criado com {n} linhas.")
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--applicants-feat", default="applicants_feat")
    ap.add_argument("--prospects-labels", default="prospects_labels")
    ap.add_argument("--gold-table", default="gold_applicants")
    ap.add_argument("--if-exists", default="replace", choices=["replace", "append"])
    ap.add_argument("--chunk-rows", type=int, default=100_000)
    args = ap.parse_args()
    n = build_and_write_gold_streamed(
        applicants_feat_table=args.applicants_feat,
        prospects_labels_table=args.prospects_labels,
        gold_table=args.gold_table,
        if_exists=args.if_exists,
        chunk_rows=args.chunk_rows,
    )
    print(f"Total inserido: {n}")
