import argparse, json, io, math, sys, pandas as pd
from typing import Dict, Any
from ..utils import make_engine_from_env

def read_applicants_json(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)
    rows = [bloco for _, bloco in raw.items()]
    return pd.json_normalize(rows)

def _print_percent(inserted: int, total: int, last_pct: int) -> int:
    pct = int((inserted / total) * 100) if total else 100
    if pct > last_pct:
        print(f"\rProgresso: {pct:3d}%", end="", flush=True)
    return pct

def write_applicants_raw_fast(
    json_path: str,
    table: str = "applicants_raw",
    if_exists: str = "replace",       
    chunk_rows: int = 50_000           
) -> int:
    df = read_applicants_json(json_path)
    total = len(df)
    if total == 0:
        print(f"Nenhuma linha para inserir em {table}.")
        return 0

    # se total < 100 chunks, diminui chunk_rows para tentar ~1% por chunk
    # (isso dá mais “passos” na barra)
    chunk_rows = min(chunk_rows, max(1, math.ceil(total / 100)))

    eng = make_engine_from_env()

    with eng.begin() as conn:
        df.head(0).to_sql(table, conn, if_exists=if_exists, index=False)

    raw_conn = eng.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            cur.execute("SET synchronous_commit = OFF;")
            cols = list(df.columns)
            cols_quoted = ", ".join(f'"{c}"' for c in cols)
            copy_sql = f"COPY {table} ({cols_quoted}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')"

            inserted = 0
            last_pct = -1
            num_chunks = math.ceil(total / chunk_rows)

            print(f"Carregando {total} linhas em '{table}' (chunks de ~{chunk_rows}):")
            _print_percent(0, total, last_pct); last_pct = 0

            for i in range(num_chunks):
                start = i * chunk_rows
                end = min(start + chunk_rows, total)
                chunk = df.iloc[start:end]

                buf = io.StringIO()
                chunk.to_csv(buf, index=False)
                buf.seek(0)
                cur.copy_expert(copy_sql, buf)

                inserted = end
                last_pct = _print_percent(inserted, total, last_pct)

            raw_conn.commit()
            print("\rProgresso: 100%")  # garante 100% no fim
    finally:
        raw_conn.close()

    return total

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Caminho do applicants.json")
    ap.add_argument("--table", default="applicants_raw")
    ap.add_argument("--if-exists", default="replace", choices=["replace","append"])
    ap.add_argument("--chunk-rows", type=int, default=50_000)
    args = ap.parse_args()

    n = write_applicants_raw_fast(
        json_path=args.json,
        table=args.table,
        if_exists=args.if_exists,
        chunk_rows=args.chunk_rows
    )
    print(f"✅ applicants_raw: {n} linhas")
