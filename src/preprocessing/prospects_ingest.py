import argparse, json, pandas as pd
from typing import Dict, Any, List
from ..utils import make_engine_from_env

def read_prospects_json(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        bruto: Dict[str, Any] = json.load(f)
    linhas: List[Dict[str, Any]] = []
    for _, vaga in bruto.items():
        for p in (vaga.get("prospects") or []):
            linhas.append(p)
    return pd.DataFrame(linhas)

def write_prospects_raw(json_path: str, table="prospects_raw", if_exists="replace") -> int:
    df = read_prospects_json(json_path)
    eng = make_engine_from_env()
    with eng.begin() as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False, method="multi")
    return len(df)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--table", default="prospects_raw")
    ap.add_argument("--if-exists", default="replace", choices=["replace","append","fail"])
    args = ap.parse_args()
    n = write_prospects_raw(args.json, args.table, args.if_exists)
    print(f"✅ prospects_raw: {n} linhas")
