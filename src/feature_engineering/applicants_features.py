import io, math, re, pandas as pd
from typing import Any, Dict, Optional
from sqlalchemy import text
from ..utils import make_engine_from_env

DOMINIOS_EMAIL_GRATIS = {"gmail.com","hotmail.com","yahoo.com","outlook.com","live.com","icloud.com","bol.com.br","uol.com.br","terra.com.br"}
MAP_ING = {"nenhum":"nenhum","básico":"basico","basico":"basico","intermediário":"intermediario","intermediario":"intermediario","avançado":"avancado","avancado":"avancado"}
MAP_ESP = MAP_ING
MAP_ESCOLARIDADE = {"ensino superior incompleto":"superior_incompleto","ensino superior completo":"superior_completo","pós-graduação":"pos","pos-graduação":"pos","pos":"pos","tecnólogo":"tecnologo","ensino médio":"medio"}
PALAVRAS_CHAVE_AREA = {"Administrativa":"admin","Financeira":"financeiro","Financeiro":"financeiro","TI":"ti","Tecnologia":"ti"}
PALAVRAS_CHAVE_TITULO_OBJ = {"administr":"admin","finance":"financeiro","bi":"dados_bi","dados":"dados_bi","analist":"admin","ti":"ti"}
PALAVRAS_CHAVE_CERT = {r"\b77-418\b":"cert_mos_word", r"\b77-420\b":"cert_mos_excel", r"\b77-423\b":"cert_mos_outlook", r"\b77-422\b":"cert_mos_powerpoint", r"\bsap\s*fi\b":"cert_sap_fi"}
PALAVRAS_CHAVE_CV = {r"\bexcel\s+avancado\b":"cv_excel_avancado", r"\bkpi":"cv_kpi", r"\bcontrolador":"cv_controladoria", r"\bcontab":"cv_contabil", r"\bfinanceir":"cv_financeiro", r"\badministr":"cv_administrativo", r"\bsap\b":"cv_sap", r"\bprotheus\b":"cv_protheus", r"\bnavision\b":"cv_navision"}

def _norm(s: Optional[str]) -> str: return (s or "").strip()
def _so_digitos(s: str) -> str: return re.sub(r"\D+", "", s or "")
def _dominio_email(email: str) -> Optional[str]:
    email = _norm(email).lower(); m = re.search(r"@([^@\s]+)$", email); return m.group(1) if m else None
def _eh_dominio_corporativo(dominio: Optional[str]) -> Optional[int]:
    if not dominio: return None
    return 0 if dominio in DOMINIOS_EMAIL_GRATIS else 1
def _parse_salario(s: str) -> Optional[float]:
    if not s: return None
    txt = re.sub(r"[R$\s]", "", s, flags=re.I).replace(".", "").replace(",", ".")
    try: val = float(txt); return val if val > 0 else None
    except: return None
def _map_idioma(nivel_raw: str, mapping: Dict[str, str]) -> str:
    s = _norm(nivel_raw).lower()
    s = (s.replace("á","a").replace("ã","a").replace("â","a").replace("í","i").replace("ó","o").replace("ô","o").replace("é","e").replace("ê","e").replace("ç","c"))
    for k,v in mapping.items():
        if k in s: return v
    return "nenhum" if not s or s in {"-","nenhum"} else "outro"
def _escolaridade_onehot(raw: str) -> Dict[str,int]:
    s = _norm(raw).lower()
    s = (s.replace("á","a").replace("ã","a").replace("â","a").replace("í","i").replace("ó","o").replace("ô","o").replace("é","e").replace("ê","e").replace("ç","c"))
    chave=None
    for k,v in MAP_ESCOLARIDADE.items():
        if k in s: chave=v; break
    onehot = {f"esc_{v}":0 for v in set(MAP_ESCOLARIDADE.values())}
    if chave: onehot[f"esc_{chave}"]=1
    return onehot
def _bool_int(cond: bool) -> int: return 1 if cond else 0

def construir_features_candidatos_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Transforma um chunk de applicants_raw (json_normalize) em applicants_feat (features)."""
    g = lambda c: df_raw.get(c, pd.Series([None]*len(df_raw)))

    codigo_prof   = pd.to_numeric(g("infos_basicas.codigo_profissional"), errors="coerce")
    email         = g("infos_basicas.email").fillna(g("informacoes_pessoais.email")).astype(str)
    telefone      = g("infos_basicas.telefone").fillna(g("informacoes_pessoais.telefone_celular")).astype(str)
    linkedin      = g("informacoes_pessoais.url_linkedin").astype(str)
    local         = g("infos_basicas.local").astype(str)
    objetivo      = g("infos_basicas.objetivo_profissional").astype(str)
    titulo_prof   = g("informacoes_profissionais.titulo_profissional").astype(str)
    area_atuacao  = g("informacoes_profissionais.area_atucao").fillna(g("informacoes_profissionais.area_atuacao")).astype(str)  # tolera nome errado
    remuneracao   = g("informacoes_profissionais.remuneracao").astype(str)
    nivel_acad    = g("formacao_e_idiomas.nivel_academico").astype(str)
    nivel_ing     = g("formacao_e_idiomas.nivel_ingles").astype(str)
    nivel_esp     = g("formacao_e_idiomas.nivel_espanhol").astype(str)
    outro_idioma  = g("formacao_e_idiomas.outro_idioma").astype(str)
    certificacoes = g("informacoes_profissionais.certificacoes").astype(str)
    outras_cert   = g("informacoes_profissionais.outras_certificacoes").astype(str)
    conhecimentos = g("informacoes_profissionais.conhecimentos_tecnicos").astype(str)
    cv_pt         = g("cv_pt").astype(str)

    linhas = []
    for i in range(len(df_raw)):
        dom = _dominio_email(email.iloc[i])
        email_corp = _eh_dominio_corporativo(dom)
        tem_email = _bool_int(bool(email.iloc[i] and email.iloc[i]!="nan"))
        tem_tel   = _bool_int(bool(_so_digitos(telefone.iloc[i])))
        tem_link  = _bool_int(bool(linkedin.iloc[i]))
        tem_local = _bool_int(bool(local.iloc[i]))
        tem_obj   = _bool_int(bool(objetivo.iloc[i]))

        ing = _map_idioma(nivel_ing.iloc[i], MAP_ING)
        esp = _map_idioma(nivel_esp.iloc[i], MAP_ESP)
        idiomas = {
            "ingl_nenhum":0,"ingl_basico":0,"ingl_intermediario":0,"ingl_avancado":0,"ingl_outro":0,
            "esp_nenhum":0,"esp_basico":0,"esp_intermediario":0,"esp_avancado":0,"esp_outro":0,
            "outro_idioma_presente": _bool_int(bool(_norm(outro_idioma.iloc[i]) and _norm(outro_idioma.iloc[i])!="-")),
        }
        idiomas[f"ingl_{ing if ing in {'nenhum','basico','intermediario','avancado'} else 'outro'}"]=1
        idiomas[f"esp_{esp if esp in {'nenhum','basico','intermediario','avancado'} else 'outro'}"]=1

        escol = _escolaridade_onehot(nivel_acad.iloc[i])

        area = {f"area_{v}":0 for v in {"admin","financeiro","ti"}}
        for k,v in PALAVRAS_CHAVE_AREA.items():
            if k.lower() in _norm(area_atuacao.iloc[i]).lower():
                area[f"area_{v}"]=1

        titulo = {f"titulo_{v}":0 for v in {"admin","financeiro","dados_bi","ti"}}
        blob_titulo = f"{_norm(titulo_prof.iloc[i])} {_norm(objetivo.iloc[i])}".lower()
        for k,v in PALAVRAS_CHAVE_TITULO_OBJ.items():
            if k in blob_titulo:
                titulo[f"titulo_{v}"]=1

        texto_cert = f"{_norm(certificacoes.iloc[i])} {_norm(outras_cert.iloc[i])}".lower()
        certs = {v:0 for v in PALAVRAS_CHAVE_CERT.values()}
        certs["has_cert"] = _bool_int(bool(texto_cert))
        for padrao, col in PALAVRAS_CHAVE_CERT.items():
            if re.search(padrao, texto_cert, flags=re.I):
                certs[col]=1

        blob_cv = f"{_norm(cv_pt.iloc[i])} {_norm(conhecimentos.iloc[i])}".lower()
        blob_cv = (blob_cv.replace("ç","c").replace("á","a").replace("ã","a").replace("â","a")
                           .replace("í","i").replace("ó","o").replace("ô","o")
                           .replace("é","e").replace("ê","e"))
        cv_feats = {v:0 for v in PALAVRAS_CHAVE_CV.values()}
        for padrao, col in PALAVRAS_CHAVE_CV.items():
            if re.search(padrao, blob_cv, flags=re.I):
                cv_feats[col]=1
        cv_feats["cv_tamanho_maior_1500"] = _bool_int(len(_norm(cv_pt.iloc[i]))>1500)

        salario_valor = _parse_salario(remuneracao.iloc[i])

        linha = {
            "codigo_profissional": pd.to_numeric(codigo_prof.iloc[i], errors="coerce"),
            "tem_email": tem_email, "tem_telefone": tem_tel, "tem_linkedin": tem_link,
            "tem_local": tem_local, "tem_objetivo": tem_obj,
            "email_corporativo": 0 if email_corp is None else int(email_corp),
            "salario_valor": salario_valor,
        }
        linha.update(idiomas); linha.update(escol); linha.update(area); linha.update(titulo); linha.update(certs); linha.update(cv_feats)
        linhas.append(linha)

    df = pd.DataFrame(linhas)
    df = df.dropna(subset=["codigo_profissional"]).copy()
    df["codigo_profissional"] = df["codigo_profissional"].astype("Int64")
    cols_bin = [c for c in df.columns if c not in {"codigo_profissional","email_corporativo","salario_valor"}]
    df[cols_bin] = df[cols_bin].fillna(0).astype(int)
    df["email_corporativo"] = df["email_corporativo"].fillna(0).astype(int)
    df["salario_valor"] = pd.to_numeric(df["salario_valor"], errors="coerce")
    return df

def _print_percent(done: int, total: int, last_pct: int) -> int:
    pct = int((done/total)*100) if total else 100
    if pct > last_pct:
        print(f"\rProgresso: {pct:3d}%", end="", flush=True)
    return pct

def build_and_write_applicants_feat(
    raw_table: str = "applicants_raw",
    feat_table: str = "applicants_feat",
    if_exists: str = "replace",
    read_chunk_rows: int = 50_000
) -> int:
    """
    Lê applicants_raw em chunks, transforma e grava applicants_feat via COPY,
    exibindo progresso percentual (1..100%).
    """
    eng = make_engine_from_env()
    total_raw = 0
    with eng.begin() as conn:
        total_raw = conn.execute(text(f"SELECT COUNT(*) FROM {raw_table}")).scalar() or 0
    if total_raw == 0:
        print(f"Nenhuma linha em {raw_table}."); return 0

    read_chunk_rows = min(read_chunk_rows, max(1, math.ceil(total_raw/100)))

    created = False
    inserted_feat = 0
    processed_raw = 0
    last_pct = -1

    print(f"Lendo {total_raw} linhas de '{raw_table}' em chunks de ~{read_chunk_rows}...")

    raw_conn = eng.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            cur.execute("SET synchronous_commit = OFF;")

            with eng.connect() as conn:
                for df_raw in pd.read_sql(text(f"SELECT * FROM {raw_table}"), conn, chunksize=read_chunk_rows):
                    processed_raw += len(df_raw)

                    df_feat = construir_features_candidatos_from_raw(df_raw)

                    if not created:
                        with eng.begin() as c2:
                            df_feat.head(0).to_sql(feat_table, c2, if_exists=if_exists, index=False)
                        cols = ", ".join(f'"{c}"' for c in df_feat.columns)
                        copy_sql = f"COPY {feat_table} ({cols}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')"
                        created = True

                    buf = io.StringIO()
                    df_feat.to_csv(buf, index=False)
                    buf.seek(0)
                    cur.copy_expert(copy_sql, buf)
                    inserted_feat += len(df_feat)

                    last_pct = _print_percent(processed_raw, total_raw, last_pct)

            raw_conn.commit()
            print("\rProgresso: 100%")
    finally:
        raw_conn.close()

    print(f"\n✅ '{feat_table}' escrito com {inserted_feat} linhas (a partir de {total_raw} brutas).")
    return inserted_feat

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-table",  default="applicants_raw")
    ap.add_argument("--feat-table", default="applicants_feat")
    ap.add_argument("--if-exists",  default="replace", choices=["replace","append"])
    ap.add_argument("--chunk-rows", type=int, default=50_000)
    args = ap.parse_args()
    n = build_and_write_applicants_feat(args.raw_table, args.feat_table, args.if_exists, args.chunk_rows)
    print(f"Total inserido: {n}")
