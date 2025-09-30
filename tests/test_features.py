import pandas as pd
from src.feature_engineering.applicants_features import construir_features_candidatos_from_raw

def test_build_minimal_row():
    raw = pd.DataFrame([{
        "infos_basicas.codigo_profissional": "31001",
        "infos_basicas.email": "a@empresa.com",
        "informacoes_profissionais.remuneracao": "3000",
        "formacao_e_idiomas.nivel_ingles": "Básico",
    }])
    df = construir_features_candidatos_from_raw(raw)
    assert "codigo_profissional" in df.columns
    assert "tem_email" in df.columns
    assert "salario_valor" in df.columns
    assert len(df) == 1
