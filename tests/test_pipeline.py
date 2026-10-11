"""Testes do pipeline de atualização, sem acessar a internet."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline.atualizar import validar  # noqa: E402
from pipeline.coletar import converter, localizar_codigos  # noqa: E402

METADADOS = {
    "variaveis": [
        {"id": 7169, "nome": "PMC - Número-índice (2022=100)"},
        {"id": 7170, "nome": "PMC - Número-índice com ajuste sazonal (2022=100)"},
        {"id": 11708, "nome": "PMC - Variação mês/mês imediatamente anterior, com ajuste sazonal (M/M-1)"},
    ],
    "classificacoes": [
        {"id": 11046, "nome": "Tipos de índice", "categorias": [
            {"id": 56735, "nome": "Índice de receita nominal de vendas no comércio varejista ampliado"},
            {"id": 56736, "nome": "Índice de volume de vendas no comércio varejista ampliado"}]},
        {"id": 85, "nome": "Atividades", "categorias": [
            {"id": 90671, "nome": "Combustíveis e lubrificantes"},
            {"id": 2759, "nome": "Material de construção"}]},
    ],
}

VALORES = [
    {"V": "Valor", "D3C": "Mês (Código)", "D3N": "Mês"},
    {"V": "62.26781", "D3C": "200301", "D3N": "janeiro 2003"},
    {"V": "...", "D3C": "200302", "D3N": "fevereiro 2003"},
    {"V": "59.37559", "D3C": "200303", "D3N": "março 2003"},
]


def test_localiza_codigos_pelo_nome():
    variavel, filtros = localizar_codigos(METADADOS)
    assert variavel == 7169  # Sem ajuste sazonal
    assert filtros == "c85/2759/c11046/56736"


def test_converte_resposta_da_api_e_ignora_valores_indisponiveis():
    serie = converter(VALORES)
    assert list(serie["data"]) == [pd.Timestamp("2003-01-01"), pd.Timestamp("2003-03-01")]
    assert serie["indice"].iloc[0] == pytest.approx(62.26781)


def serie_teste(n=36, inicio="2020-01-01", fator=1.0):
    datas = pd.date_range(inicio, periods=n, freq="MS")
    return pd.DataFrame({"data": datas, "indice": [100 * fator + i for i in range(n)]})


def test_validacao_aceita_serie_com_mes_novo():
    validar(serie_teste(37), serie_teste(36))


def test_validacao_rejeita_mes_faltante():
    nova = serie_teste(37).drop(index=10)
    with pytest.raises(ValueError, match="faltantes"):
        validar(nova, serie_teste(36))


def test_validacao_rejeita_historico_muito_alterado():
    with pytest.raises(ValueError, match="histórico mudou"):
        validar(serie_teste(37, fator=1.2), serie_teste(36))
