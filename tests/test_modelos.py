import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modelos import carregar_serie, features_calendario, holt_winters, sazonal_ingenuo  # noqa: E402
from avaliacao import definir_cortes, mape, rmse  # noqa: E402


@pytest.fixture(scope="module")
def serie():
    return carregar_serie()


def test_serie_mensal_sem_lacunas(serie):
    assert serie.index.freqstr == "MS"
    assert serie.isna().sum() == 0
    assert serie.index.min() == pd.Timestamp("2003-01-01")


def test_dias_uteis_e_sabados():
    cal = features_calendario(pd.DatetimeIndex(["2024-02-01", "2026-08-01"]))
    assert cal.loc["2024-02-01", "dias_no_mes"] == 29  # ano bissexto
    assert cal.loc["2024-02-01", "dias_uteis"] == 21
    assert cal.loc["2026-08-01", "sabados"] == 5


def test_sazonal_ingenuo_repete_ano_anterior(serie):
    treino = serie[:"2024-12-01"]
    prev = sazonal_ingenuo(treino, 12)
    assert prev.index[0] == pd.Timestamp("2025-01-01")
    assert np.allclose(prev.values, treino["2024"].values)


def test_holt_winters_gera_horizonte_completo(serie):
    prev = holt_winters(serie[:"2024-12-01"], 12)
    assert len(prev) == 12
    assert (prev > 0).all()


def test_metricas():
    real, prev = pd.Series([100.0, 200.0]), pd.Series([110.0, 180.0])
    assert mape(real, prev) == pytest.approx(10.0)
    assert rmse(real, prev) == pytest.approx(np.sqrt((100 + 400) / 2))


def test_cortes_deixam_12_meses_de_teste(serie):
    cortes = definir_cortes(serie)
    assert len(cortes) == 6
    assert cortes[-1] == serie.index[-1] - pd.DateOffset(months=12)
