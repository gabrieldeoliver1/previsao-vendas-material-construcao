"""Carga dos dados, features de calendário e modelos de previsão.

Todos os modelos seguem a mesma interface: recebem a série de treino
(índice mensal) e o horizonte, e devolvem a previsão para os próximos meses.
"""
import calendar
import warnings
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

RAIZ = Path(__file__).resolve().parents[1]
ARQUIVO_SERIE = RAIZ / "data" / "processed" / "material_construcao_brasil.csv"


def carregar_serie(caminho=ARQUIVO_SERIE):
    """Série mensal do índice de volume de vendas (2022 = 100)."""
    df = pd.read_csv(caminho, parse_dates=["data"])
    return df.set_index("data")["indice"].asfreq("MS")


def features_calendario(datas):
    """Dias úteis e sábados de cada mês.

    Meses com mais dias úteis tendem a vender mais; o sábado importa
    porque é um dia forte de vendas para o varejo de material de construção.
    """
    linhas = []
    for d in datas:
        dias = calendar.monthrange(d.year, d.month)[1]
        semana = [calendar.weekday(d.year, d.month, dia) for dia in range(1, dias + 1)]
        linhas.append({
            "dias_uteis": sum(w < 5 for w in semana),
            "sabados": sum(w == 5 for w in semana),
            "dias_no_mes": dias,
        })
    return pd.DataFrame(linhas, index=datas)


def _datas_futuras(serie, horizonte):
    return pd.date_range(serie.index[-1] + pd.offsets.MonthBegin(1), periods=horizonte, freq="MS")


# ---------------------------------------------------------------- modelos

def sazonal_ingenuo(serie, horizonte):
    """Linha de base: repete o valor do mesmo mês do ano anterior."""
    datas = _datas_futuras(serie, horizonte)
    valores = [serie[d - pd.DateOffset(years=1)] if d - pd.DateOffset(years=1) in serie.index
               else serie.iloc[-12 + (i % 12)] for i, d in enumerate(datas)]
    return pd.Series(valores, index=datas)


def holt_winters(serie, horizonte):
    """Suavização exponencial com tendência amortecida e sazonalidade multiplicativa."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = ExponentialSmoothing(
            serie, trend="add", damped_trend=True, seasonal="mul", seasonal_periods=12
        ).fit(optimized=True)
    return modelo.forecast(horizonte)


def _ajustar_sarima(serie, exog=None):
    """Escolhe a melhor ordem SARIMA em uma grade pequena, pelo critério AIC."""
    melhor = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p, q, P, Q in product([0, 1, 2], [0, 1], [0, 1], [0, 1]):
            try:
                ajuste = SARIMAX(np.log(serie), exog=exog, order=(p, 1, q),
                                 seasonal_order=(P, 1, Q, 12)).fit(disp=False)
            except Exception:
                continue
            if melhor is None or ajuste.aic < melhor.aic:
                melhor = ajuste
    return melhor


def sarima_sem_calendario(serie, horizonte):
    """Mesmo SARIMA, sem as variáveis de calendário (para medir o ganho delas)."""
    ajuste = _ajustar_sarima(serie)
    media = np.exp(ajuste.forecast(horizonte))
    media.index = _datas_futuras(serie, horizonte)
    return media


def sarima(serie, horizonte, intervalo=False):
    """SARIMA sobre o log da série, com dias úteis e sábados como variáveis externas."""
    exog = features_calendario(serie.index)[["dias_uteis", "sabados"]]
    ajuste = _ajustar_sarima(serie, exog)
    datas = _datas_futuras(serie, horizonte)
    exog_futuro = features_calendario(datas)[["dias_uteis", "sabados"]]
    previsao = ajuste.get_forecast(horizonte, exog=exog_futuro)
    media = np.exp(previsao.predicted_mean)
    media.index = datas
    if not intervalo:
        return media
    ic = np.exp(previsao.conf_int(alpha=0.05))
    ic.index = datas
    ic.columns = ["limite_inferior", "limite_superior"]
    return media, ic, ajuste


DEFASAGENS = [1, 2, 3, 6, 12, 24]


def _tabela_ml(serie):
    """Monta as variáveis do modelo de machine learning a partir da série."""
    df = pd.DataFrame({"y": serie})
    for d in DEFASAGENS:
        df[f"lag_{d}"] = serie.shift(d)
    df["media_3m"] = serie.shift(1).rolling(3).mean()
    df["media_12m"] = serie.shift(1).rolling(12).mean()
    df["mes"] = serie.index.month
    return df.join(features_calendario(serie.index))


def lightgbm_recursivo(serie, horizonte):
    """LightGBM que prevê a variação anual (log) e avança mês a mês.

    Modelos de árvore não extrapolam tendência, por isso o alvo é a
    diferença entre o log do mês e o log do mesmo mês do ano anterior.
    """
    def alvo(df):
        return np.log(df["y"]) - np.log(df["lag_12"])

    tabela = _tabela_ml(serie).dropna()
    X, y = tabela.drop(columns="y"), alvo(tabela)
    modelo = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=8,
                               min_child_samples=10, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, random_state=42, verbose=-1)
    modelo.fit(X, y)
    historico = serie.copy()
    for data in _datas_futuras(serie, horizonte):
        historico.loc[data] = np.nan
        linha = _tabela_ml(historico.asfreq("MS")).loc[[data]].drop(columns="y")
        variacao = modelo.predict(linha)[0]
        historico.loc[data] = historico[data - pd.DateOffset(years=1)] * np.exp(variacao)
    return historico.iloc[-horizonte:]


MODELOS = {
    "Sazonal ingênuo (linha de base)": sazonal_ingenuo,
    "Holt-Winters": holt_winters,
    "SARIMA": sarima,
    "SARIMA sem calendário": sarima_sem_calendario,
    "LightGBM": lightgbm_recursivo,
}
