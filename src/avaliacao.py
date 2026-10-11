"""Validação temporal dos modelos e geração da previsão final.

A validação simula o uso real: em cada ponto de corte, o modelo é treinado
só com o passado e prevê os 12 meses seguintes, que são comparados com o
que realmente aconteceu. Nada é embaralhado.

Uso: python src/avaliacao.py
"""
import json
import time

import numpy as np
import pandas as pd

from modelos import MODELOS, RAIZ, carregar_serie, sarima

HORIZONTE = 12
N_JANELAS = 6


def definir_cortes(serie):
    """6 pontos de corte semestrais; o último deixa 12 meses de teste até o fim da série."""
    ultimo_corte = serie.index[-1] - pd.DateOffset(months=HORIZONTE)
    return pd.date_range(end=ultimo_corte, periods=N_JANELAS, freq="6MS")
SAIDA = RAIZ / "data" / "resultados"
CACHE = SAIDA / "cache"


def mape(real, previsto):
    return float(np.mean(np.abs((real - previsto) / real)) * 100)


def rmse(real, previsto):
    return float(np.sqrt(np.mean((real - previsto) ** 2)))


def validar(serie):
    previsoes = []
    for corte in definir_cortes(serie):
        treino = serie[:corte]
        teste = serie[corte + pd.offsets.MonthBegin(1):][:HORIZONTE]
        for nome, funcao in MODELOS.items():
            # Cada combinação fica salva em cache, para a validação poder ser retomada
            arquivo = CACHE / f"{corte:%Y-%m}_{nome.split(' (')[0].replace(' ', '_')}.csv"
            if arquivo.exists():
                previsoes.append(pd.read_csv(arquivo, parse_dates=["corte", "data"]))
                continue
            inicio = time.time()
            prev = funcao(treino, HORIZONTE).loc[teste.index]
            resultado = pd.DataFrame({
                "corte": corte, "modelo": nome, "data": teste.index,
                "real": teste.values, "previsto": prev.values,
                "segundos": time.time() - inicio,
            })
            resultado.to_csv(arquivo, index=False)
            previsoes.append(resultado)
    previsoes = pd.concat(previsoes, ignore_index=True)

    # Combinação: média simples das previsões dos modelos estatísticos e de ML
    base = previsoes[previsoes.modelo.isin(["Holt-Winters", "SARIMA", "LightGBM"])]
    combinacao = (base.groupby(["corte", "data"], as_index=False)
                  .agg(real=("real", "first"), previsto=("previsto", "mean"), segundos=("segundos", "sum")))
    combinacao["modelo"] = "Combinação (média dos 3 modelos)"
    return pd.concat([previsoes, combinacao], ignore_index=True)


def resumir(previsoes):
    por_janela = (previsoes.groupby(["modelo", "corte"])
                  .apply(lambda g: pd.Series({"MAPE": mape(g.real, g.previsto),
                                              "RMSE": rmse(g.real, g.previsto)}), include_groups=False)
                  .reset_index())
    resumo = (por_janela.groupby("modelo")
              .agg(MAPE=("MAPE", "mean"), RMSE=("RMSE", "mean"), MAPE_desvio=("MAPE", "std"))
              .sort_values("MAPE"))
    base = resumo.loc["Sazonal ingênuo (linha de base)", "MAPE"]
    resumo["ganho_vs_linha_de_base_%"] = (1 - resumo["MAPE"] / base) * 100
    return resumo.round(2), por_janela


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    serie = carregar_serie()

    previsoes = validar(serie)
    resumo, por_janela = resumir(previsoes)
    previsoes.to_csv(SAIDA / "validacao_previsoes.csv", index=False)
    por_janela.to_csv(SAIDA / "validacao_por_janela.csv", index=False)
    resumo.to_csv(SAIDA / "validacao_resumo.csv")
    print(resumo.to_string())

    # Previsão final para os próximos 12 meses, com o modelo usado no painel (SARIMA)
    media, ic, ajuste = sarima(serie, HORIZONTE, intervalo=True)
    futuro = pd.concat([media.rename("previsao"), ic], axis=1).rename_axis("data")
    futuro.round(2).to_csv(SAIDA / "previsao_12_meses.csv")
    info = {"ordem": list(ajuste.model.order), "ordem_sazonal": list(ajuste.model.seasonal_order),
            "aic": round(float(ajuste.aic), 1), "ultimo_mes": str(serie.index[-1].date()),
            # O modelo usa o log da série: exp(coef) - 1 é o efeito percentual de um dia a mais
            "efeito_dia_util_%": round(float(np.expm1(ajuste.params["dias_uteis"]) * 100), 2),
            "efeito_sabado_%": round(float(np.expm1(ajuste.params["sabados"]) * 100), 2)}
    (SAIDA / "modelo_final.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(futuro.round(1).to_string())
    print(info)


if __name__ == "__main__":
    main()
