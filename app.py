"""Painel de previsão de vendas do varejo de material de construção.

Uso: streamlit run app.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parent
RESULTADOS = RAIZ / "data" / "resultados"
AZUL, LARANJA, CINZA = "#1f4e79", "#e07b00", "#8c8c8c"
MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

st.set_page_config(page_title="Previsão de Vendas · Material de Construção", page_icon="🧱", layout="wide")


@st.cache_data
def carregar():
    serie = pd.read_csv(RAIZ / "data" / "processed" / "material_construcao_brasil.csv",
                        parse_dates=["data"]).set_index("data")["indice"]
    previsao = pd.read_csv(RESULTADOS / "previsao_12_meses.csv", parse_dates=["data"]).set_index("data")
    resumo = pd.read_csv(RESULTADOS / "validacao_resumo.csv", index_col="modelo")
    janelas = pd.read_csv(RESULTADOS / "validacao_previsoes.csv", parse_dates=["corte", "data"])
    info = json.loads((RESULTADOS / "modelo_final.json").read_text())
    return serie, previsao, resumo, janelas, info


serie, previsao, resumo, janelas, info = carregar()
mape_final = resumo.loc["SARIMA", "MAPE"]
ganho = resumo.loc["SARIMA", "ganho_vs_linha_de_base_%"]

# ------------------------------------------------------------------ cabeçalho
st.title("🧱 Previsão de vendas do varejo de material de construção")
st.markdown(
    "Previsão do **volume de vendas** do varejo de material de construção no Brasil para os próximos "
    "12 meses, com base na Pesquisa Mensal de Comércio do IBGE (índice 2022 = 100)."
)

ultimo = serie.index[-1]
var_anual = (serie.iloc[-1] / serie[ultimo - pd.DateOffset(years=1)] - 1) * 100
media_prevista = previsao["previsao"].mean()
media_12m = serie.iloc[-12:].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Último dado ({MESES[ultimo.month - 1]}/{ultimo.year})", f"{serie.iloc[-1]:.1f}",
          f"{var_anual:+.1f}% em 12 meses")
c2.metric("Média prevista (próximos 12 meses)", f"{media_prevista:.1f}",
          f"{(media_prevista / media_12m - 1) * 100:+.1f}% vs. últimos 12 meses")
c3.metric("Erro médio do modelo (MAPE)", f"{mape_final:.1f}%", "validação em 6 janelas", delta_color="off")
c4.metric("Ganho sobre a linha de base", f"{ganho:.0f}%", "menos erro que o sazonal ingênuo", delta_color="off")

aba_prev, aba_saz, aba_mod, aba_met = st.tabs(
    ["📈 Previsão", "📅 Sazonalidade", "🏆 Comparação de modelos", "🧪 Metodologia"])

# ------------------------------------------------------------------ previsão
with aba_prev:
    inicio = st.slider("Mostrar histórico a partir de", int(serie.index.year.min()), int(ultimo.year), 2015)
    hist = serie[str(inicio):]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(previsao.index) + list(previsao.index[::-1]),
                             y=list(previsao.limite_superior) + list(previsao.limite_inferior[::-1]),
                             fill="toself", fillcolor="rgba(224,123,0,0.15)", line=dict(width=0),
                             name="Intervalo de 95%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Histórico", line=dict(color=AZUL, width=2)))
    fig.add_trace(go.Scatter(x=[hist.index[-1], *previsao.index], y=[hist.iloc[-1], *previsao.previsao],
                             name="Previsão (SARIMA)", line=dict(color=LARANJA, width=2.5, dash="dash")))
    fig.update_layout(height=460, hovermode="x unified", yaxis_title="Índice de volume (2022 = 100)",
                      legend=dict(orientation="h", y=1.08), margin=dict(t=30, b=10))
    st.plotly_chart(fig, width="stretch")

    tabela = previsao.copy()
    tabela.index = [f"{MESES[d.month - 1]}/{d.year}" for d in tabela.index]
    tabela.columns = ["Previsão", "Limite inferior (95%)", "Limite superior (95%)"]
    st.dataframe(tabela.style.format("{:.1f}"), width="stretch")
    st.caption("O intervalo de 95% se abre com o horizonte: quanto mais distante o mês, maior a incerteza.")

# ------------------------------------------------------------------ sazonalidade
with aba_saz:
    st.markdown("Como as vendas se comportam ao longo do ano. Cada linha é um ano.")
    anos = sorted(serie.index.year.unique())
    escolhidos = st.multiselect("Anos", anos, default=anos[-4:])
    fig = go.Figure()
    for ano in escolhidos:
        s = serie[str(ano)]
        fig.add_trace(go.Scatter(x=[MESES[m - 1] for m in s.index.month], y=s.values, name=str(ano),
                                 mode="lines+markers"))
    fig.update_layout(height=420, yaxis_title="Índice de volume", hovermode="x unified", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

    tendencia = serie.rolling(12, center=True).mean()
    perfil = ((serie / tendencia).groupby(serie.index.month).mean() - 1) * 100
    fig = go.Figure(go.Bar(x=MESES, y=perfil.values,
                           marker_color=[LARANJA if v > 0 else CINZA for v in perfil.values],
                           text=[f"{v:+.1f}%" for v in perfil.values], textposition="outside"))
    fig.update_layout(height=360, yaxis_title="% acima ou abaixo da tendência", margin=dict(t=30),
                      title="Perfil sazonal médio")
    st.plotly_chart(fig, width="stretch")
    st.info("O segundo semestre concentra as melhores vendas, e fevereiro é o mês mais fraco, em parte por ter "
            "menos dias úteis. Por isso o modelo usa o número de dias úteis e de sábados de cada mês.")

# ------------------------------------------------------------------ modelos
with aba_mod:
    st.markdown(
        "Cada modelo foi treinado só com o passado e testado nos 12 meses seguintes, em **6 janelas** "
        "entre 2023 e 2026. O **MAPE** é o erro percentual médio: quanto menor, melhor.")
    tab = resumo[["MAPE", "RMSE", "ganho_vs_linha_de_base_%"]].copy()
    tab.columns = ["MAPE (%)", "RMSE", "Ganho sobre a linha de base (%)"]
    st.dataframe(tab.style.format("{:.2f}").highlight_min(subset=["MAPE (%)"], color="#d4edda"),
                 width="stretch")

    fig = go.Figure(go.Bar(x=resumo["MAPE"], y=resumo.index, orientation="h",
                           marker_color=[LARANJA if m == "SARIMA" else CINZA for m in resumo.index],
                           text=[f"{v:.2f}%" for v in resumo["MAPE"]], textposition="outside"))
    fig.update_layout(height=320, xaxis_title="MAPE médio (%)", yaxis=dict(autorange="reversed"),
                      margin=dict(t=10, r=40))
    st.plotly_chart(fig, width="stretch")

    corte = st.selectbox("Ver uma janela de teste", sorted(janelas.corte.unique()),
                         format_func=lambda d: f"Treino até {MESES[d.month - 1]}/{d.year}")
    jan = janelas[janelas.corte == corte]
    fig = go.Figure()
    real = jan.drop_duplicates("data").set_index("data")["real"]
    fig.add_trace(go.Scatter(x=real.index, y=real.values, name="Real", line=dict(color="black", width=3)))
    for modelo, g in jan.groupby("modelo"):
        fig.add_trace(go.Scatter(x=g.data, y=g.previsto, name=modelo, mode="lines",
                                 line=dict(width=2.5 if modelo == "SARIMA" else 1.2,
                                           color=LARANJA if modelo == "SARIMA" else None)))
    fig.update_layout(height=420, hovermode="x unified", yaxis_title="Índice de volume", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ metodologia
with aba_met:
    ordem, saz = info["ordem"], info["ordem_sazonal"]
    st.markdown(f"""
**Dados.** Índice de volume de vendas do varejo de material de construção (IBGE, PMC, tabela 8883),
sem ajuste sazonal, de jan/2003 a {MESES[ultimo.month - 1].lower()}/{ultimo.year}.

**Modelos comparados.**
- *Sazonal ingênuo*: repete o mesmo mês do ano anterior. É a linha de base que qualquer modelo precisa superar.
- *Holt-Winters*: suavização exponencial com tendência amortecida e sazonalidade multiplicativa.
- *SARIMA*: modelo ARIMA sazonal sobre o logaritmo da série, com dias úteis e sábados de cada mês como
  variáveis externas. A ordem é escolhida pelo critério AIC.
- *LightGBM*: modelo de machine learning com defasagens, médias móveis e calendário, prevendo a variação anual.
- *Combinação*: média das previsões de Holt-Winters, SARIMA e LightGBM.

**Validação.** Origem móvel: 6 pontos de corte semestrais, de jan/2023 a jul/2025, cada um prevendo
12 meses à frente sem usar nenhum dado do futuro.

**Modelo final.** SARIMA{tuple(ordem)}x{tuple(saz)}, com o menor erro médio na validação.
""")

st.divider()
st.caption("Fonte: IBGE, Pesquisa Mensal de Comércio (tabela 8883). "
           "Projeto de Gabriel de Oliveira · [GitHub](https://github.com/gabrieldeoliver1)")
