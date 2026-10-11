"""Gera as imagens do README a partir dos resultados mais recentes.

Uso: python src/graficos.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from modelos import RAIZ, carregar_serie  # noqa: E402

RESULTADOS = RAIZ / "data" / "resultados"
DOCS = RAIZ / "docs"


def main():
    DOCS.mkdir(exist_ok=True)
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 11})
    serie = carregar_serie()
    previsao = pd.read_csv(RESULTADOS / "previsao_12_meses.csv", parse_dates=["data"]).set_index("data")

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=150)
    historico = serie[str(serie.index[-1].year - 7):]
    ax.plot(historico.index, historico.values, color="#1f4e79", lw=2, label="Histórico")
    ax.plot([historico.index[-1], *previsao.index], [historico.iloc[-1], *previsao.previsao], color="#e07b00",
            lw=2.5, ls="--", label="Previsão (SARIMA)")
    ax.fill_between(previsao.index, previsao.limite_inferior, previsao.limite_superior, color="#e07b00",
                    alpha=0.15, label="Intervalo de 95%")
    ax.set_title("Volume de vendas do varejo de material de construção: histórico e previsão", loc="left",
                 fontweight="bold")
    ax.set_ylabel("Índice (2022 = 100)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(DOCS / "previsao.png")
    plt.close(fig)

    resumo = pd.read_csv(RESULTADOS / "validacao_resumo.csv", index_col="modelo").sort_values("MAPE", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
    ax.barh(resumo.index, resumo.MAPE, color=["#e07b00" if m == "SARIMA" else "#9aa5b1" for m in resumo.index])
    for i, v in enumerate(resumo.MAPE):
        ax.text(v + 0.05, i, f"{v:.2f}%", va="center")
    ax.set_xlabel("MAPE médio em 6 janelas de teste (%)")
    ax.set_title("Erro de previsão por modelo (menor é melhor)", loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(DOCS / "comparacao_modelos.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
