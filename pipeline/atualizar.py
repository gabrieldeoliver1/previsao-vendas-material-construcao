"""Atualização automática: coleta, valida, refaz a previsão e monitora o erro.

Uso: python -m pipeline.atualizar
Executado toda semana pelo GitHub Actions (.github/workflows/atualizar.yml).
"""
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from modelos import ARQUIVO_SERIE  # noqa: E402

RESULTADOS = RAIZ / "data" / "resultados"
STATUS = RESULTADOS / "status_pipeline.json"
HISTORICO = RESULTADOS / "historico_previsoes.csv"
MONITORAMENTO = RESULTADOS / "monitoramento.csv"
README = RAIZ / "README.md"


def validar(nova, atual):
    """Regras de qualidade: a atualização é interrompida se algo estiver fora do esperado."""
    problemas = []
    meses = pd.date_range(nova["data"].min(), nova["data"].max(), freq="MS")
    if len(meses) != len(nova):
        problemas.append("a série tem meses faltantes")
    if (nova["indice"] <= 0).any() or (nova["indice"] > 500).any():
        problemas.append("há valores fora da faixa plausível")
    if len(nova) < len(atual):
        problemas.append("a série nova é menor que a atual")
    # O IBGE revisa os meses recentes, mas o histórico deve ser praticamente o mesmo
    comum = atual.merge(nova, on="data", suffixes=("_atual", "_nova"))
    if not comum.empty:
        diferenca = (comum["indice_nova"] / comum["indice_atual"] - 1).abs()
        if diferenca.iloc[:-12].max() > 0.05:
            problemas.append(f"o histórico mudou mais de 5% (máximo de {diferenca.max():.1%})")
    if problemas:
        raise ValueError("Validação falhou: " + "; ".join(problemas))


def registrar_previsao(ultimo_mes):
    """Guarda cada previsão feita, para comparar com o valor real quando ele for publicado."""
    previsao = pd.read_csv(RESULTADOS / "previsao_12_meses.csv", parse_dates=["data"])
    previsao.insert(0, "ultimo_mes_observado", ultimo_mes)
    historico = pd.read_csv(HISTORICO) if HISTORICO.exists() else pd.DataFrame()
    historico = pd.concat([historico, previsao.astype({"data": str})], ignore_index=True)
    historico = historico.drop_duplicates(["ultimo_mes_observado", "data"], keep="last")
    historico.to_csv(HISTORICO, index=False)


def atualizar_monitoramento(serie):
    """Compara o valor real de cada mês com a previsão feita um mês antes."""
    if not HISTORICO.exists():
        return None
    historico = pd.read_csv(HISTORICO, parse_dates=["ultimo_mes_observado", "data"])
    um_passo = historico[historico["data"] == historico["ultimo_mes_observado"] + pd.offsets.MonthBegin(1)]
    real = serie.rename(columns={"indice": "real"})
    monitor = um_passo.merge(real, on="data")[["data", "previsao", "limite_inferior", "limite_superior", "real"]]
    if monitor.empty:
        return None
    monitor["erro_pct"] = (monitor["previsao"] / monitor["real"] - 1) * 100
    monitor["dentro_do_intervalo"] = monitor["real"].between(monitor["limite_inferior"], monitor["limite_superior"])
    numericas = ["previsao", "limite_inferior", "limite_superior", "real", "erro_pct"]
    monitor[numericas] = monitor[numericas].round(3)
    monitor.to_csv(MONITORAMENTO, index=False, date_format="%Y-%m-%d")
    return monitor.iloc[-1]


def atualizar_readme(serie, ultimo_monitor):
    """Reescreve a seção de atualização automática do README."""
    previsao = pd.read_csv(RESULTADOS / "previsao_12_meses.csv", parse_dates=["data"])
    resumo = pd.read_csv(RESULTADOS / "validacao_resumo.csv", index_col="modelo")
    proxima = previsao.iloc[0]
    meses = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
    nome = lambda d: f"{meses[d.month - 1]}/{d.year}"  # noqa: E731
    num = lambda x: f"{x:.1f}".replace(".", ",")  # noqa: E731  (vírgula decimal)
    linhas = [
        f"- **Última verificação:** {datetime.now(timezone.utc):%d/%m/%Y}",
        f"- **Último mês publicado pelo IBGE:** {nome(serie['data'].iloc[-1])} (índice {num(serie['indice'].iloc[-1])})",
        f"- **Previsão para {nome(proxima['data'])}:** {num(proxima['previsao'])} "
        f"(intervalo de 95%: {num(proxima['limite_inferior'])} a {num(proxima['limite_superior'])})",
        f"- **Erro médio do modelo na validação (MAPE):** {resumo.loc['SARIMA', 'MAPE']:.2f}%".replace(".", ","),
    ]
    if ultimo_monitor is not None:
        dentro = "dentro" if ultimo_monitor["dentro_do_intervalo"] else "fora"
        linhas.append(f"- **Previsão anterior para {nome(ultimo_monitor['data'])}:** {num(ultimo_monitor['previsao'])}; "
                      f"valor real: {num(ultimo_monitor['real'])} (erro de {num(ultimo_monitor['erro_pct'])}%, "
                      f"{dentro} do intervalo de 95%)")
    bloco = "<!-- INICIO_ATUALIZACAO -->\n" + "\n".join(linhas) + "\n<!-- FIM_ATUALIZACAO -->"
    texto = README.read_text(encoding="utf-8")
    README.write_text(re.sub(r"<!-- INICIO_ATUALIZACAO -->.*?<!-- FIM_ATUALIZACAO -->", bloco, texto, flags=re.S),
                      encoding="utf-8")


def salvar_status(**campos):
    campos["executado_em"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    STATUS.write_text(json.dumps(campos, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(campos, ensure_ascii=False, indent=2, default=str))


def main(forcar=False):
    import shutil

    import avaliacao
    import graficos
    from pipeline.coletar import coletar

    atual = pd.read_csv(ARQUIVO_SERIE, parse_dates=["data"])
    try:
        nova = coletar()
        validar(nova, atual)
    except Exception as erro:
        salvar_status(situacao="erro", etapa="coleta e validação", erro=str(erro))
        raise

    tem_mes_novo = nova["data"].max() > atual["data"].max()
    primeira_execucao = not HISTORICO.exists()
    if not (tem_mes_novo or forcar or primeira_execucao):
        salvar_status(situacao="sem novidades", ultimo_mes=nova["data"].max().date())
        return

    try:
        nova.to_csv(ARQUIVO_SERIE, index=False, date_format="%Y-%m-%d")
        shutil.rmtree(avaliacao.CACHE, ignore_errors=True)  # Os dados mudaram: refaz toda a validação
        avaliacao.main()
        graficos.main()
        monitor = atualizar_monitoramento(nova)  # Compara o mês novo com a previsão feita antes
        registrar_previsao(nova["data"].max().date())
        atualizar_readme(nova, monitor)
    except Exception as erro:
        salvar_status(situacao="erro", etapa="previsão", erro=str(erro), detalhe=traceback.format_exc()[-1500:])
        raise
    salvar_status(situacao="atualizado", ultimo_mes=nova["data"].max().date(), meses=len(nova),
                  mes_novo=bool(tem_mes_novo))


if __name__ == "__main__":
    main(forcar="--forcar" in sys.argv)
