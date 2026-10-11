"""Coleta a série de vendas de material de construção direto da API do IBGE (SIDRA).

Os códigos da tabela mudam pouco, mas não são óbvios. Por isso, em vez de fixar códigos,
o script consulta os metadados da tabela e encontra as variáveis e categorias pelo nome.
"""
import json
import urllib.request

import pandas as pd

TABELA = 8883
URL_METADADOS = f"https://servicodados.ibge.gov.br/api/v3/agregados/{TABELA}/metadados"
URL_VALORES = "https://apisidra.ibge.gov.br/values/t/{tabela}/n1/all/v/{variavel}/p/all/{filtros}"

VARIAVEL = "Número-índice (2022=100)"
CATEGORIAS = {"atividade": "Material de construção", "tipo de índice": "volume de vendas"}


def baixar_json(url, tentativas=3):
    """Baixa um JSON, tentando de novo em caso de falha temporária da API."""
    ultimo_erro = None
    for _ in range(tentativas):
        try:
            requisicao = urllib.request.Request(url, headers={"User-Agent": "pipeline-previsao-vendas"})
            with urllib.request.urlopen(requisicao, timeout=120) as resposta:
                return json.loads(resposta.read().decode("utf-8"))
        except Exception as erro:  # noqa: BLE001 - qualquer falha de rede é tentada de novo
            ultimo_erro = erro
    raise RuntimeError(f"Falha ao acessar {url}: {ultimo_erro}")


def localizar_codigos(metadados):
    """Encontra, pelo nome, o código da variável e das categorias desejadas."""
    variaveis = [v for v in metadados["variaveis"] if VARIAVEL in v["nome"] and "ajuste" not in v["nome"].lower()]
    if len(variaveis) != 1:
        raise ValueError(f"Variável não encontrada de forma única: {[v['nome'] for v in metadados['variaveis']]}")

    filtros = []
    for texto in CATEGORIAS.values():
        encontrados = [(c["id"], cat["id"]) for c in metadados["classificacoes"] for cat in c["categorias"]
                       if texto.lower() in cat["nome"].lower()]
        if len(encontrados) != 1:
            raise ValueError(f"Categoria '{texto}' não encontrada de forma única: {encontrados}")
        filtros.append(f"c{encontrados[0][0]}/{encontrados[0][1]}")
    return variaveis[0]["id"], "/".join(filtros)


def converter(valores):
    """Converte a resposta da API SIDRA (primeira linha = cabeçalho) em uma série mensal."""
    cabecalho, linhas = valores[0], valores[1:]
    coluna_mes = next(k for k, v in cabecalho.items() if v.startswith("Mês (Código)"))
    registros = []
    for linha in linhas:
        valor = linha["V"].strip()
        if valor in {"-", "...", "X", ""}:  # Convenções do IBGE para valor indisponível
            continue
        codigo = linha[coluna_mes]  # Formato AAAAMM
        registros.append((pd.Timestamp(int(codigo[:4]), int(codigo[4:]), 1), float(valor)))
    serie = pd.DataFrame(registros, columns=["data", "indice"]).sort_values("data").reset_index(drop=True)
    return serie


def coletar():
    """Devolve a série completa mais recente publicada pelo IBGE."""
    variavel, filtros = localizar_codigos(baixar_json(URL_METADADOS))
    valores = baixar_json(URL_VALORES.format(tabela=TABELA, variavel=variavel, filtros=filtros))
    return converter(valores)
