"""Converte a tabela 8883 do IBGE (formato largo) na série mensal usada no projeto.

Uso: python src/preparar_dados.py
"""
import csv

import pandas as pd

from modelos import ARQUIVO_SERIE, RAIZ

ARQUIVO_BRUTO = RAIZ / "data" / "raw" / "tabela8883.csv"
MESES = {"janeiro": 1, "fevereiro": 2, "março": 3, "abril": 4, "maio": 5, "junho": 6, "julho": 7,
         "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12}
SEM_VALOR = {"-", "...", "X", ""}  # convenções do IBGE para valor inexistente ou indisponível


def preparar():
    linhas = list(csv.reader(open(ARQUIVO_BRUTO, encoding="utf-8-sig"), delimiter=";"))
    cabecalho = next(l for l in linhas if len(l) > 2 and l[2] == "janeiro 2000")[2:]
    valores = next(l for l in linhas if len(l) > 2 and l[1] == "Material de construção")[2:]

    registros = []
    for mes_ano, valor in zip(cabecalho, valores):
        mes, ano = mes_ano.split()
        if valor.strip() not in SEM_VALOR:
            registros.append((pd.Timestamp(int(ano), MESES[mes], 1), float(valor.replace(",", "."))))

    serie = pd.DataFrame(registros, columns=["data", "indice"])
    esperado = len(pd.date_range(serie.data.min(), serie.data.max(), freq="MS"))
    assert len(serie) == esperado, "A série tem meses faltantes"
    ARQUIVO_SERIE.parent.mkdir(parents=True, exist_ok=True)
    serie.to_csv(ARQUIVO_SERIE, index=False)
    print(f"{len(serie)} meses salvos em {ARQUIVO_SERIE.relative_to(RAIZ)}")


if __name__ == "__main__":
    preparar()
