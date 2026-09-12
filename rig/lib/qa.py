#!/usr/bin/env python3
"""
qa.py  -  51 perguntas de fato conhecido, gabarito verificado contra o dado.

duas familias:
  A  fato single-system: a query descreve o registro pelo rotulo, a resposta e o
     codigo. alvo de recuperacao = aquele registro. cobre todos os sistemas.
  B  cross-tabela pelo hub de pais (IOC): a query da uma chave de juncao que NAO
     e o codigo primario do registro (ISO alpha-3 ou alpha-2, que vivem em extra.iso3
     e extra.iso2), e a resposta e outro campo do mesmo registro-hub
     (TLD ou capital).

cada pergunta carrega (q, target_system, target_code, answer, family). a
construcao FALHA em voz alta se um gabarito nao resolver no corpus - um item de
gabarito que some seria um erro de medicao, nao um detalhe a engolir.
"""
import catalog

# entidades escolhidas por serem inequivocas e de fato conhecido.
A_ITEMS = [
    ("trp/iata", "GRU", "Aeroporto Internacional de São Paulo-Guarulhos", "Qual o código IATA do {}?"),
    ("trp/iata", "CGH", "Aeroporto de São Paulo-Congonhas", "Qual o código IATA do {}?"),
    ("trp/iata", "JFK", "John F Kennedy International Airport", "Qual o código IATA do {}?"),
    ("trp/iata", "LHR", "London Heathrow Airport", "Qual o código IATA do {}?"),
    ("trp/iata", "HND", "Tokyo Haneda International Airport", "Qual o código IATA do {}?"),
    ("trp/iata", "CDG", "Paris Charles de Gaulle Airport", "Qual o código IATA do {}?"),
    ("trp/iata", "FRA", "Frankfurt Airport", "Qual o código IATA do {}?"),
    ("trp/iata", "DXB", "Dubai International Airport", "Qual o código IATA do {}?"),
    ("trp/icao", "SBGR", "Aeroporto Internacional de São Paulo-Guarulhos", "Qual o código ICAO do {}?"),
    ("trp/icao", "EGLL", "London Heathrow Airport", "Qual o código ICAO do {}?"),
    ("trp/icao", "KJFK", "John F Kennedy International Airport", "Qual o código ICAO do {}?"),
    ("trp/icao", "EDDF", "Frankfurt Airport", "Qual o código ICAO do {}?"),
    ("cls/elm", "Na", "Sódio", "Qual o símbolo químico do elemento {}?"),
    ("cls/elm", "Fe", "Ferro", "Qual o símbolo químico do elemento {}?"),
    ("cls/elm", "Au", "Ouro", "Qual o símbolo químico do elemento {}?"),
    ("cls/elm", "O", "Oxigênio", "Qual o símbolo químico do elemento {}?"),
    ("cls/elm", "K", "Potássio", "Qual o símbolo químico do elemento {}?"),
    ("cls/elm", "He", "Hélio", "Qual o símbolo químico do elemento {}?"),
    ("cls/elm", "Cl", "Cloro", "Qual o símbolo químico do elemento {}?"),
    ("cmp/sts", "404", "Not Found", "Qual o código de status HTTP para '{}'?"),
    ("cmp/sts", "500", "Internal Server Error", "Qual o código de status HTTP para '{}'?"),
    ("cmp/sts", "301", "Moved Permanently", "Qual o código de status HTTP para '{}'?"),
    ("cmp/sts", "403", "Forbidden", "Qual o código de status HTTP para '{}'?"),
    ("cmp/sts", "200", "OK", "Qual o código de status HTTP para '{}'?"),
    ("lng/i31", "BRA", "Brazil", "Qual o código ISO 3166 alpha-3 do país {}?"),
    ("lng/i31", "JPN", "Japan", "Qual o código ISO 3166 alpha-3 do país {}?"),
    ("lng/i31", "DEU", "Germany", "Qual o código ISO 3166 alpha-3 do país {}?"),
    ("lng/i31", "FRA", "France", "Qual o código ISO 3166 alpha-3 do país {}?"),
    ("fin/i42", "BRL", "Brazilian Real", "Qual o código ISO 4217 da moeda {}?"),
    ("fin/i42", "JPY", "Yen", "Qual o código ISO 4217 da moeda {}?"),
    ("fin/i42", "CHF", "Swiss Franc", "Qual o código ISO 4217 da moeda {}?"),
    ("fin/i42", "GBP", "Pound Sterling", "Qual o código ISO 4217 da moeda {}?"),
    ("bio/aa3", "Met", "Methionine", "Qual o código de três letras do aminoácido {}?"),
    ("bio/aa3", "Gly", "Glycine", "Qual o código de três letras do aminoácido {}?"),
    ("bio/aa1", "M", "Methionine", "Qual o código de uma letra do aminoácido {}?"),
    ("bio/aa1", "W", "Tryptophan", "Qual o código de uma letra do aminoácido {}?"),
    ("com/mor", "S", "S", "Qual o código Morse da letra '{}'?"),
    ("com/mor", "Q", "Q", "Qual o código Morse da letra '{}'?"),
]

# familia B: (chave de juncao dada na query, tipo da chave, codigo IOC alvo, campo-resposta, texto)
B_ITEMS = [
    ("BRA", "alpha-3 ISO 3166", "BRA", "tld", "Qual o TLD do país cujo código alpha-3 é {}?"),
    ("DEU", "alpha-3 ISO 3166", "GER", "capital", "Qual a capital do país cujo código alpha-3 é {}?"),
    ("JPN", "alpha-3 ISO 3166", "JPN", "tld", "Qual o TLD do país cujo código alpha-3 é {}?"),
    ("FRA", "alpha-3 ISO 3166", "FRA", "capital", "Qual a capital do país cujo código alpha-3 é {}?"),
    ("GBR", "alpha-3 ISO 3166", "GBR", "tld", "Qual o TLD do país cujo código alpha-3 é {}?"),
    ("CH", "alpha-2 ISO 3166", "SUI", "capital", "Qual a capital do país cujo código alpha-2 é {}?"),
    ("BR", "alpha-2 ISO 3166", "BRA", "capital", "Qual a capital do país cujo código alpha-2 é {}?"),
    ("IT", "alpha-2 ISO 3166", "ITA", "capital", "Qual a capital do país cujo código alpha-2 é {}?"),
    ("USA", "alpha-3 ISO 3166", "USA", "tld", "Qual o TLD do país cujo código alpha-3 é {}?"),
    ("CAN", "alpha-3 ISO 3166", "CAN", "capital", "Qual a capital do país cujo código alpha-3 é {}?"),
    ("AUS", "alpha-3 ISO 3166", "AUS", "capital", "Qual a capital do país cujo código alpha-3 é {}?"),
    ("ARG", "alpha-3 ISO 3166", "ARG", "tld", "Qual o TLD do país cujo código alpha-3 é {}?"),
    ("MEX", "alpha-3 ISO 3166", "MEX", "tld", "Qual o TLD do país cujo código alpha-3 é {}?"),
]


def build():
    recs = catalog.load_all()
    by = {n: {} for n in recs}
    for n, rs in recs.items():
        for r in rs:
            by[n].setdefault(r.code, r)        # primeiro registro com cada codigo

    qs, problems = [], []

    for sysid, code, label, tmpl in A_ITEMS:
        r = by[sysid].get(code)
        if r is None:
            problems.append(f"A: {sysid}:{code} ({label}) ausente no corpus")
            continue
        qs.append({"q": tmpl.format(label), "target_system": sysid,
                   "target_code": code, "answer": code, "family": "A", "label": label})

    for joinval, joinkind, ioc_code, field, tmpl in B_ITEMS:
        r = by["lng/ioc"].get(ioc_code)
        if r is None:
            problems.append(f"B: hub IOC {ioc_code} ausente")
            continue
        joined = joinval in (r.f("iso3"), r.f("iso2"))
        if not joined:
            problems.append(f"B: chave {joinval} nao resolve no registro IOC {ioc_code}")
            continue
        ans = r.f(field)
        if not ans:
            problems.append(f"B: {ioc_code}.{field} vazio (juncao {joinval})")
            continue
        qs.append({"q": tmpl.format(joinval), "target_system": "lng/ioc",
                   "target_code": ioc_code, "answer": ans, "family": "B",
                   "join": f"{joinkind}={joinval}"})

    if problems:
        raise SystemExit("gabarito invalido (corrija antes de medir):\n  " + "\n  ".join(problems))
    return qs


if __name__ == "__main__":
    qs = build()
    from collections import Counter
    fam = Counter(q["family"] for q in qs)
    print(f"{len(qs)} perguntas validadas  (A={fam['A']}, B={fam['B']})")
    for q in qs:
        print(f"  [{q['family']}] {q['q']:<70} -> {q['target_system']}:{q['target_code']} = {q['answer']}")
