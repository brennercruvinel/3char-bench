#!/usr/bin/env python3
"""
catalog.py  -  carrega o dataset publicado em REGISTROS uniformes.

le o parquet do 3char no schema canonico (system_id, code, label, description,
parent_code, status, aliases, extra). a coluna-codigo nao e mais curada aqui: o
pipeline do dataset ja decidiu o que e codigo em cada sistema e validou contra o
regex do sistema. o que continua curado e a PROSA: um template por sistema, porque
a qualidade da verbalizacao e justamente o que a grade mede.

cada registro:
  system   'trp/iata'
  code     'GRU'
  label    'Sao Paulo/Guarulhos ... International Airport'
  aliases  ['...'] outros literais do mesmo referente
  fields   dict plano com label, description, aliases e as chaves de extra

renderizacao de chunk (as duas formas testadas na grade):
  raw_chunk()   k=v de todos os campos do registro
  verb_chunk()  prosa via template do sistema
e a variante de chave:
  bare    codigo nu             'GRU'
  ns      codigo com namespace  'iata:GRU'
"""
import json

import conf
import pyarrow.parquet as pq


def _j(v):
    return ", ".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)


# ----- prosa curada por sistema -------------------------------------------------
# verb: template. filt: predicado opcional sobre o registro (ex: tirar linhas
# 'unassigned' do registro IANA, que sao 436 de 500 e nao descrevem nada).

def _country_i31(r):
    return (f"O país {r.label} tem código ISO 3166 alpha-3 {r.code}, alpha-2 {r.f('aliases')}, "
            f"código numérico {r.f('numeric')}.")


def _country_ioc(r):
    return (f"O país {r.label} tem código IOC {r.code}, ISO alpha-3 {r.f('iso3')}, "
            f"alpha-2 {r.f('iso2')}, TLD {r.f('tld')}, capital {r.f('capital')}.")


def _airport(kind):
    return lambda r: (f"O aeroporto {r.label}, código {kind} {r.code}, no país {r.f('country')}"
                      f"{', município ' + r.f('municipality') if r.f('municipality') else ''}.")


SPECS = {
    "lng/i31": {"verb": _country_i31},
    "lng/ioc": {"verb": _country_ioc},
    "trp/iata": {"verb": _airport("IATA"), "filt": lambda r: r.f("type") != "closed"},
    "trp/icao": {"verb": _airport("ICAO"), "filt": lambda r: r.f("type") != "closed"},
    "fin/i42": {"verb": lambda r: (f"A moeda {r.label}, código ISO 4217 {r.code} "
                                   f"(numérico {r.f('numeric')}), usada em {r.f('countries')}.")},
    "cls/elm": {"verb": lambda r: (f"O elemento químico {r.label}, símbolo {r.code}, "
                                   f"número atômico {r.f('AtomicNumber')}.")},
    "cmp/sts": {"verb": lambda r: f"O código de status HTTP {r.code} significa {r.label}.",
                "filt": lambda r: r.status != "unassigned"},
    # tabelas de codigo curto que NAO cruzam por chave (contraste de universalidade)
    "bio/aa1": {"verb": lambda r: (f"O aminoácido {r.label}, código de 1 letra {r.code}, "
                                   f"de 3 letras {r.f('three_letter')}.")},
    "bio/aa3": {"verb": lambda r: (f"O aminoácido {r.label}, código de 3 letras {r.code}, "
                                   f"de 1 letra {r.f('one_letter')}.")},
    "com/mor": {"verb": lambda r: f"O caractere '{r.code}' em código Morse é '{r.label}'."},
}
NODES = list(SPECS)


class Record:
    __slots__ = ("system", "code", "label", "status", "aliases", "fields")

    def __init__(self, system, code, label, status, aliases, fields):
        self.system, self.code, self.label = system, code, label
        self.status, self.aliases, self.fields = status, aliases, fields

    def f(self, k):
        return self.fields.get(k, "")

    def key(self, ns=False):
        return f"{self.system.split('/')[1]}:{self.code}" if ns else self.code

    def raw_chunk(self, ns=False):
        head = f"{self.key(ns)} | " if ns else ""
        body = ",".join(f"{k}={v}" for k, v in self.fields.items())
        return head + f"code={self.code}," + body

    def verb_chunk(self, ns=False):
        text = SPECS[self.system]["verb"](self)
        return (f"{self.key(ns)} - " + text) if ns else text


def _records(system_id):
    t = pq.read_table(conf.parquet(system_id)).to_pylist()
    for row in t:
        aliases = [a for a in (row.get("aliases") or []) if a]
        fields = {"label": row["label"] or ""}
        if row.get("description"):
            fields["description"] = row["description"]
        if aliases:
            fields["aliases"] = _j(aliases)
        for k, v in sorted(json.loads(row["extra"] or "{}").items()):
            if v not in (None, "", []):
                fields[k] = _j(v)
        yield Record(system_id, row["code"], row["label"] or "", row.get("status"), aliases, fields)


def load(system_id):
    """registros de um sistema curado, na ordem do parquet, com o filtro do spec."""
    filt = SPECS[system_id].get("filt")
    return [r for r in _records(system_id) if not filt or filt(r)]


def load_all(nodes=None):
    return {n: load(n) for n in (nodes or NODES)}


def published():
    """system_id de todo sistema publicado no hub, do systems.parquet."""
    t = pq.read_table(f"{conf.hub()}/systems.parquet", columns=["system_id", "name"]).to_pylist()
    return {r["system_id"]: r["name"] for r in t}


def codes(system_id):
    """so a coluna code de um sistema, para medidas sobre o dataset inteiro."""
    return pq.read_table(conf.parquet(system_id), columns=["code"]).column("code").to_pylist()
