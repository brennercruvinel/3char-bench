#!/usr/bin/env python3
"""
catalog.py  -  loads the published dataset as uniform RECORDS.

reads the 3char parquet in the canonical schema (system_id, code, label, description,
parent_code, status, aliases, extra). the code column is no longer curated here: the dataset
pipeline already decided what counts as a code in each system and validated it against the
system's regex. what remains curated is the PROSE: one template per system, because the quality
of the verbalization is exactly what the grid measures.

each record:
  system   'trp/iata'
  code     'GRU'
  label    'Sao Paulo/Guarulhos ... International Airport'
  aliases  ['...'] other literals of the same referent
  fields   flat dict with label, description, aliases and the keys of extra

chunk rendering (the two forms tested in the grid):
  raw_chunk()   k=v of every field of the record
  verb_chunk()  prose through the system's template
and the key variant:
  bare    bare code              'GRU'
  ns      code with namespace    'iata:GRU'
"""
import json

import conf
import pyarrow.parquet as pq


def _j(v):
    return ", ".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)


# ----- curated prose per system -------------------------------------------------
# verb: template. filt: optional predicate over the record (e.g. drop the
# 'unassigned' rows of the IANA registry, 436 of 500, which describe nothing).

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
    # short-code tables that do NOT join by key (universality contrast)
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
    """records of a curated system, in parquet order, with the spec filter applied."""
    filt = SPECS[system_id].get("filt")
    return [r for r in _records(system_id) if not filt or filt(r)]


def load_all(nodes=None):
    return {n: load(n) for n in (nodes or NODES)}


def published():
    """system_id of every system published on the hub, from systems.parquet."""
    t = pq.read_table(f"{conf.hub()}/systems.parquet", columns=["system_id", "name"]).to_pylist()
    return {r["system_id"]: r["name"] for r in t}


def codes(system_id):
    """only the code column of a system, for measures over the whole dataset."""
    return pq.read_table(conf.parquet(system_id), columns=["code"]).column("code").to_pylist()
