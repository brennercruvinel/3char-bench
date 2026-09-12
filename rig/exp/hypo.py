#!/usr/bin/env python3
"""
hypo.py  -  motor de hipoteses reutilizavel + biblioteca de pattern matchers.

uma hipotese e DADO, nao codigo: vive em rig/hypotheses.json como
  {id, statement, matcher, params, predicate}
o motor roda o matcher sobre o corpus, extrai evidencia numerica e aplica o
predicado para emitir um veredito (supported|refuted). adicionar hipotese =
adicionar uma entrada e rerodar; nenhum numero embutido, nenhum erro engolido.

matchers (todos os tipos de pattern matching pedidos):
  length     distribuicao e moda do comprimento dos codigos
  charset    composicao: alpha / digito / alfanumerico / simbolo
  case       caixa: upper / lower / title / mixed
  vowel      razao de vogais (principio abjad)
  regex      taxa de match de um regex
  collision  fracao dos codigos reivindicada por outros sistemas
  join       cobertura de uma chave contra um sistema de referencia
"""
import pathlib
import sys

_RIG = pathlib.Path(__file__).resolve().parents[1]
for _p in ("lib", "exp"):
    _d = str(_RIG / _p)
    if _d not in sys.path:
        sys.path.insert(0, _d)

import collections
import re
import statistics as st

import catalog
import jio

# ---------------- biblioteca de matchers ----------------
# cada matcher: (corpus, params) -> dict de evidencia (numeros prontos p/ predicado)

def _recs(corpus, params):
    sysids = params.get("systems") or list(corpus)
    return [r for s in sysids for r in corpus[s]]


def m_length(corpus, params):
    recs = _recs(corpus, params)
    lens = [len(r.code) for r in recs]
    hist = collections.Counter(lens)
    mode = hist.most_common(1)[0][0]
    share = {k: round(v / len(lens), 4) for k, v in hist.items()}
    return {"mode_len": mode, "mean_len": round(st.mean(lens), 2),
            "share_2_3": round(share.get(2, 0) + share.get(3, 0), 4),
            "hist": dict(sorted(hist.items()))}


def m_charset(corpus, params):
    recs = _recs(corpus, params)
    n = len(recs)
    alpha = sum(c.code.isalpha() for c in recs) / n
    digit = sum(c.code.isdigit() for c in recs) / n
    alnum = sum(c.code.isalnum() for c in recs) / n
    sym = sum(not c.code.isalnum() for c in recs) / n
    return {"frac_alpha": round(alpha, 4), "frac_digit": round(digit, 4),
            "frac_alnum": round(alnum, 4), "frac_symbol": round(sym, 4)}


def m_case(corpus, params):
    recs = _recs(corpus, params)
    n = len(recs)
    up = sum(c.code.isupper() for c in recs) / n
    lo = sum(c.code.islower() for c in recs) / n
    ti = sum(c.code.istitle() for c in recs) / n
    return {"frac_upper": round(up, 4), "frac_lower": round(lo, 4), "frac_title": round(ti, 4)}


def m_vowel(corpus, params):
    recs = _recs(corpus, params)
    letters = "".join(c for r in recs for c in r.code.lower() if c.isalpha())
    if not letters:
        return {"vowel_ratio": None, "n_letters": 0}
    vr = sum(c in "aeiou" for c in letters) / len(letters)
    return {"vowel_ratio": round(vr, 4), "n_letters": len(letters), "english_baseline": 0.38}


def m_regex(corpus, params):
    recs = _recs(corpus, params)
    pat = re.compile(params["pattern"])
    hit = sum(1 for r in recs if pat.fullmatch(r.code)) / len(recs)
    return {"pattern": params["pattern"], "match_rate": round(hit, 4), "n": len(recs)}


def m_collision(corpus, params):
    sysids = params.get("systems") or list(corpus)
    own = set()
    for s in sysids:
        own |= {r.code.upper() for r in corpus[s]}
    others = collections.defaultdict(set)
    for s, rs in corpus.items():
        if s in sysids:
            continue
        for r in rs:
            others[r.code.upper()].add(s)
    claimed = {c for c in own if c in others}
    return {"n_codes": len(own), "n_collided": len(claimed),
            "frac_collided": round(len(claimed) / max(len(own), 1), 4),
            "examples": sorted(claimed)[:8]}


def m_join(corpus, params):
    ref_field = params["ref_field"]          # campo no sistema de referencia
    ref_sys = params["ref_system"]
    key_field = params["key_field"]          # campo no sistema testado
    test_sys = params["test_system"]
    def pick(r, field):                        # 'code' | 'alias' | chave de fields
        if field == "code":
            return [r.code]
        if field == "alias":
            return list(r.aliases)
        return [r.fields.get(field, "")]
    ref = {v.upper() for r in corpus[ref_sys] for v in pick(r, ref_field) if v}
    vals = [v.upper() for r in corpus[test_sys] for v in pick(r, key_field) if v]
    cov = sum(1 for v in vals if v in ref) / max(len(vals), 1)
    return {"coverage": round(cov, 4), "n_keys": len(vals), "ref_size": len(ref)}


MATCHERS = {"length": m_length, "charset": m_charset, "case": m_case, "vowel": m_vowel,
            "regex": m_regex, "collision": m_collision, "join": m_join}

OPS = {"==": lambda a, b: a == b, "!=": lambda a, b: a != b,
       ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
       "<": lambda a, b: a < b, "<=": lambda a, b: a <= b}


def evaluate(hyp, corpus):
    matcher = MATCHERS[hyp["matcher"]]
    ev = matcher(corpus, hyp.get("params", {}))
    p = hyp["predicate"]                      # {metric, op, value}
    val = ev.get(p["metric"])
    verdict = "indeterminado" if val is None else (
        "supported" if OPS[p["op"]](val, p["value"]) else "refuted")
    return {"id": hyp["id"], "statement": hyp["statement"], "matcher": hyp["matcher"],
            "predicate": p, "evidence": ev, "verdict": verdict}


# hipoteses-semente: data-driven, substituem o estofo retorico por testes que
# sustentam uma decisao. novas hipoteses (humanas ou geradas) sao anexadas ao
# mesmo arquivo e re-testadas sem tocar no codigo.
SEED = [
    {"id": "H1", "statement": "no cluster geo-economico o comprimento modal de codigo e 2 ou 3",
     "matcher": "length", "params": {"systems": ["lng/i31", "lng/ioc", "fin/i42"]},
     "predicate": {"metric": "mode_len", "op": "<=", "value": 3}},
    {"id": "H2", "statement": "simbolos de elemento descartam vogais (abjad): razao < baseline ingles 0.38",
     "matcher": "vowel", "params": {"systems": ["cls/elm"]},
     "predicate": {"metric": "vowel_ratio", "op": "<", "value": 0.38}},
    {"id": "H3", "statement": "no espaco de 3 letras, codigos IATA colidem com outros sistemas em massa",
     "matcher": "collision", "params": {"systems": ["trp/iata"]},
     "predicate": {"metric": "n_collided", "op": ">=", "value": 100}},
    {"id": "H4", "statement": "todo pais de aeroporto (extra.country) resolve em ISO 3166 alpha-2",
     "matcher": "join", "params": {"ref_system": "lng/i31", "ref_field": "alias",
                                   "test_system": "trp/iata", "key_field": "country"},
     "predicate": {"metric": "coverage", "op": ">=", "value": 0.99}},
    {"id": "H5", "statement": "codigos de status HTTP sao puramente numericos de 3 digitos",
     "matcher": "regex", "params": {"systems": ["cmp/sts"], "pattern": r"\d{3}"},
     "predicate": {"metric": "match_rate", "op": "==", "value": 1.0}},
]


def load_hypotheses():
    try:
        return jio.load("hypotheses.json")["hypotheses"]
    except Exception:
        jio.dump({"hypotheses": SEED}, "hypotheses.json")
        return SEED


def run():
    corpus = catalog.load_all()
    hyps = load_hypotheses()
    results = [evaluate(h, corpus) for h in hyps]
    jio.dump({"results": results}, "hypo_results.json")
    return results


if __name__ == "__main__":
    for r in run():
        v = r["verdict"].upper()
        print(f"[{r['id']}] {v:13} {r['statement']}")
        print(f"      matcher={r['matcher']}  {r['predicate']}  ->  {r['evidence']}")
