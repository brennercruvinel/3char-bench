#!/usr/bin/env python3
"""
collide.py  -  experiment 2: collision disambiguation.

"GRU", "BR", "FRA", "S" are claimed by several systems. two measures:

1. combinatorial cost of the ambiguity. how many literals collide, across how many systems, and
   the expected routing error rate for a bare query without context = mean of (1 - 1/k). this is
   the cost of NOT namespacing, as a number.

2. routing measured by retrieval. for each pair (literal, intended system), three conditions:
     bare    doc = bare code;              query = bare code (no context)
     intent  doc = verbalized record;      query = intent phrase + code
     ns      doc = 'sysid:code';           query = 'sysid:code'
   accuracy = top-1 falls in the intended system. quantifies how much namespace/intent recover
   the precision the raw string loses.

case-insensitive comparison (whoever types 'br' or 'BR' wants the same thing).
writes out/collide.json.
"""
import pathlib
import sys

_RIG = pathlib.Path(__file__).resolve().parents[1]
for _p in ("lib", "exp"):
    _d = str(_RIG / _p)
    if _d not in sys.path:
        sys.path.insert(0, _d)

import collections

import catalog
import embed
import faiss
import jio
import numpy as np

# intent phrase per system (the user knows which domain they want)
INTENT = {
    "trp/iata": "codigo IATA de aeroporto",
    "trp/icao": "codigo ICAO de aeroporto",
    "lng/i31": "codigo de pais ISO 3166",
    "lng/ioc": "codigo de pais IOC",
    "fin/i42": "codigo de moeda ISO 4217",
    "cls/elm": "simbolo de elemento quimico",
    "cmp/sts": "codigo de status HTTP",
    "bio/aa1": "aminoacido codigo de uma letra",
    "bio/aa3": "aminoacido codigo de tres letras",
    "com/mor": "caractere em codigo Morse",
}


def collisions(corpus):
    claim = collections.defaultdict(set)        # LITERAL(upper) -> {system}
    rec_by = collections.defaultdict(dict)      # system -> {LITERAL: Record}
    for node, rs in corpus.items():
        for r in rs:
            lit = r.code.upper()
            claim[lit].add(node)
            rec_by[node].setdefault(lit, r)
    coll = {lit: sorted(sys) for lit, sys in claim.items() if len(sys) >= 2}
    return coll, rec_by


def dataset_collisions():
    """the same census over ALL systems published on the hub, code column only.
    e uma propriedade do dataset, nao do corpus curado: quem carrega o config 'all'
    finds exactly these literals claimed by more than one system."""
    claim = collections.defaultdict(set)
    for sid in catalog.published():
        for c in set(catalog.codes(sid)):
            claim[c.upper()].add(sid)
    coll = {lit: sorted(s) for lit, s in claim.items() if len(s) >= 2}
    ks = [len(s) for s in coll.values()]
    worst = sorted(coll.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:20]
    per_sys = collections.Counter(s for syss in coll.values() for s in syss)
    return {"n_systems": len(claim and set().union(*claim.values())),
            "n_literals_total": len(claim), "n_literals_collided": len(coll),
            "k_hist": dict(sorted(collections.Counter(ks).items())),
            "worst": [{"literal": l, "k": len(s), "systems": s} for l, s in worst],
            "most_collided_systems": dict(per_sys.most_common(10))}


def run():
    corpus = catalog.load_all()
    coll, rec_by = collisions(corpus)

    # ---- 1. combinatorial cost ----
    ks = [len(s) for s in coll.values()]
    hist = dict(sorted(collections.Counter(ks).items()))
    amb_cost = float(np.mean([1 - 1 / k for k in ks])) if ks else 0.0
    worst = sorted(coll.items(), key=lambda kv: -len(kv[1]))[:15]

    # ---- 2. measured routing ----
    # (literal, system) pairs limited to the most ambiguous literals for a dense test
    sample_lits = [lit for lit, _ in sorted(coll.items(), key=lambda kv: -len(kv[1]))[:60]]
    pairs = [(lit, s) for lit in sample_lits for s in coll[lit]]
    # random-tie ceiling: a raw query without context cannot tell the k claimants
    # apart; the best one can expect is 1/k. if acc_bare ~ this ceiling, the bare
    # condition measures the combinatorial floor, not semantics (honest caveat).
    tie_ceiling = float(np.mean([1.0 / len(coll[lit]) for lit, _ in pairs]))

    # single index with every record, in three renderings
    all_recs = [(node, r) for node, rs in corpus.items() for r in rs]
    bare_docs = [r.code for _, r in all_recs]
    ns_docs = [f"{node.split('/')[1]}:{r.code}" for node, r in all_recs]
    intent_docs = [r.verb_chunk(False) for node, r in all_recs]
    sys_of = [node for node, _ in all_recs]

    report = {"_meta": {"n_literais_colidentes": len(coll), "k_hist": hist,
                        "ambiguity_cost_bare": round(amb_cost, 4),
                        "bare_random_tie_ceiling": round(tie_ceiling, 4),
                        "n_pairs": len(pairs),
                        "worst": [{"literal": l, "k": len(s), "systems": s} for l, s in worst]},
              "routing": {},
              "dataset": dataset_collisions()}

    for emb_name in embed.NAMES:
        E = embed.get(emb_name)
        dv_bare = E.docs(bare_docs).astype(np.float32)
        dv_ns = E.docs(ns_docs).astype(np.float32)
        dv_int = E.docs(intent_docs).astype(np.float32)

        def acc(dvecs, qtexts, want_sys):
            idx = faiss.IndexFlatIP(dvecs.shape[1])
            idx.add(dvecs)
            qv = E.queries(qtexts).astype(np.float32)
            _, order = idx.search(qv, 1)
            hit = sum(1 for i, top in enumerate(order[:, 0]) if sys_of[int(top)] == want_sys[i])
            return round(hit / len(qtexts), 4)

        want = [s for _, s in pairs]
        a_bare = acc(dv_bare, [lit for lit, _ in pairs], want)
        a_int = acc(dv_int, [f"{INTENT[s]} {lit}" for lit, s in pairs], want)
        a_ns = acc(dv_ns, [f"{s.split('/')[1]}:{lit}" for lit, s in pairs], want)
        report["routing"][emb_name] = {
            "n_pairs": len(pairs), "acc_bare_context_free": a_bare,
            "acc_intent_verbalized": a_int, "acc_namespaced": a_ns}

    jio.dump(report, "collide.json")
    return report


if __name__ == "__main__":
    r = run()
    m = r["_meta"]
    print(f"colisoes: {m['n_literais_colidentes']} literais, k_hist={m['k_hist']}")
    print(f"custo combinatorio de ambiguidade (query crua): {m['ambiguity_cost_bare']:.1%} de roteamento errado esperado")
    print("piores:", ", ".join(f"{w['literal']}(k={w['k']})" for w in m['worst'][:8]))
    d = r["dataset"]
    print(f"dataset inteiro ({d['n_systems']} sistemas): {d['n_literals_collided']} de {d['n_literals_total']} literais "
          f"em >=2 sistemas, k_hist={d['k_hist']}; piores: " + ", ".join(f"{w['literal']}(k={w['k']})" for w in d['worst'][:6]))
    for emb, rt in r["routing"].items():
        print(f"[{emb}] roteamento top-1 -> bare={rt['acc_bare_context_free']:.1%}  "
              f"intent={rt['acc_intent_verbalized']:.1%}  ns={rt['acc_namespaced']:.1%}  (n={rt['n_pairs']})")
