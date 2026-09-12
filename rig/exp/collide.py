#!/usr/bin/env python3
"""
collide.py  -  experimento 2: desambiguacao de colisao (o teste mais interessante).

"GRU", "BR", "FRA", "S" sao reivindicados por varios sistemas. duas medidas:

1. custo combinatorio da ambiguidade. quantos literais colidem, por quantos
   sistemas, e a taxa esperada de erro de roteamento para uma consulta crua sem
   contexto = media de (1 - 1/k). isto e o custo de NAO namespacear, em numero.

2. roteamento medido por recuperacao. para cada par (literal, sistema-pretendido),
   tres condicoes:
     bare    doc = codigo nu;          query = codigo nu (sem contexto)
     intent  doc = registro verbalizado; query = frase de intencao + codigo
     ns      doc = 'sysid:codigo';      query = 'sysid:codigo'
   acuracia = top-1 cai no sistema pretendido. quantifica quanto namespace/intencao
   recuperam a precisao que a string crua perde.

comparacao case-insensitive (quem digita 'br' ou 'BR' quer a mesma coisa).
escreve out/collide.json.
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

# frase de intencao por sistema (o usuario sabe que dominio quer)
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
    """o mesmo censo sobre TODOS os sistemas publicados no hub, so a coluna code.
    e uma propriedade do dataset, nao do corpus curado: quem carrega o config 'all'
    encontra exatamente estes literais reivindicados por mais de um sistema."""
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

    # ---- 1. custo combinatorio ----
    ks = [len(s) for s in coll.values()]
    hist = dict(sorted(collections.Counter(ks).items()))
    amb_cost = float(np.mean([1 - 1 / k for k in ks])) if ks else 0.0
    worst = sorted(coll.items(), key=lambda kv: -len(kv[1]))[:15]

    # ---- 2. roteamento medido ----
    # pares (literal, sistema) limitados aos literais mais ambiguos para um teste denso
    sample_lits = [lit for lit, _ in sorted(coll.items(), key=lambda kv: -len(kv[1]))[:60]]
    pairs = [(lit, s) for lit in sample_lits for s in coll[lit]]
    # teto de empate aleatorio: uma consulta crua sem contexto nao distingue os k
    # reivindicantes; o melhor que se pode esperar e 1/k. se acc_bare ~ este teto, a
    # condicao bare nao mede semantica, mede o piso combinatorio (caveat honesto).
    tie_ceiling = float(np.mean([1.0 / len(coll[lit]) for lit, _ in pairs]))

    # index unico com todos os registros, em tres renderizacoes
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
