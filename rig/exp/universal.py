#!/usr/bin/env python3
"""
universal.py  -  experiment 3: convergence and universality.

the question: is there a UNIVERSAL code/term that crosses every domain? and if not, what is the
real join axis?

three measures:
1. term overlap between the primary code sets of each pair of systems (containment and Jaccard).
   shows that across domains code overlap is ~0: 'a table of 3 chars' is a morphological cut, not
   a semantic one.
2. the most 'universal' literal = the one claimed by the most systems (from the collision test).
   none crosses every domain.
3. the join axis by COUNTRY (ISO 3166): for each system that carries a country key, its coverage
   against the ISO reference. systems with high coverage form ONE connected component; the rest
   are islands. -> universality is not in the code, it is in the join key, and only for a
   sub-cluster.

writes out/universal.json.
"""
import pathlib
import sys

_RIG = pathlib.Path(__file__).resolve().parents[1]
for _p in ("lib", "exp"):
    _d = str(_RIG / _p)
    if _d not in sys.path:
        sys.path.insert(0, _d)

import collections
import itertools

import catalog
import jio


# how to extract the country key (alpha-2 uppercase) from each system that has one.
# the other systems have no country referent: they stay out of the join graph.
def _norm(name):
    return name.upper().replace("(THE)", "").strip()


def _country_keys(corpus, name_to_a2):
    out = {}
    for node, rs in corpus.items():
        vals = set()
        for r in rs:
            a2s = []
            if node == "lng/i31":
                a2s = r.aliases[:1]                       # alpha-2 lives in aliases
            elif node == "lng/ioc":
                a2s = [r.f("iso2")]
            elif node in ("trp/iata", "trp/icao"):
                a2s = [r.f("country")]
            elif node == "fin/i42":                       # currency lists country names
                a2s = [name_to_a2.get(_norm(n)) for n in r.f("countries").split(", ")]
            vals |= {a.upper() for a in a2s if a and len(a) == 2}
        if vals:
            out[node] = vals
    return out


def run():
    corpus = catalog.load_all()

    # ISO 3166 reference
    i31 = corpus["lng/i31"]
    iso_a2 = {r.aliases[0].upper() for r in i31 if r.aliases}
    name_to_a2 = {_norm(r.label): r.aliases[0].upper() for r in i31 if r.aliases}

    # ---- 1. term overlap (primary code) between pairs ----
    sets = {n: {r.code.upper() for r in rs} for n, rs in corpus.items()}
    overlaps = []
    for a, b in itertools.combinations(sorted(sets), 2):
        A, B = sets[a], sets[b]
        inter = len(A & B)
        if inter == 0:
            continue
        overlaps.append({
            "a": a, "b": b, "shared": inter,
            "containment": round(inter / min(len(A), len(B)), 4),
            "jaccard": round(inter / len(A | B), 4)})
    overlaps.sort(key=lambda d: -d["containment"])

    # ---- 2. most universal literal ----
    claim = collections.defaultdict(set)
    for n, S in sets.items():
        for lit in S:
            claim[lit].add(n)
    top_lit = sorted(((lit, sorted(s)) for lit, s in claim.items() if len(s) >= 2),
                     key=lambda kv: -len(kv[1]))[:10]
    domains = {n.split("/")[0] for n in sets}
    # no literal crosses every domain:
    max_domain_span = max((len({s.split('/')[0] for s in syss}) for _, syss in top_lit), default=0)

    # ---- 3. join axis by country ----
    ckeys = _country_keys(corpus, name_to_a2)
    coverage = {n: round(len(v & iso_a2) / len(v), 4) for n, v in ckeys.items()}
    # graph: edge between systems that share the country key with high coverage
    joinable = [n for n, c in coverage.items() if c >= 0.8]
    isolated = [n for n in sets if n not in ckeys]

    report = {
        "term_overlap_top": overlaps[:15],
        "cross_domain_overlap_count": sum(
            1 for o in overlaps if o["a"].split("/")[0] != o["b"].split("/")[0]),
        "most_universal_literal": [{"literal": l, "k": len(s), "systems": s} for l, s in top_lit],
        "n_domains": len(domains), "max_domain_span_of_any_literal": max_domain_span,
        "country_axis": {
            "systems_with_country_key": sorted(ckeys),
            "coverage_vs_iso3166": coverage,
            "joinable_cluster": sorted(joinable),
            "isolated_systems": sorted(isolated)},
        "verdict": {
            "universal_code_exists": False,
            "universal_join_axis": "pais (ISO 3166)",
            "joinable_cluster_size": len(joinable),
            "isolated_count": len(isolated)},
    }
    jio.dump(report, "universal.json")
    return report


if __name__ == "__main__":
    r = run()
    print("=== sobreposicao de termos (top, por containment) ===")
    for o in r["term_overlap_top"][:8]:
        print(f"  {o['a']:9} ∩ {o['b']:9}  shared={o['shared']:>4}  containment={o['containment']:.2f}  jaccard={o['jaccard']:.3f}")
    print(f"\npares de sobreposicao CROSS-dominio: {r['cross_domain_overlap_count']}")
    print(f"dominios: {r['n_domains']}  |  maior span de dominios de um literal: {r['max_domain_span_of_any_literal']}")
    print("\nliteral mais universal:", ", ".join(f"{x['literal']}(k={x['k']})" for x in r["most_universal_literal"][:6]))
    ca = r["country_axis"]
    print("\neixo de juncao por pais (cobertura vs ISO 3166):")
    for n, c in sorted(ca["coverage_vs_iso3166"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:9} {c:.1%}")
    print(f"cluster joinavel: {ca['joinable_cluster']}")
    print(f"sistemas ilhados (sem referente de pais): {ca['isolated_systems']}")
    print(f"\nveredito: codigo universal existe? {r['verdict']['universal_code_exists']}  |  "
          f"eixo universal: {r['verdict']['universal_join_axis']}  |  "
          f"cluster={r['verdict']['joinable_cluster_size']} ilhados={r['verdict']['isolated_count']}")
