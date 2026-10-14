#!/usr/bin/env python3
"""
decomp.py  -  decomposes the effect of 'verbalizing' (answers the adversarial critique C1).

the 'raw' form dumps ALL kept columns; 'verb' is prose over a curated subset. so raw->verb mixes
two changes: (a) replacing k=v with prose and (b) dropping noise fields. to avoid selling one as
the other, a third intermediate form is measured:

  raw_all      k=v of every kept column                  (the grid's 'raw' form)
  raw_curated  k=v of only the columns the prose cites   (isolates the field drop)
  verb         prose over the same curated columns       (isolates the prose)

effect of dropping fields  = raw_all  -> raw_curated
effect of prose            = raw_curated -> verb
fixed at e5/record/bare (without the namespace confounder). writes out/decomp.json.
"""
import pathlib
import sys

_RIG = pathlib.Path(__file__).resolve().parents[1]
for _p in ("lib", "exp"):
    _d = str(_RIG / _p)
    if _d not in sys.path:
        sys.path.insert(0, _d)

import embed
import faiss
import jio
import numpy as np
import qa
import tok
from corpus import build_corpus

# columns each prose template actually cites (subset of 'keep').
# if a system has no noise column, raw_curated == raw_all (null effect, honest).
VF = {
    "lng/i31": ["aliases", "numeric"],                   # dropa region, sub_region
    "lng/ioc": ["aliases", "iso2", "tld", "capital"],    # dropa continent
    "trp/iata": ["country", "municipality"],             # dropa type, scheduled
    "trp/icao": ["country", "municipality"],             # dropa type, scheduled
    "fin/i42": ["numeric", "countries"],                 # dropa minor_units, country_count, published
    "cls/elm": ["AtomicNumber"],                         # dropa massa, grupo, estado, ano
    "cmp/sts": [],                                       # dropa reference
    "bio/aa1": ["three_letter"],                         # dropa formula, systematic_name, description
    "bio/aa3": ["one_letter"],
    "com/mor": [],                                       # dropa description, dots, dashes, length, section
}


def render(corpus, mode):
    texts, ids = [], []
    for node, rs in corpus.items():
        allow = {"label"} | set(VF[node])
        for r in rs:
            if mode == "raw_all":
                t = r.raw_chunk(False)
            elif mode == "raw_curated":
                t = f"code={r.code}," + ",".join(f"{k}={v}" for k, v in r.fields.items() if k in allow)
            else:  # verb
                t = r.verb_chunk(False)
            texts.append(t)
            ids.append((node, r.code))
    return texts, ids


def measure(E, texts, ids, qv, targets):
    pos = {}
    for i, idd in enumerate(ids):
        pos.setdefault(idd, i)
    dv = E.docs(texts).astype(np.float32)
    idx = faiss.IndexFlatIP(dv.shape[1])
    idx.add(dv)
    _, order = idx.search(qv, 5)
    r1 = 0
    toks = []
    for j, tgt in enumerate(targets):
        gi = pos.get(tgt)
        if gi is None:
            continue
        if int(order[j][0]) == gi:
            r1 += 1
        toks.append(tok.count(texts[gi]))
    return round(r1 / len(targets), 4), round(float(np.mean(toks)), 1)


def run():
    questions = qa.build()
    corpus = build_corpus(questions)
    targets = [(q["target_system"], q["target_code"]) for q in questions]
    E = embed.get("e5")
    qv = E.queries([q["q"] for q in questions]).astype(np.float32)

    out = {}
    for mode in ("raw_all", "raw_curated", "verb"):
        texts, ids = render(corpus, mode)
        r1, tk = measure(E, texts, ids, qv, targets)
        out[mode] = {"recall@1": r1, "tokens_per_answer": tk}

    a, b, c = out["raw_all"], out["raw_curated"], out["verb"]
    out["_decomposition"] = {
        "field_drop_recall_pp": round((b["recall@1"] - a["recall@1"]) * 100, 1),
        "field_drop_token_ratio": round(b["tokens_per_answer"] / a["tokens_per_answer"], 3),
        "prose_recall_pp": round((c["recall@1"] - b["recall@1"]) * 100, 1),
        "prose_token_ratio": round(c["tokens_per_answer"] / b["tokens_per_answer"], 3),
        "total_recall_pp": round((c["recall@1"] - a["recall@1"]) * 100, 1),
        "total_token_ratio": round(c["tokens_per_answer"] / a["tokens_per_answer"], 3),
        "embedder": "e5", "granularity": "record", "key": "bare", "n_questions": len(questions),
    }
    jio.dump(out, "decomp.json")
    return out


if __name__ == "__main__":
    r = run()
    for m in ("raw_all", "raw_curated", "verb"):
        print(f"  {m:12} recall@1={r[m]['recall@1']*100:5.1f}%  tok={r[m]['tokens_per_answer']:.1f}")
    d = r["_decomposition"]
    print(f"\n  descarte de campo : {d['field_drop_recall_pp']:+.1f}pp recall, {d['field_drop_token_ratio']}x token")
    print(f"  prosa             : {d['prose_recall_pp']:+.1f}pp recall, {d['prose_token_ratio']}x token")
    print(f"  total raw->verb   : {d['total_recall_pp']:+.1f}pp recall, {d['total_token_ratio']}x token")
