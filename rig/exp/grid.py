#!/usr/bin/env python3
"""
grid.py  -  experimento 1: a grade controlada.

um recuperador fixo (faiss IndexFlatIP sobre vetores normalizados = cosseno),
variando UMA dimensao por vez:

  granularidade  record | block(B=20) | file
  forma          raw (linha CSV) | verb (prosa)
  chave          bare ('GRU') | ns ('iata:GRU')
  embedder       minilm | e5

corpus, render e busca vem de lib/corpus. mede, por celula: recall@1, recall@5,
mrr e tokens_por_resposta (tiktoken o200k no chunk que carrega a resposta).
escreve out/grid.json.
"""
import pathlib
import sys

_RIG = pathlib.Path(__file__).resolve().parents[1]
for _p in ("lib", "exp"):
    _d = str(_RIG / _p)
    if _d not in sys.path:
        sys.path.insert(0, _d)

import embed
import jio
import numpy as np
import qa
import tok
from corpus import (
    BLOCK,
    CAP,
    SEED,
    build_corpus,
    metrics,
    rank_of,
    render_blocks,
    render_files,
    render_records,
    search,
)


def run():
    questions = qa.build()
    corpus = build_corpus(questions)
    sizes = {n: len(rs) for n, rs in corpus.items()}
    total_docs = sum(sizes.values())
    targets = [(q["target_system"], q["target_code"]) for q in questions]
    qtexts = [q["q"] for q in questions]

    results = {"_meta": {"n_questions": len(questions), "cap_per_system": CAP,
                         "block_size": BLOCK, "seed": SEED, "tok_mode": tok.MODE,
                         "corpus_docs": total_docs, "system_sizes": sizes,
                         "family": {f: sum(1 for q in questions if q["family"] == f) for f in "AB"}},
               "cells": []}

    for emb_name in embed.NAMES:
        E = embed.get(emb_name)
        qv = E.queries(qtexts).astype(np.float32)
        print(f"[{emb_name}] queries embutidas", file=sys.stderr)

        for form in ("raw", "verb"):
            for key in ("bare", "ns"):
                # ---- record ----
                texts, ids = render_records(corpus, form, key)
                pos = {}
                for i, idd in enumerate(ids):
                    pos.setdefault(idd, i)
                dv = E.docs(texts).astype(np.float32)
                order = search(dv, qv, 5)
                ranks, gt_toks = [], []
                for j, tgt in enumerate(targets):
                    gi = pos.get(tgt)
                    ranks.append(rank_of(order[j], lambda di: di == gi) if gi is not None else None)
                    gt_toks.append(tok.count(texts[gi]) if gi is not None else 0)
                results["cells"].append({
                    "embedder": emb_name, "granularity": "record", "form": form, "key": key,
                    **metrics(ranks), "tokens_per_answer": round(float(np.mean(gt_toks)), 1),
                    "n_docs": len(texts)})

                # ---- block ----
                btexts, bowners = render_blocks(corpus, form, key)
                bdv = E.docs(btexts).astype(np.float32)
                border = search(bdv, qv, 5)
                branks, bgt = [], []
                owner_of = {}
                for bi, own in enumerate(bowners):
                    for idd in own:
                        owner_of[idd] = bi
                for j, tgt in enumerate(targets):
                    bi = owner_of.get(tgt)
                    branks.append(rank_of(border[j], lambda di: di == bi) if bi is not None else None)
                    bgt.append(tok.count(btexts[bi]) if bi is not None else 0)
                results["cells"].append({
                    "embedder": emb_name, "granularity": "block", "form": form, "key": key,
                    **metrics(branks), "tokens_per_answer": round(float(np.mean(bgt)), 1),
                    "n_docs": len(btexts)})

                # ---- file ----
                ftexts, fowners = render_files(corpus, form, key)
                fdv = E.docs(ftexts).astype(np.float32)
                forder = search(fdv, qv, 5)
                franks, fgt = [], []
                fowner_of = {}
                for fi, own in enumerate(fowners):
                    for idd in own:
                        fowner_of[idd] = fi
                for j, tgt in enumerate(targets):
                    fi = fowner_of.get(tgt)
                    franks.append(rank_of(forder[j], lambda di: di == fi) if fi is not None else None)
                    fgt.append(tok.count(ftexts[fi]) if fi is not None else 0)
                results["cells"].append({
                    "embedder": emb_name, "granularity": "file", "form": form, "key": key,
                    **metrics(franks), "tokens_per_answer": round(float(np.mean(fgt)), 1),
                    "n_docs": len(ftexts)})
                print(f"[{emb_name}] form={form} key={key} ok", file=sys.stderr)

    jio.dump(results, "grid.json")
    return results


if __name__ == "__main__":
    r = run()
    print(f"grade: {len(r['cells'])} celulas, {r['_meta']['corpus_docs']} docs, "
          f"{r['_meta']['n_questions']} perguntas -> out/grid.json")
