#!/usr/bin/env python3
"""
corpus.py  -  primitivas de corpus e recuperacao compartilhadas.

extraidas de grid para que grid, store e decomp reusem sem depender uns dos
outros (cada experimento vira folha; o que e comum mora aqui). responsabilidade
unica: montar o corpus amostrado, renderizar chunks nas tres granularidades, e
rodar a busca faiss.

  build_corpus(questions)         corpus amostrado, alvos do gabarito sempre dentro
  render_records/blocks/files     chunks por granularidade (form x key aplicados)
  search(doc_vecs, q_vecs, k)     faiss IndexFlatIP, retorna a ordem top-k
  rank_of / metrics               posicao do alvo e recall@1/@5/mrr
"""
import random

import catalog
import faiss

CAP = 600          # teto por sistema (alvos do gabarito sempre incluidos)
BLOCK = 20         # registros por bloco na granularidade 'block'
SEED = 20260623


def build_corpus(questions):
    recs = catalog.load_all()
    targets = {(q["target_system"], q["target_code"]) for q in questions}
    rng = random.Random(SEED)
    out = {}
    for node, rs in recs.items():
        gt = [r for r in rs if (node, r.code) in targets]
        rest = [r for r in rs if (node, r.code) not in targets]
        if len(rest) > CAP - len(gt):
            rest = rng.sample(rest, max(0, CAP - len(gt)))
        sel = gt + rest
        sel.sort(key=lambda r: r.code)          # ordem deterministica p/ blocos
        out[node] = sel
    return out


def render_records(corpus, form, key):
    """-> (texts, ids) por registro. id = (system, code)."""
    texts, ids = [], []
    ns = (key == "ns")
    for node, rs in corpus.items():
        for r in rs:
            t = r.verb_chunk(ns) if form == "verb" else r.raw_chunk(ns)
            texts.append(t)
            ids.append((node, r.code))
    return texts, ids


def render_blocks(corpus, form, key):
    """-> (texts, owners) onde owners[i] = set de ids no bloco i."""
    texts, owners = [], []
    ns = (key == "ns")
    for node, rs in corpus.items():
        for b in range(0, len(rs), BLOCK):
            chunk = rs[b:b + BLOCK]
            lines = [(r.verb_chunk(ns) if form == "verb" else r.raw_chunk(ns)) for r in chunk]
            texts.append("\n".join(lines))
            owners.append({(node, r.code) for r in chunk})
    return texts, owners


def render_files(corpus, form, key):
    """-> (texts, owners) um chunk por sistema (arquivo inteiro)."""
    texts, owners = [], []
    ns = (key == "ns")
    for node, rs in corpus.items():
        lines = [(r.verb_chunk(ns) if form == "verb" else r.raw_chunk(ns)) for r in rs]
        texts.append("\n".join(lines))
        owners.append({(node, r.code) for r in rs})
    return texts, owners


def search(doc_vecs, q_vecs, k):
    idx = faiss.IndexFlatIP(doc_vecs.shape[1])
    idx.add(doc_vecs)
    _, order = idx.search(q_vecs, min(k, doc_vecs.shape[0]))
    return order


def rank_of(order_row, hit_pred):
    for rank, di in enumerate(order_row, 1):
        if hit_pred(int(di)):
            return rank
    return None


def metrics(ranks):
    arr = [r for r in ranks if r is not None]
    n = len(ranks)
    r1 = sum(1 for r in arr if r == 1) / n
    r5 = sum(1 for r in arr if r <= 5) / n
    mrr = sum(1.0 / r for r in arr) / n
    return {"recall@1": round(r1, 4), "recall@5": round(r5, 4), "mrr": round(mrr, 4)}
