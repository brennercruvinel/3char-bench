#!/usr/bin/env python3
"""
corpus.py  -  shared corpus and retrieval primitives.

extracted from grid so that grid, store and decomp reuse them without depending on each other
(each experiment is a leaf; what is common lives here). single responsibility: build the sampled
corpus, render chunks at the three granularities, and run the faiss search.

  build_corpus(questions)         sampled corpus, answer-key targets always included
  render_records/blocks/files     chunks per granularity (form x key applied)
  search(doc_vecs, q_vecs, k)     faiss IndexFlatIP, returns the top-k order
  rank_of / metrics               rank of the target and recall@1/@5/mrr
"""
import random

import catalog
import faiss

CAP = 600          # cap per system (answer-key targets always included)
BLOCK = 20         # records per block at the 'block' granularity
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
        sel.sort(key=lambda r: r.code)          # deterministic order for blocks
        out[node] = sel
    return out


def render_records(corpus, form, key):
    """-> (texts, ids) per record. id = (system, code)."""
    texts, ids = [], []
    ns = (key == "ns")
    for node, rs in corpus.items():
        for r in rs:
            t = r.verb_chunk(ns) if form == "verb" else r.raw_chunk(ns)
            texts.append(t)
            ids.append((node, r.code))
    return texts, ids


def render_blocks(corpus, form, key):
    """-> (texts, owners) where owners[i] = set of ids in block i."""
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
    """-> (texts, owners) one chunk per system (whole file)."""
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
