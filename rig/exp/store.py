#!/usr/bin/env python3
"""
store.py  -  experiment 4: lossless storage + vector cost.

two layers, measured separately:

A. DATA (must be lossless - it is the source of the answer).
   compares, over the curated corpus: raw file concatenated, record-per-line (JSONL, the
   'graceful-line' format that survives truncation), and both under gzip.

B. VECTOR (may be approximate - reconstructible from the data).
   the same set of records embedded once (e5, verb+ns, the grid winner); compares the serialized
   size and the retained recall of three faiss indexes: Flat fp32 (exact), ScalarQuantizer 8-bit
   (4x smaller), PQ (m=48). shows how far the vector can be compressed without killing recall.

writes out/store.json.
"""
import gzip
import json
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
from corpus import CAP, build_corpus, render_records


def _sz(b):
    return len(b)


def data_layer(corpus):
    raw_parts, jsonl_lines = [], []
    rec_tok, n = [], 0
    for node, rs in corpus.items():
        for r in rs:
            raw_parts.append(r.raw_chunk(False))
            jsonl_lines.append(json.dumps(
                {"sys": node, "code": r.code, "label": r.label, **r.fields},
                ensure_ascii=False))
            rec_tok.append(tok.count(r.verb_chunk(True)))
            n += 1
    raw = ("\n".join(raw_parts)).encode("utf-8")
    jsonl = ("\n".join(jsonl_lines)).encode("utf-8")
    gz_raw = gzip.compress(raw, 9)
    gz_jsonl = gzip.compress(jsonl, 9)
    return {
        "n_records": n,
        "raw_bytes": _sz(raw), "jsonl_bytes": _sz(jsonl),
        "raw_gzip_bytes": _sz(gz_raw), "jsonl_gzip_bytes": _sz(gz_jsonl),
        "raw_gzip_ratio": round(_sz(raw) / _sz(gz_raw), 1),
        "jsonl_gzip_ratio": round(_sz(jsonl) / _sz(gz_jsonl), 1),
        "mean_tokens_per_record_chunk": round(float(np.mean(rec_tok)), 1),
        "bytes_per_record_jsonl_gzip": round(_sz(gz_jsonl) / n, 1),
    }


def vector_layer(corpus, questions):
    E = embed.get("e5")
    texts, ids = render_records(corpus, "verb", "ns")
    pos = {}
    for i, idd in enumerate(ids):
        pos.setdefault(idd, i)
    dv = E.docs(texts).astype(np.float32)
    qv = E.queries([q["q"] for q in questions]).astype(np.float32)
    targets = [(q["target_system"], q["target_code"]) for q in questions]
    d = dv.shape[1]

    def recall(idx):
        idx_obj = idx
        _, order = idx_obj.search(qv, 5)
        r1 = r5 = 0
        for j, tgt in enumerate(targets):
            gi = pos.get(tgt)
            if gi is None:
                continue
            row = [int(x) for x in order[j]]
            if row and row[0] == gi:
                r1 += 1
            if gi in row:
                r5 += 1
        n = len(targets)
        return round(r1 / n, 4), round(r5 / n, 4)

    out = {}

    flat = faiss.IndexFlatIP(d)
    flat.add(dv)
    r1, r5 = recall(flat)
    out["flat_fp32"] = {"bytes": _sz(faiss.serialize_index(flat)),
                        "recall@1": r1, "recall@5": r5}

    sq = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    sq.train(dv)
    sq.add(dv)
    r1, r5 = recall(sq)
    out["scalar_q8"] = {"bytes": _sz(faiss.serialize_index(sq)),
                        "recall@1": r1, "recall@5": r5}

    m = 48  # 384 / 8; 48 bytes per vector
    pq = faiss.IndexPQ(d, m, 8, faiss.METRIC_INNER_PRODUCT)
    pq.train(dv)
    pq.add(dv)
    r1, r5 = recall(pq)
    out["pq_m48"] = {"bytes": _sz(faiss.serialize_index(pq)),
                     "recall@1": r1, "recall@5": r5}

    base = out["flat_fp32"]["bytes"]
    for k in out:
        out[k]["x_smaller_than_flat"] = round(base / out[k]["bytes"], 1)
        out[k]["bytes_per_vector"] = round(out[k]["bytes"] / dv.shape[0], 1)
    out["_n_vectors"] = dv.shape[0]
    out["_dim"] = d
    return out


def run():
    questions = qa.build()
    corpus = build_corpus(questions)
    report = {"_meta": {"tok_mode": tok.MODE, "cap_per_system": CAP},
              "data_layer": data_layer(corpus),
              "vector_layer": vector_layer(corpus, questions)}
    jio.dump(report, "store.json")
    return report


if __name__ == "__main__":
    r = run()
    dl = r["data_layer"]
    print("=== camada de DADO (sem perda) ===")
    print(f"  registros: {dl['n_records']}")
    print(f"  raw   : {dl['raw_bytes']:>9} B  | gzip {dl['raw_gzip_bytes']:>8} B  ({dl['raw_gzip_ratio']}x)")
    print(f"  jsonl : {dl['jsonl_bytes']:>9} B  | gzip {dl['jsonl_gzip_bytes']:>8} B  ({dl['jsonl_gzip_ratio']}x)")
    print(f"  -> {dl['bytes_per_record_jsonl_gzip']} B/registro (jsonl+gzip), {dl['mean_tokens_per_record_chunk']} tok/chunk")
    print("\n=== camada de VETOR (e5, verb+ns) ===")
    vl = r["vector_layer"]
    print(f"  {vl['_n_vectors']} vetores x {vl['_dim']}d")
    for k in ("flat_fp32", "scalar_q8", "pq_m48"):
        v = vl[k]
        print(f"  {k:10} {v['bytes']:>9} B  ({v['bytes_per_vector']:>5} B/vec, {v['x_smaller_than_flat']}x menor)  "
              f"recall@1={v['recall@1']:.2f} recall@5={v['recall@5']:.2f}")
