#!/usr/bin/env python3
"""
embed.py  -  interchangeable embedders on onnxruntime (no torch).

two fixed retrievers, both 384d, so that the verdict does not depend on a single model:
  minilm  all-MiniLM-L6-v2        fast, english-centric
  e5      multilingual-e5-small   multilingual (questions in portuguese), uses the
                                  'query:'/'passage:' prefixes the model requires

masked mean pooling + L2 normalization (cosine == inner product). the only per-model difference
in treatment is the e5 prefix, encapsulated here.
"""
import conf
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class Embedder:
    def __init__(self, name, model_dir, q_prefix="", d_prefix="", max_len=192):
        self.name = name
        self.q_prefix, self.d_prefix = q_prefix, d_prefix
        self.tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
        self.tok.enable_truncation(max_length=max_len)
        pad = self.tok.token_to_id("<pad>")
        if pad is None:
            pad = self.tok.token_to_id("[PAD]")
        pad_tok = "<pad>" if self.tok.token_to_id("<pad>") is not None else "[PAD]"
        self.tok.enable_padding(pad_id=pad or 0, pad_token=pad_tok)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(f"{model_dir}/onnx/model.onnx",
                                         sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.inames = {i.name for i in self.sess.get_inputs()}

    def _run(self, texts):
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inames:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self.sess.run(None, feed)[0]
        m = mask[..., None].astype(np.float32)
        v = (out * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)

    def _embed(self, texts, prefix, bs=64):
        chunks = []
        for i in range(0, len(texts), bs):
            batch = [prefix + t for t in texts[i:i + bs]]
            chunks.append(self._run(batch).astype(np.float32))
        return np.vstack(chunks) if chunks else np.zeros((0, 384), np.float32)

    def docs(self, texts):
        return self._embed(texts, self.d_prefix)

    def queries(self, texts):
        return self._embed(texts, self.q_prefix)


_CACHE = {}


def get(name):
    if name in _CACHE:
        return _CACHE[name]
    if name == "minilm":
        e = Embedder("minilm", conf.minilm())
    elif name == "e5":
        e = Embedder("e5", conf.e5(), q_prefix="query: ", d_prefix="passage: ")
    else:
        raise ValueError(f"embedder desconhecido: {name!r} (use 'minilm' ou 'e5')")
    _CACHE[name] = e
    return e


NAMES = ["minilm", "e5"]
