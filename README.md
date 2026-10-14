# 3char-bench

an experimental report on feeding the [3char](https://huggingface.co/datasets/brennercruvinel/3char) tables to a dense retriever: which chunk granularity, chunk form, key form and vector storage recover the right record, measured on a fixed sample with a fixed seed. this repository holds the rig and its output; the parquet and the two embedders are fetched from the Hub on first run.

## sample

| | |
| --- | --- |
| systems | 10 of 58: `lng/i31`, `lng/ioc`, `trp/iata`, `trp/icao`, `fin/i42`, `cls/elm`, `cmp/sts`, `bio/aa1`, `bio/aa3`, `com/mor` |
| records | 2,123. systems above 600 rows are sampled to 600 with every answer record kept inside; seed 20260623 |
| questions | 51 natural language questions, each with exactly one answer record, verified against the parquet before any run (`rig/lib/qa.py`); 38 in family A (code to entity), 13 in family B (entity to code) |
| embedders | all-MiniLM-L6-v2 and multilingual-e5-small, onnx, cosine over normalised vectors |
| tokenizer for cost | o200k_base; a missing `tiktoken` yields a declared estimate |

the sample was chosen for coverage of domains and code lengths, not at random from the 58 systems. results are about these ten systems and these 51 questions. the grid over all 58 systems has not been run and is tracked as an issue with its cost.

## metrics

- recall@k: fraction of the 51 questions whose answer record is among the top k chunks by cosine similarity.
- MRR: mean over questions of 1 / rank of the answer record.
- tokens per answer: mean o200k_base token count of the top-1 chunk, the text a model would have to read to answer.
- routing accuracy (`collide.py`): for 182 pairs of records that share a literal across two systems in the sample, fraction of queries whose top-1 record is in the intended system, under three query forms: the bare literal, the literal with the intent in words, the literal with the system name.

with 51 questions one question is 1.96 points of recall; differences under about 4 points are within one question and are not read as differences.

## grid

granularity (record, block of 20 records, whole system as one chunk) x form (fields as `key=value`, one sentence of prose) x key (`GRU`, `iata:GRU`) x embedder: 24 cells, `rig/out/grid.json`.

| cell (e5) | recall@1 | recall@5 | tokens per answer |
| --- | --- | --- | --- |
| record, prose, bare key | 92.2% | 100% | 22.6 |
| record, prose, namespaced key | 88.2% | 100% | 27.0 |
| record, key=value, bare key | 60.8% | 82.4% | 35.4 |
| block of 20, prose, bare key | 47.1% | 80.4% | 459.8 |
| whole system, prose, namespaced key | 90.2% | 100% | 7,467.1 |

minilm is lower in every cell (record, prose, bare key: 52.9% recall@1). the full 24 cells are in the json.

## results

| finding | evidence | file |
| --- | --- | --- |
| one record per chunk, written as a sentence, is the best cell for both embedders | 92.2% recall@1 at 22.6 tokens (e5) | `grid.json` |
| prose beats the same fields as key=value; dropping noisy fields without rewriting lowers recall | +37.3 points of recall@1 for 1.31x tokens; field drop alone −5.9 points at 0.49x tokens | `decomp.json` |
| a block of records halves recall and costs twenty times the tokens | 47.1% at 459.8 tokens | `grid.json` |
| the whole system as one chunk recovers almost as often as a record, at thousands of tokens, because the hit is the system, not the row | 90.2% at 7,467 tokens | `grid.json` |
| a bare literal routes to the right system at the combinatorial floor; the system name in the chunk or key resolves it | 33.0% bare, 98.4% (minilm) and 98.9% (e5) with the intent in words, 100% with the namespace; 182 pairs | `collide.json` |
| across the 58 published systems, 14,911 of 623,141 distinct casefolded literals belong to two or more systems; `AND` and `CAR` to eleven | counts over the full parquet, not the sample | `collide.json` |
| no literal spans more than three of the seven sampled domains; the country (ISO 3166) is the only join axis found | 11 cross-domain overlaps | `universal.json` |
| an 8-bit scalar quantizer keeps recall at a quarter of the index; product quantization does not | q8: 4.0x smaller, recall@1 88.2% unchanged; pq m48: 6.6x smaller, 60.8% | `store.json` |

`hypo_results.json` holds 14 threshold checks on the sample (mode code length, vowel ratio, coverage, collision counts) evaluated by `rig/exp/hypo.py` from predicates in `hypotheses.json`. they describe the corpus; they are not hypotheses about a model and all 14 pass by construction of the thresholds.

## limits of generalization

- 51 questions, 10 systems, two small embedders, one run per cell. the ranking of configurations is stable across the two embedders; the absolute numbers are not estimates for other systems or models.
- the answer key was written by the same person who built the rig, from the data; questions are factual lookups, not user queries.
- the namespaced key costs two questions here; on a corpus where more literals collide the sign may reverse, as the routing experiment suggests but does not show.
- no reranker, no hybrid lexical retrieval, no instruction prefixes for e5.

the rig exposed a defect in release 0.1.0 of the dataset (`lng/ioc` aliases unlabelled, with a footnote marker as a code), fixed in 0.1.1.

## reproduce

```bash
bash rig/bootstrap.sh          # venv via uv, pinned deps, smoke test; python 3.11
.venv/bin/python rig/run.py    # six experiments in order, one json each in rig/out/
.venv/bin/python rig/run.py grid
.venv/bin/python rig/exp/collide.py
```

`CH3_HUB`, `CH3_MINILM`, `CH3_E5` and `CH3_OUT` override the cache and output folders.

| path | what it is |
| --- | --- |
| `rig/lib/` | conf, catalog, qa, corpus, embed, tok, jio |
| `rig/exp/` | grid, decomp, collide, universal, store, hypo |
| `rig/out/` | the json, versioned, regenerated by `run.py` |

code is MIT. the dataset and the embedders carry their own terms.
