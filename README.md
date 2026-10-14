# 3char-bench

An experimental report on presenting the [3char](https://huggingface.co/datasets/brennercruvinel/3char) tables to a dense retriever: which chunk granularity, chunk form, key form and vector storage retrieve the right record. All measurements use a fixed sample and a fixed seed. This repository holds the rig and its outputs; the Parquet files and the two embedders are downloaded from the Hugging Face Hub on the first run.

## Setup

- **Systems:** 10 of the 58 published systems: `lng/i31`, `lng/ioc`, `trp/iata`, `trp/icao`, `fin/i42`, `cls/elm`, `cmp/sts`, `bio/aa1`, `bio/aa3` and `com/mor`.
- **Records:** 2,123. Systems with more than 600 rows are sampled down to 600, keeping every answer record (seed 20260623).
- **Questions:** 51 natural-language questions, each with exactly one answer record, verified against the Parquet files before any run (`rig/lib/qa.py`). 38 belong to family A (code to entity) and 13 to family B (entity to code).
- **Embedders:** all-MiniLM-L6-v2 and multilingual-e5-small, as ONNX exports, with cosine similarity over normalized vectors. The e5 model uses its standard `query:` and `passage:` prefixes.
- **Language:** the questions, the prose templates and the routing phrases are written in Portuguese. multilingual-e5-small is a multilingual model, while all-MiniLM-L6-v2 was trained on English text.
- **Token counts:** o200k_base. Without `tiktoken`, the rig falls back to an estimate and records that it did so.

The ten systems were chosen to cover domains and code lengths, not drawn at random from the 58, so the results describe these systems and these 51 questions. The grid has not been run over all 58 systems.

## Metrics

- **recall@k:** the fraction of the 51 questions whose answer record is among the top k chunks by cosine similarity.
- **MRR:** the mean over questions of 1 / rank of the answer record.
- **Tokens per answer:** the mean o200k_base token count of the top-1 chunk, which is the text a model would have to read to answer.
- **Routing accuracy** (`collide.py`): the fraction of queries whose top-1 record belongs to the intended system, over 182 (literal, system) pairs drawn from the 60 most ambiguous literals in the sample. Three query forms are compared: the bare literal, the literal with the intended meaning in words, and the literal prefixed with the system name.

With 51 questions, one question is worth 1.96 points of recall, so differences below about 4 points (two questions) are not interpreted.

## Grid

The grid crosses granularity (one record, a block of 20 records, or the whole system as one chunk), form (fields as `key=value` or one sentence of prose), key (bare `GRU` or namespaced `iata:GRU`) and embedder, for 24 cells in total. Recall@1 for every cell is shown below; recall@5 and MRR are in `rig/out/grid.json`.

| Granularity | Form | Key | e5 recall@1 | MiniLM recall@1 | Tokens per answer |
| --- | --- | --- | ---: | ---: | ---: |
| Record | Prose | Bare | 92.2% | 52.9% | 22.6 |
| Record | Prose | Namespaced | 88.2% | 52.9% | 27.0 |
| Record | key=value | Bare | 60.8% | 29.4% | 35.4 |
| Record | key=value | Namespaced | 72.5% | 39.2% | 39.9 |
| Block of 20 | Prose | Bare | 47.1% | 29.4% | 459.8 |
| Block of 20 | Prose | Namespaced | 47.1% | 27.5% | 549.6 |
| Block of 20 | key=value | Bare | 21.6% | 15.7% | 741.9 |
| Block of 20 | key=value | Namespaced | 31.4% | 11.8% | 831.7 |
| Whole system | Prose | Bare | 84.3% | 82.4% | 6,257.9 |
| Whole system | Prose | Namespaced | 90.2% | 82.4% | 7,467.1 |
| Whole system | key=value | Bare | 52.9% | 60.8% | 8,659.0 |
| Whole system | key=value | Namespaced | 64.7% | 60.8% | 9,868.2 |

The two embedders do not share a best configuration. With e5, one record per chunk written as a sentence ranks first, at 92.2% recall@1 and 22.6 tokens per answer. With MiniLM, the whole system as one prose chunk ranks first (82.4% with either key), at 6,258 to 7,467 tokens per answer, while one record per sentence reaches 52.9%. MiniLM scores below e5 in 11 of the 12 configurations; the exception is the whole system as `key=value` with the bare key (60.8% against 52.9%).

Two patterns hold for both embedders. Prose outperforms `key=value` in every pair of configurations that differ only in form. A block of 20 records retrieves less than both a single record and the whole system for every form and key, while costing about 20 times the tokens of a single record.

The whole system as one chunk scores high because a hit only requires retrieving the right system; the reader must then find the row among thousands of tokens.

## Other results

- **Decomposition** (`decomp.json`; e5, one record per chunk, bare key). Dropping noisy fields from the `key=value` chunk without rewriting it lowers recall@1 by 5.9 points (60.8% to 54.9%) at 0.49 times the tokens. Rewriting the remaining fields as prose then raises recall@1 by 37.3 points, to 92.2%, at 1.31 times the tokens. The net change is +31.4 points at 0.64 times the original token count.
- **Routing** (`collide.json`). With the bare literal as both query and document, the systems that share a literal cannot be told apart, and accuracy is 33.0%, the rate expected from a random choice among them. With the intended meaning in words, accuracy is 98.4% with MiniLM and 98.9% with e5. With the system name as a prefix, it is 100% with both.
- **Collisions in the full dataset** (`collide.json`). Across the 58 published systems, 14,911 of 623,141 distinct case-folded literals belong to two or more systems, and `and` and `car` belong to eleven. These counts cover the full Parquet files, not the sample.
- **Shared literals across domains** (`universal.json`). Eleven pairs of sampled systems from different domains share literals, and no literal spans more than three of the seven sampled domains. The five systems with a country key (`fin/i42`, `lng/i31`, `lng/ioc`, `trp/iata`, `trp/icao`) join through ISO 3166, with coverage of at least 99.1%; the rig found no join key for the other five. No join axis other than the country was tested.
- **Vector storage** (`store.json`; e5, one record per sentence with the namespaced key, 2,123 vectors of dimension 384). An 8-bit scalar quantizer makes the index 4.0 times smaller and leaves recall@1 unchanged at 88.2%. Product quantization with 48 subquantizers makes it 6.6 times smaller and lowers recall@1 to 60.8%.

`hypo_results.json` holds 14 threshold checks on the sample (modal code length, vowel ratio, coverage, collision counts), evaluated by `rig/exp/hypo.py` against predicates in `hypotheses.json`. They describe the corpus rather than test hypotheses about a model, and all 14 pass by construction of their thresholds.

The rig also exposed a defect in release 0.1.0 of the dataset: the `lng/ioc` aliases were unlabelled and included a footnote marker stored as a code. The defect was fixed in release 0.1.1.

## Limitations

- The study uses 51 questions, 10 systems, two small embedders and one run per cell. The ranking of configurations differs between the two embedders, and the absolute numbers do not estimate performance on other systems or models.
- Because the questions and prose chunks are in Portuguese, the gap between the two embedders may reflect language coverage as much as model quality; this was not tested.
- The answer key was written from the data by the person who built the rig. The questions are factual lookups, not real user queries.
- With e5, the namespaced key costs two questions when one record is written as a sentence (88.2% against 92.2%). On a corpus where more literals collide, the sign may reverse, as the routing experiment suggests but does not show.
- No reranker and no hybrid lexical retrieval were tested, and e5 was used without task instructions.

## Reproducing

```bash
bash rig/bootstrap.sh            # creates the venv with uv, installs pinned dependencies, runs a smoke test; Python 3.11
.venv/bin/python rig/run.py      # runs the six experiments in order, writing one JSON file each to rig/out/
.venv/bin/python rig/run.py grid # runs a single experiment
```

`CH3_HUB`, `CH3_MINILM`, `CH3_E5` and `CH3_OUT` override the dataset, embedder and output folders.

- `rig/lib/`: configuration, catalog, questions, corpus, embedding, tokenization and JSON I/O.
- `rig/exp/`: the six experiments (grid, decomp, collide, universal, store, hypo).
- `rig/out/`: the versioned JSON outputs, regenerated by `run.py`.

The code is released under the MIT license. The dataset and the embedders carry their own terms.
