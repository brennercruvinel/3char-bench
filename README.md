# 3char-bench

When a code list from [3char](https://huggingface.co/datasets/brennercruvinel/3char) is indexed for a dense retriever, how should each record be written, and how much should go into each chunk, for the right row to come back? This repository answers that question on a fixed sample of ten systems and 51 questions with two small embedders, and reports what the sample can and cannot support. It contains the experimental rig and its outputs; the Parquet files and the embedders are downloaded from the Hugging Face Hub on the first run.

## Setup

The sample covers ten systems chosen to span domains and code lengths: `lng/i31`, `lng/ioc`, `trp/iata`, `trp/icao`, `fin/i42`, `cls/elm`, `cmp/sts`, `bio/aa1`, `bio/aa3` and `com/mor`. Systems with more than 600 rows are sampled down to 600, always keeping the answer records, which gives 2,123 records (seed 20260623). The systems were not drawn at random from the 58, so the results describe these systems only.

There are 51 natural-language questions, each with exactly one answer record, verified against the Parquet files before any run (`rig/lib/qa.py`). Thirty-eight ask for the entity behind a code and thirteen for the code of an entity. The questions, the prose templates used to write records and the phrases used in the routing test are in Portuguese.

The embedders are multilingual-e5-small, a multilingual model used with its standard `query:` and `passage:` prefixes, and all-MiniLM-L6-v2, which was trained on English text. Both run as ONNX exports, with cosine similarity over normalized vectors. Token counts use the o200k_base tokenizer; without `tiktoken`, the rig falls back to an estimate and records that it did so.

## What counts as a hit

The main metric is recall@1, the share of questions whose answer record is in the top-ranked chunk, together with the number of tokens in that chunk, which is what a reader would have to process to answer. With 51 questions, one question is worth 1.96 points, so differences below about four points are not interpreted.

A hit does not mean the same thing at every chunk size. When a chunk holds one record or a block of 20 records, retrieving it retrieves the answer row with little else; this is record retrieval. When a chunk holds a whole system, retrieving it only shows that the right list was found, and the row still has to be located among thousands of tokens; this is system retrieval. The two are reported separately and are not compared as if they measured the same thing.

## Results

The grid crosses chunk size (one record, a block of 20, a whole system), record form (fields as `key=value` or one sentence of prose), key form (bare `GRU` or namespaced `iata:GRU`) and embedder, for 24 cells. Recall@5 and MRR for every cell are in `rig/out/grid.json`.

**Record retrieval** (recall@1):

| Chunk | Form | Key | e5 | MiniLM | Tokens per answer |
| --- | --- | --- | ---: | ---: | ---: |
| One record | Prose | Bare | 92.2% | 52.9% | 22.6 |
| One record | Prose | Namespaced | 88.2% | 52.9% | 27.0 |
| One record | key=value | Bare | 60.8% | 29.4% | 35.4 |
| One record | key=value | Namespaced | 72.5% | 39.2% | 39.9 |
| Block of 20 | Prose | Bare | 47.1% | 29.4% | 459.8 |
| Block of 20 | Prose | Namespaced | 47.1% | 27.5% | 549.6 |
| Block of 20 | key=value | Bare | 21.6% | 15.7% | 741.9 |
| Block of 20 | key=value | Namespaced | 31.4% | 11.8% | 831.7 |

**System retrieval** (the whole system as one chunk; recall@1 here means the right system was ranked first):

| Form | Key | e5 | MiniLM | Tokens per answer |
| --- | --- | ---: | ---: | ---: |
| Prose | Bare | 84.3% | 82.4% | 6,257.9 |
| Prose | Namespaced | 90.2% | 82.4% | 7,467.1 |
| key=value | Bare | 52.9% | 60.8% | 8,659.0 |
| key=value | Namespaced | 64.7% | 60.8% | 9,868.2 |

### How should a record be written?

As a sentence. Prose outperforms `key=value` in every pair of cells that differ only in form, for both embedders and at every chunk size. With e5 and one record per chunk, prose reaches 92.2% against 60.8%.

A decomposition of that difference (`rig/out/decomp.json`; e5, one record, bare key) separates two steps. Removing noisy fields from the `key=value` chunk, without rewriting it, lowers recall@1 by 5.9 points (to 54.9%) and roughly halves the tokens. Rewriting the remaining fields as a sentence then adds 37.3 points, for 1.31 times the tokens of the trimmed chunk. The gain comes from the sentence form, not from removing fields.

### How much should go into a chunk?

One record. For every form, key and embedder, a block of 20 records retrieves the answer less often than a single record, while costing about 20 times as many tokens.

Whole-system chunks rank the right system first for 82.4% to 90.2% of questions in prose. That is a different outcome, not a better one: a reader still has to find the row among 6,000 to 7,500 tokens. With e5, one record per chunk retrieves the row itself more often (92.2%) at 22.6 tokens. With MiniLM, record retrieval is much weaker (52.9% at best), which may reflect the Portuguese questions as much as the model; this was not tested.

The namespaced key has no consistent effect. With e5 it helps in `key=value` chunks (72.5% against 60.8%) and costs two questions in prose (88.2% against 92.2%).

### Can a shared code be routed to the right system?

Only with context (`rig/out/collide.json`). The test uses 182 pairs of a code and a system, drawn from the 60 codes shared by the most sampled systems, and asks whether the top-ranked record belongs to the intended system. With the bare code as both query and document, the systems that share it cannot be told apart, and accuracy is 33.0%, the rate of a random choice among them. With the intended meaning written out in the query, accuracy is 98.4% with MiniLM and 98.9% with e5. With the system name as a prefix of both query and document, it is 100% with both.

### How small can the index be?

An 8-bit scalar quantizer keeps recall (`rig/out/store.json`; e5, one record per sentence with the namespaced key, 2,123 vectors of dimension 384). It makes the index 4.0 times smaller and leaves recall@1 unchanged at 88.2%. Product quantization with 48 subquantizers makes it 6.6 times smaller and lowers recall@1 to 60.8%.

### Descriptive outputs

Two further outputs describe the corpus rather than retrieval. `rig/out/universal.json` finds that no literal in the sample spans more than three of its seven domains, and that the five systems with a country key (`fin/i42`, `lng/i31`, `lng/ioc`, `trp/iata`, `trp/icao`) join through ISO 3166 with at least 99.1% coverage; no other join key was tested. `rig/out/hypo_results.json` holds 14 threshold checks on the sample, such as modal code length and vowel ratio; their thresholds were set with knowledge of the data, so they describe the corpus and are not tests of hypotheses.

While building the rig, a check that every IOC alias resolves in ISO 3166 exposed a defect in release 0.1.0 of the dataset (unlabelled aliases, including a footnote marker stored as a code), which was fixed in release 0.1.1.

## What this does not show

The study uses 51 questions, ten systems, two small embedders and one run per cell. The answer key was written from the data by the person who built the rig, and the questions are factual lookups rather than real user queries. Because the questions and prose are in Portuguese, the difference between the embedders may reflect language coverage as much as model quality. No reranker, hybrid lexical retrieval or task instructions for e5 were tested. The absolute numbers do not estimate performance on other systems or models.

## Reproducing

```bash
bash rig/bootstrap.sh            # creates the venv with uv, installs pinned dependencies, runs a smoke test; Python 3.11
.venv/bin/python rig/run.py      # runs the six experiments in order, writing one JSON file each to rig/out/
.venv/bin/python rig/run.py grid # runs a single experiment
```

`CH3_HUB`, `CH3_MINILM`, `CH3_E5` and `CH3_OUT` override the dataset, embedder and output folders. The experiments are in `rig/exp/` (`grid`, `decomp`, `collide`, `universal`, `store`, `hypo`), the shared code in `rig/lib/`, and the versioned outputs in `rig/out/`, one JSON file per experiment.

The code is released under the MIT license. The dataset and the embedders carry their own terms.
