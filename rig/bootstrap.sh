#!/usr/bin/env bash
# bootstrap.sh - rebuilds the rig environment from scratch, no machine path embedded.
# one command, pinned environment. dataset and embedders come from the hugging face hub on
# first run (default huggingface_hub cache), or from the CH3_* variables.
set -euo pipefail
cd "$(dirname "$0")/.."   # project root

PY="${PY:-3.11}"          # onnxruntime/faiss have arm64 wheels for 3.11; 3.14 not yet

echo "== venv (uv, python $PY) =="
uv venv --python "$PY" .venv
uv pip install --python .venv/bin/python -r requirements.txt

echo "== smoke test (downloads the dataset and embedders if missing) =="
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "rig/lib")
import catalog, conf, embed, qa, tok
print("dataset :", conf.hub(), f"({len(catalog.published())} sistemas publicados)")
print("minilm  :", conf.minilm())
print("e5      :", conf.e5())
print("tokens  :", tok.MODE)
qs = qa.build(); print("gabarito:", len(qs), "perguntas resolvem no corpus")
v = embed.get("e5").docs([catalog.load("cls/elm")[0].verb_chunk(True)]); print("embed   :", v.shape)
PY

echo
echo "pronto. rode:  .venv/bin/python rig/run.py"
