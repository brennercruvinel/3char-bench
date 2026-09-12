#!/usr/bin/env bash
# bootstrap.sh - reproduz o ambiente do rig do zero, sem caminho de maquina embutido.
# um comando, ambiente travado. dataset e embedders vem do hugging face hub na
# primeira execucao (cache padrao do huggingface_hub), ou das variaveis CH3_*.
set -euo pipefail
cd "$(dirname "$0")/.."   # raiz do projeto

PY="${PY:-3.11}"          # onnxruntime/faiss tem wheels arm64 p/ 3.11; 3.14 ainda nao

echo "== venv (uv, python $PY) =="
uv venv --python "$PY" .venv
uv pip install --python .venv/bin/python -r requirements.txt

echo "== smoke test (baixa dataset e embedders se ausentes) =="
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
