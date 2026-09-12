#!/usr/bin/env python3
"""
conf.py  -  configuracao portavel do rig.

regra unica: nada de caminho de maquina embutido no codigo. cada recurso resolve
por (1) variavel de ambiente, (2) candidato local, (3) download do hugging face
hub pro cache padrao. nunca um try/except que engole e segue com lixo.

  CH3_HUB     pasta com systems.parquet e data/<dom>/<id>.parquet
              (default: o hub/ do pipeline, se este repo mora em 3char/tools/,
              senao o dataset brennercruvinel/3char baixado do hub)
  CH3_MINILM  pasta do all-MiniLM-L6-v2 onnx   (default: Xenova/all-MiniLM-L6-v2)
  CH3_E5      pasta do multilingual-e5-small    (default: Xenova/multilingual-e5-small)
  CH3_OUT     destino dos artefatos             (default: rig/out)
"""
import os
import pathlib

LIB = pathlib.Path(__file__).resolve().parent     # rig/lib
RIG = LIB.parent                                  # rig
PROJ = RIG.parent                                 # raiz do repo

DATASET = "brennercruvinel/3char"
MODELS = {"minilm": "Xenova/all-MiniLM-L6-v2", "e5": "Xenova/multilingual-e5-small"}
MODEL_FILES = ["onnx/model.onnx", "tokenizer.json", "config.json"]


class Missing(Exception):
    """recurso ausente, com instrucao de como prover. nunca silenciado."""


def _env_dir(env, what):
    p = os.environ.get(env)
    if not p:
        return None
    if not os.path.isdir(p):
        raise Missing(f"{what}: ${env}={p} nao e diretorio.")
    return p


def _snapshot(repo, patterns, repo_type=None):
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo, repo_type=repo_type, allow_patterns=patterns)
    except Exception as e:                       # sem rede e sem cache: falha em voz alta
        raise Missing(f"nao consegui obter {repo} do hub ({e.__class__.__name__}: {e}). "
                      f"baixe uma vez com rede ou aponte a variavel de ambiente.") from e


def hub():
    p = _env_dir("CH3_HUB", "dataset 3char")
    if p:
        return p
    local = PROJ.parent.parent / "hub"           # 3char/hub, quando o rig mora em 3char/tools/bench
    if (local / "systems.parquet").is_file():
        return str(local)
    return _snapshot(DATASET, ["systems.parquet", "data/*/*.parquet"], repo_type="dataset")


def parquet(system_id):
    return os.path.join(hub(), "data", f"{system_id}.parquet")


def _model(name, env):
    return _env_dir(env, f"embedder {name}") or _snapshot(MODELS[name], MODEL_FILES)


def minilm():
    return _model("minilm", "CH3_MINILM")


def e5():
    return _model("e5", "CH3_E5")


def out(name=""):
    d = os.environ.get("CH3_OUT", str(RIG / "out"))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name) if name else d
