#!/usr/bin/env python3
"""
conf.py  -  portable configuration of the rig.

one rule: no machine path embedded in the code. every resource resolves by (1) environment
variable, (2) local candidate, (3) download from the hugging face hub into the default cache.
never a try/except that swallows an error and carries on with garbage.

  CH3_HUB     folder with systems.parquet and data/<dom>/<id>.parquet
              (default: the pipeline's hub/ when this repo lives in 3char/tools/,
              otherwise the dataset brennercruvinel/3char downloaded from the hub)
  CH3_MINILM  folder of all-MiniLM-L6-v2 onnx    (default: Xenova/all-MiniLM-L6-v2)
  CH3_E5      folder of multilingual-e5-small    (default: Xenova/multilingual-e5-small)
  CH3_OUT     destination of the artifacts        (default: rig/out)
"""
import os
import pathlib

LIB = pathlib.Path(__file__).resolve().parent     # rig/lib
RIG = LIB.parent                                  # rig
PROJ = RIG.parent                                 # repo root

DATASET = "brennercruvinel/3char"
MODELS = {"minilm": "Xenova/all-MiniLM-L6-v2", "e5": "Xenova/multilingual-e5-small"}
MODEL_FILES = ["onnx/model.onnx", "tokenizer.json", "config.json"]


class Missing(Exception):
    """missing resource, with the instruction on how to provide it. never silenced."""


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
    except Exception as e:                       # no network and no cache: fail loudly
        raise Missing(f"nao consegui obter {repo} do hub ({e.__class__.__name__}: {e}). "
                      f"baixe uma vez com rede ou aponte a variavel de ambiente.") from e


def hub():
    p = _env_dir("CH3_HUB", "dataset 3char")
    if p:
        return p
    local = PROJ.parent.parent / "hub"           # 3char/hub, when the rig lives in 3char/tools/bench
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
