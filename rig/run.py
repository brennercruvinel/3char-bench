#!/usr/bin/env python3
"""
run.py  -  orchestrator. runs the experiments in order and writes the json files.

  python rig/run.py            everything
  python rig/run.py grid       the grid only
  python rig/run.py collide universal store hypo

each step is independent and writes its json to out/. none of them writes prose. a missing
prerequisite fails loudly; no exception is swallowed and turned into a "result".
"""
import importlib
import pathlib
import sys

_RIG = pathlib.Path(__file__).resolve().parent
for _p in ("lib", "exp"):
    sys.path.insert(0, str(_RIG / _p))

STEPS = ["grid", "decomp", "collide", "universal", "store", "hypo"]


def main(argv):
    want = argv or STEPS
    bad = [s for s in want if s not in STEPS]
    if bad:
        raise SystemExit(f"etapas desconhecidas: {bad}. validas: {STEPS}")
    for step in [s for s in STEPS if s in want]:
        mod = importlib.import_module(step)
        print(f"== {step} ==", flush=True)
        mod.run()
    print("ok")


if __name__ == "__main__":
    main(sys.argv[1:])
