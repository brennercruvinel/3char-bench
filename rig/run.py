#!/usr/bin/env python3
"""
run.py  -  orquestrador. roda os experimentos na ordem e escreve os json.

  python rig/run.py            tudo
  python rig/run.py grid       so a grade
  python rig/run.py collide universal store hypo

cada etapa e independente e escreve seu json em out/. nenhuma monta prosa. falha em
voz alta se um pre-requisito faltar, nada de excecao engolida virando 'resultado'.
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
