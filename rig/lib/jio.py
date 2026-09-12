#!/usr/bin/env python3
"""jio.py  -  leitura/escrita de artefatos json em out/, determinista e legivel."""
import json

import conf


def dump(obj, name):
    path = conf.out(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, sort_keys=False)
    return path


def load(name):
    with open(conf.out(name), encoding="utf-8") as f:
        return json.load(f)
