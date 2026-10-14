#!/usr/bin/env python3
"""
tok.py  -  token counting.

uses the o200k tokenizer (the gpt-4o / o200k_base family) when available. if it is not, it does
NOT invent a plausible number silently: it degrades to a declared estimate (~4 chars/token) and
exposes the active mode through tok.MODE, so the report can say how the cost was measured.
"""
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("o200k_base")
    MODE = "o200k_base"

    def count(s):
        return len(_ENC.encode(s or ""))
except Exception:                                   # tiktoken missing or offline
    MODE = "estimated(~4ch/tok)"

    def count(s):
        return max(1, len(s or "") // 4)
