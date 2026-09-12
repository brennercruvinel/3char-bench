#!/usr/bin/env python3
"""
tok.py  -  contagem de tokens.

usa o tokenizador o200k (a familia gpt-4o/o200k_base) quando disponivel. se nao
estiver, NAO inventa um numero plausivel em silencio: degrada para uma estimativa
declarada (~4 chars/token) e expoe qual modo esta ativo via tok.MODE, para que o
relatorio possa dizer a verdade sobre como o custo foi medido.
"""
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("o200k_base")
    MODE = "o200k_base"

    def count(s):
        return len(_ENC.encode(s or ""))
except Exception:                                   # tiktoken ausente ou offline
    MODE = "estimado(~4ch/tok)"

    def count(s):
        return max(1, len(s or "") // 4)
