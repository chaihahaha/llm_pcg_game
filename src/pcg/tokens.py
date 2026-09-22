"""Cheap, dependency-free token estimation and context-budget helpers.

The real tokenizer is Qwen's (BPE, ~150k vocab).  Installed tokenizers are not
guaranteed to be present, so we approximate:

* CJK / full-width character  -> ~1.0 token
* ASCII letters / digits      -> ~0.28 token
* everything else             -> ~0.5 token

This deliberately over-estimates a little, which keeps us safely inside the
context window instead of blowing past it.
"""
from __future__ import annotations

import re
from typing import Iterable

_CJK = re.compile(r"[\u2e80-\u9fff\u3000-\u303f\uff00-\uffef]")
_ASCII = re.compile(r"[A-Za-z0-9]")
_MARKER_TOKENS = 3  # chat template overhead per message


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    ascii_n = len(_ASCII.findall(text))
    other = max(0, len(text) - cjk - ascii_n)
    return int(cjk * 1.0 + ascii_n * 0.28 + other * 0.5) + 1


def estimate_messages(messages: Iterable[dict]) -> int:
    return sum(estimate_tokens(str(m.get("content", ""))) + _MARKER_TOKENS for m in messages)


def clamp_text(text: str, max_tokens: int, keep: str = "head") -> str:
    """Truncate text to roughly max_tokens, preferring a newline boundary."""
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    # binary-ish search on characters (fast enough at these sizes)
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        chunk = text[-mid:] if keep == "tail" else text[:mid]
        if estimate_tokens(chunk) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    cut = text[-lo:] if keep == "tail" else text[:lo]
    if keep == "head":
        nl = cut.rfind("\n")
        if nl > len(cut) * 0.5:
            cut = cut[:nl]
        return cut + "\n...[内容截断]..."
    nl = cut.find("\n")
    if 0 <= nl < len(cut) * 0.5:
        cut = cut[nl + 1:]
    return "...[内容截断]...\n" + cut
