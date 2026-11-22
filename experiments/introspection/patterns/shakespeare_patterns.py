import os, re, numpy as np
from typing import Dict, List, Tuple

__all__ = ["patterns"]

Span = Tuple[int, int]
LONG_WORD_MIN_LEN = int(os.environ.get("LONG_WORD_MIN_LEN", 8))
CUSTOM_WORDS: List[str] = ["Citizen"]

def _clip_spans(spans: List[Span], S: int) -> np.ndarray:
    if not spans:
        return np.empty((0, 2), dtype=int)
    out: List[Span] = []
    for s, e in spans:
        s = int(max(0, min(S, s)))
        e = int(max(0, min(S, e)))
        if e > s:
            out.append((s, e))
    if not out:
        return np.empty((0, 2), dtype=int)
    out.sort(key=lambda x: (x[0], x[1]))
    return np.asarray(out, dtype=int)

def _build_char_token_spans(text: str):
    tokens, spans = [], []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in ".,;:!?()[]{}\"'-":
            tokens.append(ch); spans.append((i, i + 1)); i += 1; continue
        j = i
        while j < n and (text[j].isalnum() or text[j] in "_'"):
            j += 1
        if j == i:
            tokens.append(ch); spans.append((i, i + 1)); i += 1
        else:
            tokens.append(text[i:j]); spans.append((i, j)); i = j
    return tokens, spans

def patterns(text: str) -> Dict[str, np.ndarray]:
    toks, spans = _build_char_token_spans(text)
    S = len(text)
    long_spans = [(s, e) for tok, (s, e) in zip(toks, spans) if len(tok) >= LONG_WORD_MIN_LEN]
    caps_spans = [(s, e) for tok, (s, e) in zip(toks, spans) if tok and tok[0].isalpha() and tok[0].isupper()]
    punct_set = set(".,;:!?()[]{}\"'-")
    punct_spans = [(s, e) for tok, (s, e) in zip(toks, spans) if len(tok) == 1 and tok in punct_set]
    newline_spans = [(i, i + 1) for i, ch in enumerate(text) if ch == "\n"]
    out: Dict[str, np.ndarray] = {
        "Long words": _clip_spans(long_spans, S),
        "Capitalized words": _clip_spans(caps_spans, S),
        "Punctuation": _clip_spans(punct_spans, S),
        "Newlines": _clip_spans(newline_spans, S),
    }
    for w in CUSTOM_WORDS:
        if not w:
            continue
        is_simple = all((c.isalnum() or c.isspace()) for c in w)
        rx = (r"\b" + re.escape(w) + r"\b") if is_simple else re.escape(w)
        spans_w = [(m.start(), m.end()) for m in re.finditer(rx, text, flags=re.IGNORECASE)]
        out["Word: " + w] = _clip_spans(spans_w, S)
    return out
