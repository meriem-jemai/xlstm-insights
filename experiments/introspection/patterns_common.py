from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import importlib
import numpy as np

Span = Tuple[int, int]

def _merge_spans(spans: List[Span]) -> List[Span]:
    if not spans:
        return []
    spans = sorted((int(s), int(e)) for (s, e) in spans if e > s)
    merged = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged

def _mask_to_spans(mask: np.ndarray) -> List[Span]:
    mask = np.asarray(mask).astype(bool)
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]

def _provider_to_spans(patt: Dict[str, np.ndarray], S: int) -> Dict[str, List[Span]]:
    out: Dict[str, List[Span]] = {}
    for k, v in patt.items():
        a = np.asarray(v)
        spans: List[Span] = []
        if a.ndim == 2 and a.shape[1] == 2:
            for s, e in a.astype(int):
                s = max(0, min(S, int(s))); e = max(0, min(S, int(e)))
                if e > s:
                    spans.append((s, e))
        elif a.ndim == 1 and a.size == S and (a.dtype == np.bool_ or np.all((a == 0) | (a == 1))):
            spans = _mask_to_spans(a[:S])
        else:
            idx = a.ravel().astype(int, copy=False)
            idx = idx[(idx >= 0) & (idx < S)]
            if idx.size:
                idx = np.unique(idx)
                spans = [(int(t), int(t) + 1) for t in idx]
        spans = [(max(0, s), min(S, e)) for (s, e) in spans if 0 <= s < e <= S]
        out[k] = _merge_spans(spans)
    return out

def load_spans_from_provider(text: str, S: int, provider_path: Optional[str], select: Optional[list[str]] = None) -> Dict[str, List[Span]]:
    if not provider_path:
        raise RuntimeError("provider_path is required")
    mod, func = provider_path.rsplit(".", 1)
    provider_fn = getattr(importlib.import_module(mod), func)
    patt = provider_fn(text[:S])
    if not isinstance(patt, dict):
        raise RuntimeError("Pattern provider returned non-dict")
    spans = _provider_to_spans(patt, S)
    if select:
        wanted = {s.lower() for s in select}
        spans = {k: v for k, v in spans.items() if k.lower() in wanted}
    return spans

def spans_to_mask(spans: List[Span], S: int) -> np.ndarray:
    m = np.zeros(S, dtype=bool)
    for s, e in spans:
        m[s:e] = True
    return m

def spans_dict_to_masks(spans: Dict[str, List[Span]], S: int) -> Dict[str, np.ndarray]:
    return {k: spans_to_mask(v, S) for k, v in spans.items()}
