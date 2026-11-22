
import os
import numpy as np
from typing import Dict, List, Tuple

__all__ = ["patterns"]

COPY_SEQ_LEN = int(os.environ.get("COPY_SEQ_LEN", 32))
COPY_GAP = int(os.environ.get("COPY_GAP", 40))
COPY_OUT_LEN = int(os.environ.get("COPY_OUT_LEN", 32))
COPY_HAS_NEWLINE = os.environ.get("COPY_HAS_NEWLINE", "1") not in {"0", "false", "False"}

Span = Tuple[int, int]

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

def _make_copytask_spans(text: str, seq_len: int = COPY_SEQ_LEN, gap: int = COPY_GAP, out_len: int = COPY_OUT_LEN, has_newline: bool = COPY_HAS_NEWLINE) -> Dict[str, np.ndarray]:
    line_len = seq_len + gap + out_len + (1 if has_newline else 0)
    if line_len <= 0:
        return {"Lines": np.empty((0, 2), dtype=int), "Source blocks": np.empty((0, 2), dtype=int), "Gap blocks": np.empty((0, 2), dtype=int), "Copy blocks": np.empty((0, 2), dtype=int)}
    N = len(text)
    n_lines = N // line_len
    LINE_R: List[Span] = []
    SRC_R: List[Span] = []
    GAP_R: List[Span] = []
    COPY_R: List[Span] = []
    for L in range(n_lines):
        base = L * line_len
        ss = base
        se = ss + seq_len - 1
        gs = ss + seq_len
        ge = gs + gap - 1
        cs = ge + 1
        ce = cs + out_len - 1
        le = base + line_len
        LINE_R.append((base, le))
        SRC_R.append((ss, se + 1))
        GAP_R.append((gs, ge + 1))
        COPY_R.append((cs, ce + 1))
    return {"Lines": _clip_spans(LINE_R, N), "Source blocks": _clip_spans(SRC_R, N), "Gap blocks": _clip_spans(GAP_R, N), "Copy blocks": _clip_spans(COPY_R, N)}

def patterns(text: str) -> Dict[str, np.ndarray]:
    return _make_copytask_spans(text)
