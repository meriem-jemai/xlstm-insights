
import re, io, tokenize, ast, sys
import numpy as np
from typing import Dict, List, Tuple

__all__ = ["patterns"]

Span = Tuple[int, int]

def _merge(spans: List[Span]) -> List[Span]:
    if not spans: return []
    spans = sorted((int(s), int(e)) for s, e in spans if e > s)
    out = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out

def _clip_to_N(spans: List[Span], N: int) -> np.ndarray:
    if not spans: return np.empty((0,2), dtype=int)
    out = []
    for s, e in spans:
        s = max(0, min(N, int(s))); e = max(0, min(N, int(e)))
        if e > s: out.append((s, e))
    if not out: return np.empty((0,2), dtype=int)
    out.sort(key=lambda x: (x[0], x[1]))
    return np.asarray(out, dtype=int)

def _line_starts(text: str) -> List[int]:
    return [0] + [m.end() for m in re.finditer("\n", text)]

def _line_bounds(text: str) -> Tuple[List[int], List[int]]:
    starts = _line_starts(text)
    ends = starts[1:] + [len(text)]
    return starts, ends

_FENCES = [
    re.compile(r"(?ms)^[ \t]*```(?:\s*(?:python|py))?\s*\n(.*?)\n```"),
    re.compile(r"(?ms)^[ \t]*~~~(?:\s*(?:python|py))?\s*\n(.*?)\n~~~"),
]

def _find_code_blocks(raw: str) -> List[Tuple[int,int,str]]:
    blocks: List[Tuple[int,int,str]] = []
    for rx in _FENCES:
        for m in rx.finditer(raw):
            b, e = m.start(1), m.end(1)
            blocks.append((b, e, raw[b:e]))
    if blocks:
        blocks.sort(key=lambda t: (t[0], t[1]))
        out, last_e = [], -1
        for b, e, s in blocks:
            if b <= last_e:
                last_e = max(last_e, e)
                continue
            out.append((b, e, s)); last_e = e
        return out
    try:
        ast.parse(raw, mode="exec")
        return [(0, len(raw), raw)]
    except SyntaxError:
        return []

def _comment_spans_block(code: str, abs0: int) -> List[Span]:
    spans: List[Span] = []
    starts = _line_starts(code)
    def lc2abs(ln, col): return abs0 + starts[max(0, ln-1)] + col
    try:
        for ttype, _tstr, (sln, scol), (eln, ecol), _ in tokenize.generate_tokens(io.StringIO(code).readline):
            if ttype == tokenize.COMMENT:
                spans.append((lc2abs(sln, scol), lc2abs(eln, ecol)))
    except tokenize.TokenError:
        pass
    return spans

def _newline_pair_spans_block(code: str, abs0: int) -> List[Span]:
    N = len(code)
    starts, ends = _line_bounds(code)
    spans: List[Span] = []
    for (b, e), (b2, e2) in zip(zip(starts, ends), zip(starts[1:] + [None], ends[1:] + [None])):
        j = e - 1
        while j >= b and code[j] in (" ", "\t", "\n", "\r"):
            j -= 1
        if j >= b:
            spans.append((abs0 + j, abs0 + j + 1))
        if b2 is not None and e2 is not None and b2 < e2:
            spans.append((abs0 + b2, abs0 + b2 + 1))
    return spans

def _indent_ws_spans_block(code: str, abs0: int) -> List[Span]:
    starts, ends = _line_bounds(code)
    spans: List[Span] = []
    for b, e in zip(starts, ends):
        i = b
        while i < e and code[i] == ' ':
            i += 1
        if i > b:
            spans.append((abs0 + b, abs0 + i))
    return spans

def _parse_ast(code: str) -> ast.AST:
    kw = {}
    if sys.version_info >= (3, 8):
        kw["type_comments"] = True
        try:
            kw["feature_version"] = sys.version_info.minor
        except TypeError:
            kw.pop("feature_version", None)
    return ast.parse(code, mode="exec", **kw)

def _node_end_lineno(n: ast.AST) -> int:
    end_ln = getattr(n, "end_lineno", None)
    if isinstance(end_ln, int): return end_ln
    best = getattr(n, "lineno", 1)
    for field in ("decorator_list", "body", "orelse", "finalbody", "handlers"):
        val = getattr(n, field, None)
        if isinstance(val, list):
            for ch in val:
                if isinstance(ch, ast.AST): best = max(best, _node_end_lineno(ch))
        elif isinstance(val, ast.AST):
            best = max(best, _node_end_lineno(val))
    for ch in ast.iter_child_nodes(n):
        best = max(best, _node_end_lineno(ch))
    return best

def _full_line_span(lo_ln: int, hi_ln: int, code: str, abs0: int) -> Span:
    starts, _ = _line_bounds(code)
    def line_lo(ln): return abs0 + starts[max(0, ln-1)]
    def line_hi(ln):
        idx = ln-1
        end_local = (starts[idx+1] if 0 <= idx+1 < len(starts) else len(code))
        return abs0 + end_local
    s = line_lo(lo_ln)
    e = max(s, min(line_hi(hi_ln), abs0 + len(code)))
    return (s, e)

def _if_for_def_full_line_spans_block(code: str, abs0: int):
    if_sp = []; for_sp = []; def_sp = []
    tree = _parse_ast(code)
    for n in ast.walk(tree):
        if isinstance(n, ast.If):
            if_sp.append(_full_line_span(getattr(n, "lineno", 1), _node_end_lineno(n), code, abs0))
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            for_sp.append(_full_line_span(getattr(n, "lineno", 1), _node_end_lineno(n), code, abs0))
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            s_ln = getattr(n, "lineno", 1)
            if getattr(n, "decorator_list", None):
                ds = [getattr(d, "lineno", s_ln) for d in n.decorator_list]
                if ds: s_ln = min([s_ln] + ds)
            def_sp.append(_full_line_span(s_ln, _node_end_lineno(n), code, abs0))
    return if_sp, for_sp, def_sp

def _def_name_uscore_spans_block(code: str, abs0: int) -> List[Span]:
    starts, _ = _line_bounds(code)
    def line_slice(ln: int):
        idx = ln-1
        lo = starts[max(0, idx)]
        hi = (starts[idx+1] if 0 <= idx+1 < len(starts) else len(code))
        return lo, hi
    spans: List[Span] = []
    try:
        tree = _parse_ast(code)
    except SyntaxError:
        return spans
    cache: Dict[str, re.Pattern] = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and "_" in n.name:
            ln = getattr(n, "lineno", None)
            if not isinstance(ln, int): continue
            lo, hi = line_slice(ln)
            line = code[lo:hi]
            pat = cache.get(n.name)
            if pat is None:
                pat = re.compile(r"(?:^|\s)(?:async\s+def|def)\s+(" + re.escape(n.name) + r")\s*\(")
                cache[n.name] = pat
            m = pat.search(line) or re.compile(r"\b" + re.escape(n.name) + r"\b").search(line)
            if m:
                name_lo = abs0 + lo + m.start(1)
                name_hi = abs0 + lo + m.end(1)
                spans.append((name_lo, name_hi))
    return spans

def _code_line_starts_block(code: str, abs0: int) -> List[Span]:
    starts = _line_starts(code)
    return [(abs0 + s, abs0 + s + 1) for s in starts if abs0 + s < abs0 + len(code)]

def patterns(text: str) -> Dict[str, np.ndarray]:
    N = len(text)
    blocks = _find_code_blocks(text)
    spans_in_comment = []
    spans_newline = []
    spans_if = []
    spans_for = []
    spans_def = []
    spans_indent = []
    spans_defname = []
    spans_codeline = []
    for b_abs, _e_abs, code in blocks:
        spans_codeline += _code_line_starts_block(code, b_abs)
        spans_in_comment += _comment_spans_block(code, b_abs)
        spans_newline += _newline_pair_spans_block(code, b_abs)
        spans_indent += _indent_ws_spans_block(code, b_abs)
        try:
            if_sp, for_sp, def_sp = _if_for_def_full_line_spans_block(code, b_abs)
            spans_if += if_sp; spans_for += for_sp; spans_def += def_sp
            spans_defname += _def_name_uscore_spans_block(code, b_abs)
        except SyntaxError:
            continue
    out: Dict[str, np.ndarray] = {
        "in_comment":      _clip_to_N(_merge(spans_in_comment), N),
        "newline":         _clip_to_N(_merge(spans_newline),   N),
        "if_block":        _clip_to_N(_merge(spans_if),        N),
        "for_block":       _clip_to_N(_merge(spans_for),       N),
        "def_block":       _clip_to_N(_merge(spans_def),       N),
        "indent_ws":       _clip_to_N(_merge(spans_indent),    N),
        "def_name_uscore": _clip_to_N(_merge(spans_defname),   N),
    }
    return out
