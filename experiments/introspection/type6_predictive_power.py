import os
from typing import Optional, List, Dict
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config import RESULTS_DIR, PATTERN_PROVIDER
from data_utils import load_data
from hooks import extract_series_per_head
from config import RESULTS_DIR, PATTERN_PROVIDER
from type3_alignment import _load_patterns_from_provider, _spans_to_events, _auto_policy
OUTDIR = os.path.join(RESULTS_DIR, 'gates_type6_predictive_power')

@torch.no_grad()
def _forward_logits(model: torch.nn.Module, inp_ids: torch.Tensor) -> torch.Tensor:
    out = model(inp_ids)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)
    return logits

def _accuracy(logits: torch.Tensor, tgt: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == tgt).float().mean().item())

def _nll(logits: torch.Tensor, tgt: torch.Tensor) -> float:
    B, S, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B * S, V), tgt.reshape(B * S), reduction='mean')
    return float(loss.item())

def _events(text: Optional[str], S: int, select: Optional[List[str]]=None) -> Dict[str, np.ndarray]:
    if not PATTERN_PROVIDER:
        return {}
    if text is None:
        text = load_data()[0]  # c: the analysed validation window (before: the start of the training file)
    spans = _load_patterns_from_provider(text[:S], S)
    if select:
        want = {s.lower() for s in select}
        spans = {k: v for k, v in spans.items() if k.lower() in want}
    return {name: _spans_to_events(v, _auto_policy(v)) for name, v in spans.items()}

@torch.no_grad()
def run(model: torch.nn.Module, inp_ids: torch.Tensor, tgt_ids: torch.Tensor, text: Optional[str]=None, mark_patterns: Optional[List[str]]=None):
    print('[type6_predictive_power] start')
    os.makedirs(OUTDIR, exist_ok=True)
    device = next(model.parameters()).device
    inp_ids = inp_ids.to(device)
    tgt_ids = tgt_ids.to(device)
    logits = _forward_logits(model, inp_ids)
    if logits.size(0) != 1:
        logits = logits.reshape(1, -1, logits.size(-1))
        tgt_ids = tgt_ids.reshape(1, -1)
    nll = _nll(logits, tgt_ids)
    ppl = float(np.exp(nll))
    acc = _accuracy(logits, tgt_ids)
    with open(os.path.join(OUTDIR, 'type6_performance.csv'), 'w', encoding='utf-8') as fh:
        fh.write('metric,value\n')
        fh.write(f'nll,{nll}\n')
        fh.write(f'ppl,{ppl}\n')
        fh.write(f'accuracy,{acc}\n')
    S = logits.size(1)
    ev = _events(text, S, mark_patterns)
    logp = logits.log_softmax(dim=-1)
    tgt = tgt_ids[0].cpu().numpy()
    tok_lp = logp[0, np.arange(S), tgt].cpu().numpy()
    surprisal = -tok_lp
    i_all, f_all, i_ph, f_ph, _ = extract_series_per_head()
    i_all = i_all[:S]
    f_all = f_all[:S]
    i_ph = i_ph[:S]
    f_ph = f_ph[:S]
    corr_i = float(np.corrcoef(i_all, surprisal)[0, 1]) if S > 1 else np.nan
    corr_f = float(np.corrcoef(f_all, surprisal)[0, 1]) if S > 1 else np.nan
    with open(os.path.join(OUTDIR, 'type6_gate_surprisal.csv'), 'w', encoding='utf-8') as fh:
        fh.write('metric,value\n')
        fh.write(f'corr_i_surprisal,{corr_i}\n')
        fh.write(f'corr_f_surprisal,{corr_f}\n')
    LAGS = [0, 1, 2]
    with open(os.path.join(OUTDIR, 'type6_gate_surprisal_lags.csv'), 'w', encoding='utf-8') as fh:
        fh.write('lag,corr_i,corr_f,n_pairs\n')
        for k in LAGS:
            if k <= 0:
                xi, xf, ys = (i_all, f_all, surprisal)
            else:
                if S - k < 2:
                    fh.write(f'{k},,,0\n')
                    continue
                xi = i_all[:-k]
                xf = f_all[:-k]
                ys = surprisal[k:]
            r_i = float(np.corrcoef(xi, ys)[0, 1]) if xi.size > 1 else np.nan
            r_f = float(np.corrcoef(xf, ys)[0, 1]) if xf.size > 1 else np.nan
            fh.write(f'{k},{r_i},{r_f},{ys.size}\n')

    def _scatter_with_fit(x, y, title, path):
        x = x.astype(float)
        y = y.astype(float)
        A = np.vstack([x, np.ones_like(x)]).T
        w, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        yfit = w[0] * x + w[1]
        order = np.argsort(x)
        plt.figure(figsize=(5.2, 4.2))
        plt.scatter(x, y, s=6, alpha=0.4)
        plt.plot(x[order], yfit[order], linewidth=2)
        plt.title(title)
        plt.xlabel('Gate value')
        plt.ylabel('Surprisal (-log p)')
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
    _scatter_with_fit(i_all, surprisal, f'Input gate vs surprisal (r={corr_i:.2f})', os.path.join(OUTDIR, 'gate_vs_surprisal_scatter_i.png'))
    _scatter_with_fit(f_all, surprisal, f'Forget gate vs surprisal (r={corr_f:.2f})', os.path.join(OUTDIR, 'gate_vs_surprisal_scatter_f.png'))
    if ev:
        outp = os.path.join(OUTDIR, 'type6_gate_surprisal_by_pattern.csv')
        with open(outp, 'w', encoding='utf-8') as fh:
            fh.write('pattern,lag,n_pairs,corr_i,corr_f\n')
            for name, idx in ev.items():
                idx = np.asarray(idx, int)
                idx = idx[(idx >= 0) & (idx < S)]
                if idx.size == 0:
                    for k in LAGS:
                        fh.write(f'{name},{k},0,,\n')
                    continue
                for k in LAGS:
                    if k <= 0:
                        valid = idx
                        y_idx = idx
                    else:
                        valid = idx[idx + k < S]
                        y_idx = valid + k
                    if valid.size < 2:
                        fh.write(f'{name},{k},{valid.size},,\n')
                        continue
                    r_i = float(np.corrcoef(i_all[valid], surprisal[y_idx])[0, 1])
                    r_f = float(np.corrcoef(f_all[valid], surprisal[y_idx])[0, 1])
                    fh.write(f'{name},{k},{valid.size},{r_i},{r_f}\n')
    if ev:
        csvp = os.path.join(OUTDIR, 'type6_performance_by_pattern.csv')
        with open(csvp, 'w', encoding='utf-8') as fh:
            fh.write('pattern,n_events,nll_at_events,ppl_at_events,acc_at_events,nll_at_nonevents,ppl_at_nonevents,acc_at_nonevents\n')
            B, S, V = logits.shape
            tok_acc = (logits[0].argmax(dim=-1).cpu().numpy() == tgt).astype(np.float32)
            for name, idx in ev.items():
                idx = np.asarray(idx, int)
                idx = idx[(idx >= 0) & (idx < S)]
                if idx.size == 0:
                    fh.write(','.join(map(str, [name, 0] + [np.nan] * 6)) + '\n')
                    continue
                mask = np.zeros(S, dtype=bool)
                mask[idx] = True
                non = np.where(~mask)[0]
                nll_e = -float(tok_lp[idx].mean())
                ppl_e = float(np.exp(-tok_lp[idx].mean()))
                acc_e = float(tok_acc[idx].mean())
                if non.size:
                    nll_ne = -float(tok_lp[non].mean())
                    ppl_ne = float(np.exp(-tok_lp[non].mean()))
                    acc_ne = float(tok_acc[non].mean())
                else:
                    nll_ne = ppl_ne = acc_ne = np.nan
                fh.write(','.join(map(str, [name, int(idx.size), nll_e, ppl_e, acc_e, nll_ne, ppl_ne, acc_ne])) + '\n')
        auc_path = os.path.join(OUTDIR, 'type6_probe_auc_by_pattern.csv')
        with open(auc_path, 'w', encoding='utf-8') as fa:
            fa.write('pattern,auc,positives,S,features\n')
        X = np.concatenate([i_all[:, None], f_all[:, None], i_ph[:S], f_ph[:S]], axis=1)
        Xt = torch.from_numpy(X).float()
        ones = torch.ones(S, 1)

        def _auc(y_true, scores):
            y = y_true.astype(np.int64)
            if y.sum() == 0 or y.sum() == y.size:
                return np.nan
            order = np.argsort(scores)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, y.size + 1)
            n1 = y.sum()
            n0 = y.size - n1
            u = ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0
            return float(u / (n0 * n1 + 1e-12))
        auc_rows = []
        for name, idx in ev.items():
            idx = np.asarray(idx, int)
            idx = idx[(idx >= 0) & (idx < S)]
            mask = np.zeros(S, dtype=np.float32)
            mask[idx] = 1.0
            if mask.sum() == 0:
                auc_rows.append((name, np.nan, 0))
                with open(auc_path, 'a', encoding='utf-8') as fa:
                    fa.write(f'{name},,0,{S},{X.shape[1]}\n')
                continue
            y_t = torch.from_numpy(mask)[:, None]
            Xtb = torch.cat([Xt, ones], dim=1)
            w = torch.zeros(Xtb.size(1), 1, requires_grad=True)
            opt = torch.optim.LBFGS([w], max_iter=60, line_search_fn='strong_wolfe')

            def closure():
                opt.zero_grad()
                logits_local = Xtb @ w
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_local, y_t)
                loss.backward()
                return loss
            try:
                opt.step(closure)
                with torch.no_grad():
                    p = torch.sigmoid(Xtb @ w).squeeze().cpu().numpy()
                auc = _auc(mask, p)
            except Exception:
                auc = np.nan
            auc_rows.append((name, auc, int(mask.sum())))
            with open(auc_path, 'a', encoding='utf-8') as fa:
                fa.write(f'{name},{auc},{int(mask.sum())},{S},{X.shape[1]}\n')
        names = [r[0] for r in auc_rows]
        aucs = [0.0 if r[1] is None or np.isnan(r[1]) else r[1] for r in auc_rows]
        order = np.argsort(aucs)[::-1]
        plt.figure(figsize=(max(6, 0.35 * len(names)), 4.2))
        plt.bar(np.arange(len(names)), np.array(aucs)[order])
        plt.xticks(np.arange(len(names)), np.array(names)[order], rotation=45, ha='right')
        plt.ylabel('AUC')
        plt.ylim(0.5, 1.0)
        plt.title('Gate-based linear probe AUC by pattern')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, 'probe_auc_bars.png'), dpi=200)
        plt.close()
    print('[type6_predictive_power] wrote', OUTDIR)