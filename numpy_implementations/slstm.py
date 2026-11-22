"""
sLSTM cell (Beck et al., 2024) with hand-written forward pass and BPTT (NumPy).

- Exponential input and forget gates, stabilized by the state m
- Normalizer state n, so that h = o * c / n
- Multiple heads: the recurrent matrices are block-diagonal,
  so each head only mixes memory within itself
"""
import numpy as np

GATES = ("z", "i", "f", "o")  # cell input, input gate, forget gate, output gate
EPS = 1e-8                    # avoids division by zero in c / n


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(y):
    e = np.exp(y - np.max(y))  # subtract the max for numerical stability
    return e / np.sum(e)


def init_params(hidden, vocab, rng, heads=4, scale=0.01, forget_bias=0.0):
    assert hidden % heads == 0, "hidden size must be divisible by the number of heads"
    dh = hidden // heads  # units per head, e.g. 256 / 4 = 64
    p = {}
    for g in GATES:
        # input weights: every gate sees the full input x
        p[f"W{g}"] = rng.standard_normal((hidden, vocab)) * scale
        # f: recurrent weights per head, shape (heads, dh, dh) instead of (hidden, hidden).
        #    Only the blocks on the diagonal are stored, the zeros between heads are not.
        p[f"R{g}"] = rng.standard_normal((heads, dh, dh)) * scale
        p[f"b{g}"] = np.zeros((hidden, 1))
    # n: optional forget gate bias (0 = same behaviour as before)
    p["bf"] += forget_bias
    # output layer: hidden state -> scores for each character
    p["Why"] = rng.standard_normal((vocab, hidden)) * scale
    p["by"] = np.zeros((vocab, 1))
    return p


def init_state(hidden):
    # h: hidden state, c: cell state, n: normalizer, m: stabilizer
    return {k: np.zeros((hidden, 1)) for k in ("h", "c", "n", "m")}


# n: block-diagonal recurrence (this is what makes the heads independent)
def _recurrent(R, h):
    """Block-diagonal matrix product: each head multiplies its own slice of h."""
    heads, dh, _ = R.shape
    # h.reshape(heads, dh, 1): cut h into pieces, one per head, e.g. (256,1) -> (4,64,1)
    # R @ ...: (4,64,64) @ (4,64,1) does 4 separate products, head i uses only R_i on piece i
    # .reshape(-1, 1): glue the pieces back together, (4,64,1) -> (256,1)
    return (R @ h.reshape(heads, dh, 1)).reshape(-1, 1)


# n: backward pass of the block-diagonal recurrence
def _recurrent_backward(R, d_pre, h_prev):
    """Gradients of _recurrent w.r.t. R and h_prev.

    For y = R @ h the gradients are dR = dy @ h.T and dh = R.T @ dy.
    Here the same rule is applied to every head separately.
    """
    heads, dh, _ = R.shape
    # cut the incoming gradient into pieces, in the same order as h was cut in the forward pass
    d = d_pre.reshape(heads, dh, 1)
    # dR_i = dy_i @ h_i.T for each head -> shape (heads, dh, dh), same as R
    dR = d @ h_prev.reshape(heads, 1, dh)
    # dh_i = R_i.T @ dy_i for each head; transpose(0, 2, 1) transposes every block but keeps the head order.
    # The pieces are glued back with reshape. No summing across heads is needed,
    # because a unit of head i only influenced the outputs of head i.
    dh_prev = (R.transpose(0, 2, 1) @ d).reshape(-1, 1)
    return dR, dh_prev


# c: one time step as a function, shared by training and sampling
def step(x, state, P):
    """One time step. Returns (probabilities, new state, cache for backprop)."""
    h_prev, c_prev, n_prev, m_prev = state["h"], state["c"], state["n"], state["m"]

    # pre-activations of all 4 gates: input part + recurrent part (per head) + bias
    pre = {g: P[f"W{g}"] @ x + _recurrent(P[f"R{g}"], h_prev) + P[f"b{g}"] for g in GATES}

    z = np.tanh(pre["z"])  # cell input, between -1 and 1
    o = sigmoid(pre["o"])  # output gate, between 0 and 1

    # Exponential gates can become huge numbers (overflow). The stabilizer m keeps them small:
    # it is the largest exponent seen so far, and every exponent is shifted by it.
    m = np.maximum(pre["f"] + m_prev, pre["i"])  # stabilizer
    i = np.exp(pre["i"] - m)                     # stabilized input gate
    f = np.exp(pre["f"] + m_prev - m)            # stabilized forget gate

    c = f * c_prev + i * z     # cell state: keep part of the old memory, add new content
    n = f * n_prev + i         # normalizer: sums up the gate weights
    h = o * c / (n + EPS)      # hidden state: normalized memory, scaled by the output gate
    p = softmax(P["Why"] @ h + P["by"])  # probabilities for the next character

    cache = dict(x=x, h_prev=h_prev, c_prev=c_prev, n_prev=n_prev, z=z, i=i, f=f, o=o, c=c, n=n, h=h, p=p)
    return p, {"h": h, "c": c, "n": n, "m": m}, cache


def forward_backward(inputs, targets, state, P):
    """Forward over a sequence, then BPTT. Returns (loss, grads, final state).

    The stabilizer m is treated as a constant: c and n are both scaled by exp(-m),
    so h = o * c / n does not depend on m and its gradient contribution is zero.
    """
    V = P["by"].shape[0]

    # ---------- forward: run through the sequence and remember everything ----------
    caches, loss = [], 0.0
    for ix, target in zip(inputs, targets):
        x = np.zeros((V, 1)); x[ix] = 1        # one-hot encoding of the current character
        p, state, cache = step(x, state, P)
        loss -= np.log(p[target, 0])           # cross-entropy: penalize low probability of the right character
        caches.append(cache)

    # ---------- backward: go through the sequence in reverse (BPTT) ----------
    grads = {k: np.zeros_like(v) for k, v in P.items()}
    dh_next = np.zeros_like(state["h"])  # gradient flowing back from the next time step
    dc_next = np.zeros_like(state["c"])
    dn_next = np.zeros_like(state["n"])
    for cache, target in zip(reversed(caches), reversed(targets)):
        # softmax + cross-entropy: gradient is (probabilities - one-hot target)
        dy = cache["p"].copy(); dy[target] -= 1
        grads["Why"] += dy @ cache["h"].T
        grads["by"] += dy

        # gradient into h: from the output layer + from the next time step
        dh = P["Why"].T @ dy + dh_next
        z, i, f, o, c, n = (cache[k] for k in ("z", "i", "f", "o", "c", "n"))
        inv_n = 1.0 / (n + EPS)

        # h = o * c / n  ->  derivatives w.r.t. c and n (plus what flows back from t+1)
        dc = dh * o * inv_n + dc_next
        dn = -dh * o * c * inv_n ** 2 + dn_next

        # gradients w.r.t. the pre-activations of the 4 gates
        d_pre = {
            "z": dc * i * (1 - z ** 2),                               # tanh' = 1 - tanh^2
            "i": (dc * z + dn) * i,                                   # i appears in c and n; d exp(x)/dx = exp(x)
            "f": (dc * cache["c_prev"] + dn * cache["n_prev"]) * f,   # f appears in c and n
            "o": dh * c * inv_n * o * (1 - o),                        # sigmoid' = s * (1 - s)
        }

        dh_next = np.zeros_like(dh)
        for g in GATES:
            grads[f"W{g}"] += d_pre[g] @ cache["x"].T
            # f: gradients per head instead of one full matrix
            dR, dh_prev = _recurrent_backward(P[f"R{g}"], d_pre[g], cache["h_prev"])
            grads[f"R{g}"] += dR
            grads[f"b{g}"] += d_pre[g]
            dh_next += dh_prev  # every gate sends gradient back to h_prev

        # c and n are carried to the next step through f, so their gradients flow back through f
        dc_next = dc * f
        dn_next = dn * f

    return loss, grads, state
