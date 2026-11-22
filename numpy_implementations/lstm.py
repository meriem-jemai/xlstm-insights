"""LSTM cell with hand-written forward pass and backpropagation through time (NumPy).
"""
import numpy as np

GATES = ("z", "i", "f", "o")  # cell input, input gate, forget gate, output gate


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(y):
    e = np.exp(y - np.max(y))
    return e / np.sum(e)


def init_params(hidden, vocab, rng, scale=0.01):
    p = {}
    for g in GATES:
        p[f"W{g}"] = rng.standard_normal((hidden, vocab)) * scale   # input weights
        p[f"R{g}"] = rng.standard_normal((hidden, hidden)) * scale  # recurrent weights
        p[f"b{g}"] = np.zeros((hidden, 1))
    p["Why"] = rng.standard_normal((vocab, hidden)) * scale
    p["by"] = np.zeros((vocab, 1))
    return p


def init_state(hidden):
    return {"h": np.zeros((hidden, 1)), "c": np.zeros((hidden, 1))}


def step(x, state, P):
    """One time step. Returns (probabilities, new state, cache for backprop)."""
    h_prev, c_prev = state["h"], state["c"]
    z = np.tanh(P["Wz"] @ x + P["Rz"] @ h_prev + P["bz"])
    i = sigmoid(P["Wi"] @ x + P["Ri"] @ h_prev + P["bi"])
    f = sigmoid(P["Wf"] @ x + P["Rf"] @ h_prev + P["bf"])
    o = sigmoid(P["Wo"] @ x + P["Ro"] @ h_prev + P["bo"])
    c = f * c_prev + i * z
    tanh_c = np.tanh(c)
    h = o * tanh_c
    p = softmax(P["Why"] @ h + P["by"])
    cache = dict(x=x, h_prev=h_prev, c_prev=c_prev, z=z, i=i, f=f, o=o, tanh_c=tanh_c, h=h, p=p)
    return p, {"h": h, "c": c}, cache


def forward_backward(inputs, targets, state, P):
    """Forward over a sequence, then BPTT. Returns (loss, grads, final state)."""
    V = P["by"].shape[0]
    caches, loss = [], 0.0
    for ix, target in zip(inputs, targets):
        x = np.zeros((V, 1)); x[ix] = 1
        p, state, cache = step(x, state, P)
        loss -= np.log(p[target, 0])
        caches.append(cache)

    grads = {k: np.zeros_like(v) for k, v in P.items()}
    dh_next = np.zeros_like(state["h"])
    dc_next = np.zeros_like(state["c"])
    for cache, target in zip(reversed(caches), reversed(targets)):
        dy = cache["p"].copy(); dy[target] -= 1
        grads["Why"] += dy @ cache["h"].T
        grads["by"] += dy

        dh = P["Why"].T @ dy + dh_next
        dc = dh * cache["o"] * (1 - cache["tanh_c"] ** 2) + dc_next

        z, i, f, o = cache["z"], cache["i"], cache["f"], cache["o"]
        d_pre = {
            "z": dc * i * (1 - z ** 2),
            "i": dc * z * i * (1 - i),
            "f": dc * cache["c_prev"] * f * (1 - f),
            "o": dh * cache["tanh_c"] * o * (1 - o),
        }
        dh_next = np.zeros_like(dh)
        for g in GATES:
            grads[f"W{g}"] += d_pre[g] @ cache["x"].T
            grads[f"R{g}"] += d_pre[g] @ cache["h_prev"].T
            grads[f"b{g}"] += d_pre[g]
            dh_next += P[f"R{g}"].T @ d_pre[g]
        dc_next = dc * f

    return loss, grads, state
