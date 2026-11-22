"""
Gradient check: compare the hand-written BPTT gradients with numerical gradients
computed by central finite differences. Relative errors around 1e-7 or smaller
mean the backward pass is correct.
"""
import numpy as np

import lstm
import slstm


def rel_error(a, b):
    return abs(a - b) / max(1e-8, abs(a) + abs(b))


def check(name, model, params, state, inputs, targets, n_checks=10, eps=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    _, grads, _ = model.forward_backward(inputs, targets, state, params)
    worst = 0.0
    for key, value in params.items():
        for _ in range(n_checks):
            idx = tuple(rng.integers(0, s) for s in value.shape)
            old = value[idx]
            value[idx] = old + eps
            loss_plus, _, _ = model.forward_backward(inputs, targets, state, params)
            value[idx] = old - eps
            loss_minus, _, _ = model.forward_backward(inputs, targets, state, params)
            value[idx] = old
            numeric = (loss_plus - loss_minus) / (2 * eps)
            worst = max(worst, rel_error(numeric, grads[key][idx]))
    status = "PASS" if worst < 1e-5 else "FAIL"
    print(f"{name:20s} max relative error: {worst:.2e}  {status}")
    return worst < 1e-5


def random_state(model, hidden, rng):
    state = model.init_state(hidden)
    for k in state:
        state[k] = rng.standard_normal((hidden, 1)) * 0.5
    if "n" in state:
        state["n"] = np.abs(state["n"]) + 0.5  # normalizer is positive
    return state


def main():
    rng = np.random.default_rng(0)
    vocab, hidden, T = 7, 8, 6
    inputs = list(rng.integers(0, vocab, T))
    targets = list(rng.integers(0, vocab, T))

    # larger weights than in training, so that all nonlinearities are really exercised
    ok = check("LSTM", lstm, lstm.init_params(hidden, vocab, rng, scale=0.5),
               random_state(lstm, hidden, rng), inputs, targets)
    for heads in (1, 2, 4):
        params = slstm.init_params(hidden, vocab, rng, heads=heads, scale=0.5)
        ok &= check(f"sLSTM ({heads} heads)", slstm, params,
                    random_state(slstm, hidden, rng), inputs, targets)
    print("\nAll gradients correct." if ok else "\nSome gradients are wrong.")


if __name__ == "__main__":
    main()
