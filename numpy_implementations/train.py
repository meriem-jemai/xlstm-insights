"""
Train the NumPy LSTM or sLSTM on Shakespeare (character level).

Examples:
    python train.py --model lstm
    python train.py --model slstm --heads 4
    python train.py --model slstm --heads 4 --wandb

"""
import argparse
import json
import os
import time

import numpy as np

import lstm
import slstm

MODELS = {"lstm": lstm, "slstm": slstm}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, default="lstm")
    p.add_argument("--heads", type=int, default=4, help="sLSTM only")
    p.add_argument("--forget-bias", type=float, default=0.0, help="sLSTM only: initial forget gate bias")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=25)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--max-chars", type=int, default=None, help="use only the first N characters (quick tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="shakespeare.txt")
    p.add_argument("--out", default="runs")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    return p.parse_args()


def sample(model, params, state, seed_ix, length, rng):
    V = params["by"].shape[0]
    ix, out = seed_ix, []
    for _ in range(length):
        x = np.zeros((V, 1)); x[ix] = 1
        p, state, _ = model.step(x, state, params)
        ix = rng.choice(V, p=p.ravel())
        out.append(ix)
    return out


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    model = MODELS[args.model]
    name = args.model if args.model == "lstm" else f"slstm_{args.heads}heads"
    if args.hidden != 256:
        name += f"_h{args.hidden}"  # n: hidden size in the name, so runs with other sizes don't overwrite each other

    with open(args.data, encoding="utf-8") as f:
        data = f.read()
    if args.max_chars:
        data = data[: args.max_chars]
    chars = sorted(set(data))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    V, T = len(chars), args.seq_len
    print(f"{len(data)} characters, {V} unique")

    if args.model == "slstm":
        params = model.init_params(args.hidden, V, rng, heads=args.heads, forget_bias=args.forget_bias)
    else:
        params = model.init_params(args.hidden, V, rng)
    print(f"{name}: {sum(v.size for v in params.values()):,} parameters")
    adagrad = {k: np.zeros_like(v) for k, v in params.items()}

    if args.wandb:
        import wandb  
        wandb.init(project="xlstm-numpy", name=name, config=vars(args))

    n_batches = (len(data) - 1) // T
    history = []
    for epoch in range(1, args.epochs + 1):
        start, total_loss = time.perf_counter(), 0.0
        state = model.init_state(args.hidden)
        for b in range(n_batches):
            chunk = data[b * T : b * T + T + 1]
            inputs = [char_to_ix[ch] for ch in chunk[:-1]]
            targets = [char_to_ix[ch] for ch in chunk[1:]]
            loss, grads, state = model.forward_backward(inputs, targets, state, params)
            total_loss += loss
            for k in params:  # Adagrad with element-wise gradient clipping (CHANGED: clipping moved here)
                g = np.clip(grads[k], -args.clip, args.clip)
                adagrad[k] += g * g
                params[k] -= args.lr * g / (np.sqrt(adagrad[k]) + 1e-8)

        seconds = time.perf_counter() - start
        avg = total_loss / (n_batches * T)
        row = dict(epoch=epoch, loss=avg, perplexity=float(np.exp(avg)),
                   bits_per_char=avg / np.log(2), tokens_per_s=n_batches * T / seconds)
        history.append(row)
        print(f"epoch {epoch:2d} | loss {avg:.4f} | ppl {row['perplexity']:.2f} | "
              f"bpc {row['bits_per_char']:.3f} | {row['tokens_per_s']:.0f} tok/s")
        if args.wandb:
            wandb.log(row)

    text = "".join(chars[i] for i in sample(model, params, state, char_to_ix["\n"], 1000, rng))
    print("\n--- sample ---\n" + text[:500])

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.out, f"{name}_sample.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

