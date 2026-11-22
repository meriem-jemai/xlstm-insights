"""
Training loop and evaluation for the PyTorch models.

How training works:
    1. draw a random batch of text pieces (inputs) and the same pieces shifted by one character (targets)
    2. the model predicts the next character at every position -> cross-entropy loss
    3. backpropagation computes the gradients, gradient clipping limits their size
    4. Adam updates the weights
    One epoch = roughly one pass over the training text. After every epoch the model is
    evaluated on the validation text, which it never trains on.
"""
import math

import torch
import torch.nn as nn


def _move_states(states, device):
    # states can be a tensor, a tuple of tensors, or a list of tuples (xLSTM: one tuple per block)
    if torch.is_tensor(states):
        return states.to(device)
    if isinstance(states, (tuple, list)):
        return type(states)(_move_states(s, device) for s in states)
    return states


@torch.no_grad()  # no gradients needed for evaluation: faster and uses less memory
def evaluate(model, data, config, get_batch_fn, steps=200, generator=None):
    """
    Average loss, perplexity, bits per character and accuracy on random batches of `data`.

    - loss:       average cross-entropy per character (lower = better)
    - perplexity: exp(loss), "how many characters the model is choosing between" (lower = better)
    - bpc:        loss in bits instead of nats, loss / ln(2)
    - accuracy:   how often the most likely character is the right one
    """
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(reduction="sum")  # sum, so we can divide by the exact token count
    model.eval()

    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for _ in range(max(1, steps)):
        inputs, targets = get_batch_fn(data, config["batch_size"], config["seq_length"],
                                       device=device, generator=generator)
        states = _move_states(model.init_hidden(inputs.size(0)), device)  # every piece starts with empty memory
        logits, _ = model(inputs, states)
        V = logits.size(-1)
        total_loss += criterion(logits.reshape(-1, V), targets.reshape(-1)).item()
        total_correct += (logits.argmax(dim=-1) == targets).sum().item()
        total_tokens += targets.numel()

    model.train()
    loss = total_loss / total_tokens
    return {"loss": loss, "ppl": math.exp(loss), "bpc": loss / math.log(2), "accuracy": total_correct / total_tokens}


def train_model(model, train_data, val_data, config, get_batch_fn, log_fn=None, generator=None):
    """
    Trains with Adam and gradient clipping. Returns the history (one row per epoch).

    c: after every epoch the model is evaluated on the VALIDATION set (before: only once at the end,
       on the training data). log_fn is optional, e.g. wandb.log (before: W&B was required).
    """
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    # one epoch = roughly one pass over the training text (or a fixed number of steps for quick tests)
    steps_per_epoch = config.get("steps_per_epoch") or max(1, len(train_data) // (config["batch_size"] * config["seq_length"]))

    history = []
    for epoch in range(1, config["num_epochs"] + 1):
        model.train()
        total = 0.0
        for _ in range(steps_per_epoch):
            # 1) random text pieces; targets = inputs shifted by one character
            inputs, targets = get_batch_fn(train_data, config["batch_size"], config["seq_length"],
                                           device=device, generator=generator)
            states = _move_states(model.init_hidden(inputs.size(0)), device)  # empty memory at the start

            # 2) predict every next character and measure how wrong the predictions are
            optimizer.zero_grad(set_to_none=True)  # forget the gradients of the previous step
            logits, _ = model(inputs, states)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            # 3) backpropagation: gradients of the loss w.r.t. every weight
            loss.backward()
            # n: if anything became NaN/inf, save this exact batch and the weights (before the update)
            #    to nan_debug.pt and stop, so the cause can be found with debug_nan.py
            bad = not torch.isfinite(loss) or any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters())
            if bad:
                torch.save({"model": model.state_dict(), "inputs": inputs.cpu(), "targets": targets.cpu(),
                            "config": config}, "nan_debug.pt")
                raise RuntimeError(f"NaN/inf in epoch {epoch}, batch saved to nan_debug.pt")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # avoids exploding gradients

            # 4) update the weights
            optimizer.step()
            total += loss.item()

        # after every epoch: check on the validation text, which the model never trains on
        val = evaluate(model, val_data, config, get_batch_fn, steps=config.get("eval_steps", 50), generator=generator)
        row = {"epoch": epoch, "train_loss": total / steps_per_epoch,
               **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(f"epoch {epoch:2d} | train loss {row['train_loss']:.4f} | val loss {val['loss']:.4f} | "
              f"val ppl {val['ppl']:.3f} | val acc {val['accuracy']:.2%}")
        if log_fn:
            log_fn(row)

    return history
