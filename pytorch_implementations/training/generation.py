"""Sample text from a trained model, one character at a time."""
import torch


@torch.no_grad()
def generate_text(model, seed_text, char_to_idx, idx_to_char, length=1000, temperature=0.8):
    model.eval()
    device = next(model.parameters()).device

    if not seed_text:
        raise ValueError("seed_text must be a non-empty string")
    unknown = sorted(set(c for c in seed_text if c not in char_to_idx))
    if unknown:
        raise ValueError(f"seed_text has unknown characters: {''.join(unknown)!r}")

    # read the seed text once to build up the memory
    inp = torch.tensor([[char_to_idx[c] for c in seed_text]], dtype=torch.long, device=device)
    logits, states = model(inp, model.init_hidden(1))

    generated = list(seed_text)
    for _ in range(max(0, length - len(seed_text))):
        # f: sample the next character from the LAST prediction. Before, the last seed character
        #    was fed into the model a second time, so the memory saw it twice.
        probs = torch.softmax(logits[:, -1, :] / max(temperature, 1e-6), dim=-1)  # temperature < 1 = safer choices
        next_idx = torch.multinomial(probs, num_samples=1)  # [1, 1]
        generated.append(idx_to_char[int(next_idx)])
        logits, states = model(next_idx, states)

    return "".join(generated)
