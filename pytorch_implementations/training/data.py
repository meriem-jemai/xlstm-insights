"""Character-level data loading for the PyTorch models."""
import torch


def load_text(path):
    # the text is lower-cased to keep the vocabulary small
    with open(path, "r", encoding="utf-8") as f:
        return f.read().lower()


def create_vocab(text):
    vocab = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return vocab, char_to_idx, idx_to_char


def encode_text(text, char_to_idx):
    return torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)


def get_batch(data, batch_size, seq_length, device=None, generator=None):
    """Random batch of sequences; the targets are the inputs shifted by one character."""
    if device is None:
        device = data.device
    assert len(data) > seq_length + 1, "Dataset too small for given seq_length"
    # c: torch random generator instead of numpy, so a seed makes runs reproducible
    starts = torch.randint(0, len(data) - seq_length - 1, (batch_size,), generator=generator).tolist()
    inputs = torch.stack([data[i : i + seq_length] for i in starts]).to(device)
    targets = torch.stack([data[i + 1 : i + seq_length + 1] for i in starts]).to(device)
    return inputs, targets


def prepare_dataset(path, device=None, val_fraction=0.1):
    """
    Returns (train_data, val_data, vocab_size, char_to_idx, idx_to_char).

    n: the last 10 % of the text is kept as a validation set. Before, the final metrics
       were computed on the same text the model was trained on.
    """
    text = load_text(path)
    vocab, char_to_idx, idx_to_char = create_vocab(text)
    data = encode_text(text, char_to_idx)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    split = int(len(data) * (1 - val_fraction))
    return data[:split].to(device), data[split:].to(device), len(vocab), char_to_idx, idx_to_char
