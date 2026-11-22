"""
n: new file. Character-level datasets for the experiments with the official xLSTM models.

Train / validation split:
    - copy, dyck:          separate files, data/<name>_train.txt and data/<name>_val.txt
    - shakespeare, python: one file, the last 10 % is used for validation

The vocabulary (all different characters) is built from the training AND validation text together,
so every character has the same index in both. Before, evaluate.py built a new vocabulary from the
validation file alone: if one character was missing there, all following indices shifted and the
model was evaluated on wrongly encoded text.
"""
import os

import torch

DATA_DIR = "data"

# dataset name -> (training file, validation file or None = split the training file)
FILES = {
    "shakespeare": ("shakespeare.txt", None),
    "python": ("python_code.txt", None),
    "dyck": ("dyck_train.txt", "dyck_val.txt"),
    "copy": ("copy_train.txt", "copy_val.txt"),
}


def ensure_python_corpus(path, split="train[:10%]"):
    """Downloads Python code from Hugging Face once and saves it as one text file."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    from datasets import load_dataset  # only needed for this dataset
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # c: 10 % instead of 50 % of the dataset; 50 % gives a text of several hundred MB that does not fit in memory
    ds = load_dataset("jtatman/python-code-dataset-500k", split=split)
    # f: the dataset changed: there is no "code" column anymore. The code is now inside the "output"
    #    column, as ```python ... ``` blocks between explanations. Only the code blocks are kept.
    if "code" in ds.column_names:
        snippets = [x for x in ds["code"] if isinstance(x, str)]
    else:
        import re
        fence = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
        snippets = [block for x in ds["output"] if isinstance(x, str) for block in fence.findall(x)]
    if not snippets:
        raise ValueError("No Python code found in the dataset")
    text = "\n\n# ===== FILE =====\n\n".join(snippets)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _read(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return f.read()


def load_dataset_split(name, val_fraction=0.1):
    """
    Returns (train_ids, val_ids, vocab):
        train_ids, val_ids: 1D tensors with one index per character
        vocab:              sorted list of all characters (index -> character)
    """
    train_file, val_file = FILES[name]
    if name == "python":
        ensure_python_corpus(os.path.join(DATA_DIR, train_file))

    if val_file is None:
        # one file: the last 10 % is the validation part
        text = _read(train_file)
        split = int(len(text) * (1 - val_fraction))
        train_text, val_text = text[:split], text[split:]
    else:
        train_text, val_text = _read(train_file), _read(val_file)

    vocab = sorted(set(train_text) | set(val_text))  # f: one vocabulary for train and validation
    stoi = {ch: i for i, ch in enumerate(vocab)}

    def encode(text):
        return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

    return encode(train_text), encode(val_text), vocab


def get_batch(data, batch_size, seq_len, device, generator=None):
    """Random pieces of text; the targets are the same pieces shifted by one character."""
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=generator).tolist()
    x = torch.stack([data[i : i + seq_len] for i in starts]).to(device)
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in starts]).to(device)
    return x, y
