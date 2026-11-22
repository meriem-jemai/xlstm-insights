"""
Loads the text window that the introspection scripts analyse.

c: the text now comes from the VALIDATION part (text the model never trained on), and the characters are
   encoded with the vocabulary saved during training (config.VOCAB). Before, the vocabulary was rebuilt
   from the text file, and the window was taken from the start of the training text.
"""
import os

import torch

from config import DATASET, EXPERIMENTS_DIR, SEQ_LEN, VOCAB
from training.data import DATA_DIR, FILES, ensure_python_corpus


def _validation_text():
    train_file, val_file = FILES[DATASET]
    if val_file is not None:
        path = os.path.join(EXPERIMENTS_DIR, DATA_DIR, val_file)
        with open(path, encoding="utf-8") as f:
            return f.read()
    if DATASET == "python":  # n: download the Python code dataset if it is not there yet (same as in training)
        ensure_python_corpus(os.path.join(EXPERIMENTS_DIR, DATA_DIR, train_file))
    # one file: the last 10 % is the validation part (same split as in training)
    with open(os.path.join(EXPERIMENTS_DIR, DATA_DIR, train_file), encoding="utf-8") as f:
        text = f.read()
    return text[int(len(text) * 0.9):]


def load_data():
    """Returns (text, ids, stoi): the analysed text window, its indices [1, SEQ_LEN] and the vocabulary map."""
    stoi = {ch: i for i, ch in enumerate(VOCAB)}
    text = _validation_text()[:SEQ_LEN]
    ids = torch.tensor([[stoi[ch] for ch in text]], dtype=torch.long)
    return text, ids, stoi

