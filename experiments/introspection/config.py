"""
Settings for all introspection scripts.

c: the model and dataset are now chosen with two settings, MODEL and DATASET, and everything else is
   derived from the training run in ../results/<MODEL>/<DATASET>/:
     - checkpoint:  checkpoint.pt
     - vocabulary, embedding size, number of heads:  results.json (exactly as used in training)
   Before, paths, builder and sizes were set separately and the vocabulary was rebuilt from the text
   file, which could differ from the training vocabulary and silently mix up characters.

Choose the run with environment variables (PowerShell), e.g.:
    $env:MODEL = "slstm"; $env:DATASET = "copy"
"""
from __future__ import annotations

import json
import os
import sys

import torch


def _env_str(key: str, default: str) -> str:
    val = os.environ.get(key, default)
    return val.strip() if isinstance(val, str) else str(val)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


# n: the experiments folder must be importable (models/, training/), also when started from introspection/
EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

# -------------------- which run to analyse --------------------
MODEL: str = _env_str("MODEL", "mlstm")         # slstm, mlstm, xlstm_sm, xlstm_ms
DATASET: str = _env_str("DATASET", "python")    # shakespeare, python, dyck, copy

# ONE SWITCH: True = trained weights, False = untrained model (same architecture, random weights)
USE_TRAINED: bool = True

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# n: everything about the trained model is read from its training run
RUN_DIR: str = os.path.join(EXPERIMENTS_DIR, "results", MODEL, DATASET)
CKPT_PATH: str = os.path.join(RUN_DIR, "checkpoint.pt")
with open(os.path.join(RUN_DIR, "results.json"), encoding="utf-8") as _f:
    _RUN = json.load(_f)
VOCAB: list = _RUN["vocab"]                       # index -> character, exactly as in training
EMBED_DIM: int = _RUN["config"]["embed_dim"]
NUM_HEADS: int = _RUN["config"]["num_heads"]

SEQ_LEN: int = _env_int("SEQ_LEN", 1000)          # length of the text window that is analysed

MODEL_BUILDER: str = f"models.build_{MODEL}.build_model"
MODEL_TAG: str = MODEL

BASE_RESULTS_DIR: str = _env_str("RESULTS_DIR", "results")
RUN_TAG: str = f'{"untrained_" if not USE_TRAINED else ""}{DATASET}_emb{EMBED_DIM}'
RESULTS_DIR: str = os.path.join(BASE_RESULTS_DIR, MODEL_TAG, RUN_TAG)

PATTERN_PROVIDER: str = _env_str("PATTERN_PROVIDER", "")
if not PATTERN_PROVIDER:
    _auto_map = {
        "shakespeare": "patterns.shakespeare_patterns.patterns",
        "copy": "patterns.copy_patterns.patterns",
        "dyck": "patterns.dyck_patterns.patterns",
        "python": "patterns.python_patterns.patterns",
    }
    PATTERN_PROVIDER = _auto_map.get(DATASET.lower(), "patterns.python_patterns.patterns")

# -------------------- introspection thresholds --------------------
LOGIT_SAT: float = _env_float("LOGIT_SAT", 4.0)   # |pre-activation| above this counts as saturated
GATE_EPS: float = _env_float("GATE_EPS", 0.1)     # gate below eps or above 1 - eps counts as saturated

os.makedirs(RESULTS_DIR, exist_ok=True)


def print_config_summary() -> None:
    print(f"[config] DEVICE={DEVICE}")
    print(f"[config] USE_TRAINED={USE_TRAINED}")
    print(f"[config] MODEL={MODEL}  DATASET={DATASET}")
    print(f"[config] CKPT_PATH={CKPT_PATH}")
    print(f"[config] EMBED_DIM={EMBED_DIM}  NUM_HEADS={NUM_HEADS}  VOCAB={len(VOCAB)}  SEQ_LEN={SEQ_LEN}")
    print(f"[config] RESULTS_DIR={RESULTS_DIR}")
    print(f"[config] PATTERN_PROVIDER={PATTERN_PROVIDER or '(none)'}")
