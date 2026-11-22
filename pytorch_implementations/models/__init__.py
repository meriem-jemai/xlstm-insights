"""
n: new file. One place to build every model by name (used by train.py and generate.py,
   before the same function was copied into both).
"""
from .lstm import LSTMModel
from .mlstm import mLSTMModel as mLSTM_base                      # state (C, n, m)
from .mlstm_global_hidden import mLSTMModel as mLSTM_global      # state (h, C, n, m)
from .mlstm_separate_hidden import mLSTMModel as mLSTM_separate  # state (h, C, n, m)
from .slstm import sLSTMModel
from .xlstm import xLSTMModel

MLSTM_VARIANTS = {"base": mLSTM_base, "global": mLSTM_global, "separate": mLSTM_separate}


def build_model(name, vocab_size, emb, hid, heads, blocks="s,m", mlstm_variant="base"):
    name = name.lower()
    if name == "lstm":
        return LSTMModel(vocab_size, emb, hid)
    if name == "slstm":
        return sLSTMModel(vocab_size, emb, hid, num_heads=heads)
    if name == "mlstm":
        return MLSTM_VARIANTS[mlstm_variant.lower()](vocab_size, emb, hid, num_heads=heads)
    if name == "xlstm":
        block_list = [b.strip() for b in blocks.split(",")] if isinstance(blocks, str) else list(blocks)
        return xLSTMModel(vocab_size, emb, hid, block_list, num_heads=heads)
    raise ValueError("Unknown model: " + name)


def run_name(model, mlstm_variant="base", blocks="s,m"):
    """Short name used for output files, e.g. 'mlstm_global' or 'xlstm_s-m'."""
    if model == "mlstm":
        return f"mlstm_{mlstm_variant}"
    if model == "xlstm":
        return "xlstm_" + blocks.replace(",", "-")
    return model
