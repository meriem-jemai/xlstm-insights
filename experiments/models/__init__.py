"""
Builders for the authors' official xLSTM models (library `xlstm`).

    slstm     one sLSTM block
    mlstm     one mLSTM block
    xlstm_sm  two blocks: sLSTM, then mLSTM
    xlstm_ms  two blocks: mLSTM, then sLSTM
"""
from . import build_mlstm, build_slstm, build_xlstm_ms, build_xlstm_sm

# n: one function to build any model by name (used by train, evaluate and generate)
_BUILDERS = {
    "slstm": build_slstm.build_model,
    "mlstm": build_mlstm.build_model,
    "xlstm_sm": build_xlstm_sm.build_model,
    "xlstm_ms": build_xlstm_ms.build_model,
}


def build(name, vocab_size, seq_len, embed_dim, num_heads, backend="vanilla"):
    if name == "mlstm":  # the mLSTM has no backend option (it has no sLSTM inside)
        return _BUILDERS[name](vocab_size, seq_len, embed_dim, num_heads)
    return _BUILDERS[name](vocab_size, seq_len, embed_dim, num_heads, backend=backend)
