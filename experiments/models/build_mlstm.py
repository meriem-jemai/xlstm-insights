"""One mLSTM block, built with the official `xlstm` library."""
from xlstm import (
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    xLSTMLMModel,
    xLSTMLMModelConfig,
)


# c: default embedding size 256 like the other models (was 1024)
def build_model(vocab_size: int, seq_len: int, embed_dim: int = 256, num_heads: int = 4):
    cfg = xLSTMLMModelConfig(
        vocab_size=vocab_size,
        context_length=seq_len,
        embedding_dim=embed_dim,
        num_blocks=1,
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4,
                qkv_proj_blocksize=4,
                num_heads=num_heads,
            )
        ),
        slstm_block=None,
    )
    return xLSTMLMModel(cfg)
