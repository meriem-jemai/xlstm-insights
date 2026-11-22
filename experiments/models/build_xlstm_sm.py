"""xLSTM with two blocks: sLSTM first, then mLSTM (official `xlstm` library)."""
from xlstm import (
    FeedForwardConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMLMModel,
    xLSTMLMModelConfig,
)


def build_model(vocab_size: int, seq_len: int, embed_dim: int = 256, num_heads: int = 4, backend: str = "vanilla"):
    cfg = xLSTMLMModelConfig(
        vocab_size=vocab_size,
        context_length=seq_len,
        embedding_dim=embed_dim,
        num_blocks=2,  # two blocks: one sLSTM, one mLSTM
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend=backend,  # c: "cuda" (Linux) or "vanilla" (plain PyTorch), same math
                num_heads=num_heads,
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
        ),
        slstm_at=[0],  # sLSTM at block 0, so the order is sLSTM -> mLSTM
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4,
                qkv_proj_blocksize=4,
                num_heads=num_heads,
            )
        ),
    )
    return xLSTMLMModel(cfg)
