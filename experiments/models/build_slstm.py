"""One sLSTM block, built with the official `xlstm` library."""
from xlstm import (
    FeedForwardConfig,
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
        num_blocks=1,
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                # c: backend is a parameter now. "cuda" = fused CUDA kernels (Linux), "vanilla" = plain PyTorch.
                #    Both compute the same sLSTM, and checkpoints can be loaded with either.
                backend=backend,
                num_heads=num_heads,
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
        ),
        slstm_at=[0],  # place sLSTM at block 0
        mlstm_block=None,
    )
    return xLSTMLMModel(cfg)
