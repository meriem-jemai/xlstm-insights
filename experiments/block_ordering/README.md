# Does the order of the blocks matter?

An xLSTM is a stack of sLSTM and mLSTM blocks. This experiment tests whether the **order** of the two
block types changes what the model can learn, on the **Cycle Navigation** task.

**Cycle Navigation** ([Delétang et al., 2023](https://arxiv.org/abs/2207.02098)): the input is a sequence of
moves (forward, backward, stay) on a cycle of positions, and the model must output the final position.
Solving it requires keeping track of a state over the whole sequence.

## Configurations

All runs use the authors' training and dataset settings: two blocks, embedding size 64, one head,
20,000 steps, batch size 256, sequences of 64–192 moves (validation: 40–256), AdamW with warm-up and decay,
bfloat16 mixed precision.

| File | Blocks | `slstm_at` |
|---|---|---|
| `mlstm_only.yaml` | mLSTM, mLSTM | `[]` |
| `slstm_only.yaml` | sLSTM, sLSTM | `[0, 1]` |
| `slstm_then_mlstm.yaml` | sLSTM → mLSTM | `[0]` |
| `mlstm_then_slstm.yaml` | mLSTM → sLSTM | `[1]` |

## Run

The configurations are made for the experiment runner of the official repository:

```bash
git clone https://github.com/NX-AI/xlstm.git
cd xlstm
pip install -e .
python experiments/main.py --config /path/to/mlstm_then_slstm.yaml
```

The sLSTM uses the fused CUDA kernels in bfloat16, so a recent NVIDIA GPU (e.g. A100) is needed.

## Results from the thesis

Validation after 20,000 steps, **one run each** (seed 42), from the original thesis runs
(Section 7 of the [thesis](../../docs/thesis.pdf)):

| Model | Validation loss | Validation accuracy |
|---|---|---|
| mLSTM only | 0.138 | **96.4%** |
| sLSTM only | 1.794 | 36.3% |
| **mLSTM → sLSTM** | **0.390** | **90.8%** |
| sLSTM → mLSTM | 3.375 | 7.0% |

With the same two blocks, **mLSTM → sLSTM** learns the task, while **sLSTM → mLSTM** fails completely.
On this task the mLSTM alone is strongest, and the order decides whether the combined model can make use of it.

For comparison, on the **Parity** task (configurations from the official repository) the picture is reversed:
the sLSTM alone reaches 100% sequence accuracy, the mLSTM alone 49.1%, matching the paper's claim
that sLSTM is needed for state tracking.

> These are single runs. Repeating them with several seeds is planned, to check how stable the large
> difference between the two orders is.
