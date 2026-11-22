Set-Location (Join-Path $PSScriptRoot "..")  # c: run from pytorch_implementations
# n: runs all PyTorch experiments of the thesis one after the other
$py = "..\.venv\Scripts\python"

# Table 5: model comparison on Shakespeare
& $py train.py --model lstm
& $py train.py --model slstm
& $py train.py --model mlstm --mlstm_variant base
& $py train.py --model xlstm --blocks s,m

# Table 21: mLSTM variants (own idea) on three datasets
foreach ($data in "shakespeare.txt", "copy_train.txt", "dyck_train.txt") {
    foreach ($variant in "base", "global", "separate") {
        if ($data -eq "shakespeare.txt" -and $variant -eq "base") { continue }  # already trained above
        & $py train.py --model mlstm --mlstm_variant $variant --data $data
    }
}
