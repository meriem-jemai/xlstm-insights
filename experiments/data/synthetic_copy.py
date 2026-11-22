import random, string

ALPH = string.ascii_lowercase + string.digits + "     "

def gen_copy(seq_len=32, gap=32, out_len=32):
    src = "".join(random.choice(ALPH) for _ in range(seq_len))
    return src + ("#" * gap) + src[:out_len]

def write_file(path, num_lines=1000, seq_len=32, gap=32, out_len=32):
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(num_lines):
            f.write(gen_copy(seq_len, gap, out_len) + "\n")

if __name__ == "__main__":
    # n: fixed seeds, so the datasets are the same on every computer (different seeds for train and val)
    random.seed(0)
    write_file("data/copy_train.txt", num_lines=10000, seq_len=32, gap=40, out_len=32)
    random.seed(1)
    write_file("data/copy_val.txt",   num_lines=1000,  seq_len=32, gap=40, out_len=32)
    print("Wrote data/copy_train.txt and data/copy_val.txt")
