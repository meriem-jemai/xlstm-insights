import random

def gen_dyck(n, depth=3):
    # Simple Dyck-1 generator with random nesting up to 'depth'
    s = []
    cur = 0
    for _ in range(n):
        if cur == 0 or (cur < depth and random.random() < 0.6):
            s.append("("); cur += 1
        else:
            s.append(")"); cur = max(0, cur-1)
    # Close remaining
    s += [")"] * cur
    return "".join(s)

def write_file(path, num_lines=1000, length=120, depth=3):
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(num_lines):
            f.write(gen_dyck(length, depth) + "\n")

if __name__ == "__main__":
    # n: fixed seeds, so the datasets are the same on every computer (different seeds for train and val)
    random.seed(0)
    write_file("data/dyck_train.txt", num_lines=10000, length=120, depth=5)
    random.seed(1)
    write_file("data/dyck_val.txt",   num_lines=1000,  length=120, depth=5)
    print("Wrote data/dyck_train.txt and data/dyck_val.txt")
