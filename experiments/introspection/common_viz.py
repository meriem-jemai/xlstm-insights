import os, numpy as np, matplotlib.pyplot as plt

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def save_heatmap(
    mat,
    title,
    outpng,
    xlabel="Time step",
    ylabel="Head",
    extent=None,
    vmin=None,
    vmax=None,
):
    plt.figure()
    mat = np.asarray(mat) if mat is not None else np.zeros((1, 1))
    if extent is not None:
        plt.imshow(mat, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
    else:
        plt.imshow(mat, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng)
    plt.close()

def save_lines(
    lines,
    labels,
    title,
    outpng,
    xlabel="Time step",
    ylabel="value",
):
    plt.figure()
    for y, lab in zip(lines, labels):
        try:
            y = np.asarray(y)
        except Exception:
            pass
        plt.plot(y, label=lab)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if labels:
        plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpng)
    plt.close()
