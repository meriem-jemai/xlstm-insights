"""Loss curve plot (train and validation loss per epoch)."""
import matplotlib

matplotlib.use("Agg")  # save images without opening a window
import matplotlib.pyplot as plt


def plot_loss_curve(history, title="Training Loss", filename="loss_curve.png"):
    # c: takes the history rows from train_model and plots train AND validation loss
    if not history:
        return
    epochs = [r["epoch"] for r in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [r["train_loss"] for r in history], marker="o", markersize=4, label="train")
    plt.plot(epochs, [r["val_loss"] for r in history], marker="o", markersize=4, label="validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved loss plot to {filename}")
