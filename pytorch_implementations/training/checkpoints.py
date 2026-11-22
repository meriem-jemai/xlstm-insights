import os
import torch

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[Checkpoint] Saved model to {path}")

def load_checkpoint(model, path, device=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")
    checkpoint = torch.load(path, map_location=device or torch.device("cpu"))
    model.load_state_dict(checkpoint)
    print(f"[Checkpoint] Loaded model from {path}")
    return model


