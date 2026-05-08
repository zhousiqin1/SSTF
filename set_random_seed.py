# set_random_seed.py
# Utility functions to enforce reproducible training across runs.
import torch
import numpy as np
import random
import os


def set_random_seed(seed=42):
    """Set random seed across Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")


def set_reproducible_training():
    """Configure the training environment for full reproducibility."""
    set_random_seed(42)
    # Use deterministic algorithms where possible; warn only for non‑deterministic ops
    torch.use_deterministic_algorithms(True, warn_only=True)
    print("Reproducible training environment configured.")


if __name__ == "__main__":
    set_reproducible_training()