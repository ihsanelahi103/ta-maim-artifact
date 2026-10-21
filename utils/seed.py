"""Reproducibility seed setter."""
import random, numpy as np, torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


CURRENT_SEED = 42
_orig_set_seed = set_seed


def set_seed(seed, *a, **k):
    global CURRENT_SEED
    CURRENT_SEED = int(seed)
    return _orig_set_seed(seed, *a, **k)
