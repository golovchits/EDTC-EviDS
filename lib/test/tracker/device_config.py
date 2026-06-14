"""
Device-agnostic configuration for local CPU/MPS testing.
Set env var EDTC_DEVICE=cpu (or mps) to override auto-detection.
On Snellius with CUDA available, this returns cuda and is functionally
identical to the original hardcoded .cuda() calls.
"""
import os
import torch


def get_device() -> torch.device:
    env_device = os.environ.get('EDTC_DEVICE', None)
    if env_device is not None:
        return torch.device(env_device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = get_device()
