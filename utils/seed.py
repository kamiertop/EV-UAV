"""Reproducibility utilities."""

import logging
import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    # Must be set BEFORE any torch.cuda operation to take effect
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)

    # Suppress spconv's is_fx_tracing log-warning (uses logging, not warnings)
    logging.getLogger("torch.fx._symbolic_trace").setLevel(logging.ERROR)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
