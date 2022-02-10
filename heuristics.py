"""
Heuristic reward functions for AstroGym.
"""
import numpy as np
import torch


def hoyer_torch(obs):
    """
    Calculate the Hoyer sparsity of a PyTorch tensor.

    This is just the ratio between the L2 and L1 losses!
    """
    N = torch.numel(obs)**0.5
    sum_ = torch.sum(obs)
    root_sum = torch.sqrt(torch.sum(obs**2))
    hoyer = (N - (sum_/root_sum))/(N - 1)
    return hoyer

def hoyer_numpy(obs):
    """
    Calculate the Hoyer sparsity of a NumPy array.

    This is just the ratio between the L2 and L1 losses!
    """
    N = np.size(obs)**0.5
    sum_ = np.sum(obs)
    root_sum = np.sqrt(np.sum(obs**2))
    hoyer = (N - (sum_/root_sum))/(N - 1)
    return hoyer

def brightness_diff(obs, baseline):
    """
    Calculate the change in mean pixel value vs a baseline value.
    """
    return obs.mean() - baseline