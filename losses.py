"""
Loss functions for astrogym
"""
import torch

def hoyer(x):
    """
    Calculate the hoyer sparsity of a pytorch tensor.

    This is just the ratio between the L2 and L1 losses!
    """
    N = torch.numel(x)**0.5
    sum_ = torch.sum(x)
    root_sum = torch.sqrt(torch.sum(x**2))
    hoyer = (N - (sum_/root_sum))/(N - 1)
    return hoyer
