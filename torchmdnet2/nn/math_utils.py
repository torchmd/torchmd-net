import math
import torch


def safe_norm(input, dim=0, keepdims=True, eps=1e-16):
    '''Compute Euclidean norm of input so that 0-norm vectors can be used in
    the backpropagation '''
    return torch.sqrt(torch.square(input).sum(dim=dim, keepdims=keepdims) + eps) - math.sqrt(eps)

def safe_normalization(input, norms):
    '''Normalizes input using norms avoiding divitions by zero'''
    mask = (norms > 0.).flatten()
    out = input.clone()
    out[mask] = input[mask] / norms[mask]
    return out