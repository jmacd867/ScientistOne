import torch


def triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    """Compute c = a + scalar * b for CUDA tensors already on the GPU.

    Baseline: naive elementwise expression. This allocates an intermediate
    tensor for `scalar * b` plus the output tensor — two avoidable
    allocations/passes a tuned kernel can remove.
    """
    return a + scalar * b
