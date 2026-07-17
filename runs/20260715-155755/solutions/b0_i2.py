import torch

def triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    """Compute c = a + scalar * b for CUDA tensors already on the GPU.

    To maximize achieved memory bandwidth (GB/s), we must minimize total DRAM traffic. 
    The naive implementation `a + scalar * b` performs two passes over VRAM:
    1. Read 'b', compute temp, write to intermediate buffer (Read+Write).
    2. Read 'a' and temporary buffer, compute sum, write result (Read+Write).

    By using torch.compile with the default Inductor backend, PyTorch performs 
    automatic kernel fusion via Triton. This fuses both operations into a single 
    CUDA/Triton loop that reads each element of 'a' and 'b', computes the value in registers, 
    and writes to result once (Read+Write). Effectively reducing memory traffic by ~33%.

    Note: We use torch._dynamo.optimize or simply @torch.compile on a wrapper function.
    The implementation below uses an internal compiled kernel via torch.compile(mode='reduce-overhead').
    """
    # Define the fused operation logic inside a helper to ensure 100% compatibility with decorators/compilation
    @torch.compile(mode="reduce-overhead")
    def _fused_triad(a: torch.Tensor, b: torch.Tensor, s: float) -> torch.Tensor:
        # This expression is fused into a single kernel by TorchInductor (Triton).
        return a + s * b

    # We call the compiled function directly with all required arguments to avoid TypeError 
    # and ensure there are no missing positional argument errors during evaluation.
    return _fused_triad(a, b, scalar)

