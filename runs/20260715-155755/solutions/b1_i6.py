import torch
from torch import cuda


def triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    """Compute c = a + scalar * b for CUDA tensors already on the GPU using optimized fusion via TorchCompile/Triton-style logic emulation or fused operations to maximize bandwidth.

    To achieve maximum GB/s in STREAM (triad), we must minimize memory traffic per element 
    and avoid intermediate allocations. The operation c = a + scalar * b requires reading 'a' and 'b',
    and writing 'c'. Total bytes moved: size(a) + size(b) + size(c).
    For float32, this is (1+1+1)*4*N bytes per iteration. 

    The most effective way in high-level Python to trigger kernel fusion and avoid the intermediate tensor 'scalar * b'
    is using torch.compile or an out-of-place fused operation that avoids allocating a temporary buffer for scalar multiplication.
    """
    # We use torch.compile (available since PyTorch 2.0) which uses Triton under the hood to generate 
    # highly optimized, single-pass kernels with register tiling and vectorized loads/stores (.float4).
    # This eliminates the intermediate 'scalar * b' allocation and reduces memory passes from two to one.

    @torch.compile(mode="max-autotune")
    def fused_triad_kernel(a: torch.Tensor, b: torch.Tensor, s: float) -> torch.Tensor:
        # This single line inside a compiled function is transformed into an 
        # optimized Triton kernel that performs the operation in one pass over memory.
        return a + (s * b)

    try:
        # Attempt to use fused computation via compilation for maximum bandwidth performance.
        return fused_triad_kernel(a, b, scalar)
    except Exception:
        # Fallback if torch.compile is unavailable or fails in the environment 
        # (though it should be present per prompt requirements).
        # We use addcmul which is a built-in ATen function that performs c = input + value * tensor1 * tensor2,
        # but since we have scalar instead of second tensor, 'add' with scaling can also work. 
        # However, the standard expression in an optimized context (like torchscript or compile) 
        # is what triggers single-pass execution effectively without extra memory traffic for temp tensors.
        return a + b * scalar

if __name__ == "__main__":
    # Test setup to verify correctness and measure bandwidth logic if needed locally.
    n = 128000000
    a_t = torch.randn(n, device='cuda', dtype=torch.float32)
    b_t = torch.randn(n, device='cuda', dtype=torch.float32)
    s = 1.5

    # Warmup and verification
    res = triad(a_t, b_t, s)
    expected = a_t + (s * b_t)
    assert torch.allclose(res, expected), "Result mismatch!"
