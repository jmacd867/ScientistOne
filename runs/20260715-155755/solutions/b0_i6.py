import torch


def triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    """Compute c = a + scalar * b for CUDA tensors already on the GPU.

    To maximize achieved GB/s in an elementwise operation (STREAM-style), 
    the bottleneck is memory bandwidth (DRAM traffic). The naive implementation 
    (a + scalar * b) performs two separate kernels: one to compute temp = s*b, 
    and another for a + temp. This results in loading 'b', storing 'temp', 
    loading 'a' and 'temp', then writing the result.

    By using torch.compile (with Inductor backend), we trigger Triton-based 
    kernel fusion. The compiler fuses these operations into a single CUDA kernel, 
    reducing memory traffic from:
        (Read b + Write temp) + (Read a + Read temp + Write c) = 2R/1W for 'b' and 3R/1W total? No.
        Actually: Load B -> Compute s*B -> Store Temp; Load A, Load Temp -> Add -> Store C.
    Into a single pass:
        Load A (Read), Load B (Read) -> compute scalar * b + a in registers -> Write C (Write).

    This minimizes the number of trips to VRAM and eliminates intermediate allocations entirely. 
    We use 'mode="reduce-overhead"' for maximum performance on fixed shapes, as it uses CUDA graphs.
    """
    # We define an inner function so torch_compile can specialize/optimize based on context if needed.
    @torch.compile(mode="reduce-overhead")
    def fused_triad(a: torch.Tensor, b: torch.Tensor, s: float) -> torch.Tensor:
        return a + s * b

    # Note: In an actual production loop (like STREAM), the first call incurs overhead s/cache warming.
    # The benchmark environment typically accounts for this or runs multiple iterations.
    return fused_triad(a, b, scalar)


if __name__ == "__main__":
    # Test case to ensure correctness and functionality
    size = 1024 * 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    a = torch.randn(size, device=device, dtype=dtype)
    b = torch.randn(size, device=device, dtype=dtype)
    scalar = 1.5

    # Warmup for compilation overhead (essential to see real performance in benchmarks)
    for _ in range(3):
        _ = triad(a, b, scalar)

    res = triad(a, b, scalar)
    expected = a + scalar * b

    if torch.allclose(res, expected):
        print("Verification Successful")
    else:
        print("Verification Failed")
