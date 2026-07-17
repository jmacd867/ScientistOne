import torch

def triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    """Compute c = a + scalar * b for CUDA tensors already on the GPU using 
    kernel fusion and memory-efficient operations to maximize GB/s bandwidth.

    To achieve maximum throughput in STREAM-style kernels (which are compute-light), 
    the goal is to minimize global memory roundtrips. The naive `a + scalar * b` 
    performs two reads for 'b' if not optimized: one for the multiplication and 
    one for the addition, plus a write. A fused kernel performs exactly three 
    reads (1 from 'a', 2 from 'b' - wait, no) -> actually it is reading 'a'[i], 
    reading 'b'[i] once each, performing one FMA-like operation locally in registers, 
    and writing to the output.

    We use `torch.compile` with an optimized backend (Inductor). Inductor uses 
    Triton under the hood which generates fused kernels that utilize vectorized 
    loads/stores and register tiling specifically designed for Ampere+ architectures, 
    effectively implementing high-performance memory coalescing similar to manual 
    cp.async implementations but with better portability across CUDA versions.

    The bandwidth calculation (GB/s) = Total Bytes Read + Total Bytes Written / Time.
    For triad: size(a)*dtype_size + size(b)*dtype_size + size(c)*dtype_size per element loop.
    """
    # We use torch.compile to trigger Triton kernel generation. 
    # Inductor will fuse the scalar multiplication and addition into a single 
    # fused CUDA kernel, eliminating intermediate allocations (scalar * b) 
    # and reducing memory traffic from O(3N) loads/stores back toward O(1 load 'a', 
    # 1 load 'b', 1 store 'c').

    @torch.compile(mode="max-autotune")
    def fused_triad(t_a, t_b, s):
        return torch.add(t_a, (t_b * s))

    # Note: We must ensure the operation is performed in a way that doesn't 
    # trigger an allocation of 'c' if we were doing it truly in-place on existing memory, 
    # but since the signature requires returning c = ..., and standard torch ops return new tensors,
    # our goal here is to ensure only ONE output tensor is allocated.

    return fused_triad(a, b, scalar)


if __name__ == "__main__":
    # Benchmark/Verification Script (Internal use for testing correctness)
    import time

    N = 1024**3 * 8 # Large enough to exceed L3 cache (~64GB if float32 + others > ~5-10 GB total data movement)
    device = "cuda"
    dtype = torch.float32
    scalar_val = 1.5

    a = torch.randn(N, device=device, dtype=dtype).cuda()
    b = torch.randn(N, device=device, dtype=dtype).cuda()

    # Warmup for Torch Compile and CUDA Context
    for _ in range(3):
        res_warmup = triad(a, b, scalar_val)
    torch.cuda.synchronize()

    # Timing loop
    iters = 20
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        res = triad(a, b, scalar_val)
    end_event.record()
    torch.cuda.synchronize()

    avg_time = start_event.elapsed_time(end_event) / iters # ms
    # Bandwidth Calculation: 3 loads (one 'a', one 'b' for multiply, then add involves another pass if not fused?)
    # In a perfectly fused kernel: Load A[i], Load B[i], Compute [A + s*B], Store C[i].
    # Bytes per element = size(float32) * 3 (1 read 'a', 1 read 'b', 1 write result).
    bytes_per_element = a.nelement() * 4 * 3 # Assuming float32 and fused kernel logic
    total_gb = bytes_per_element / (1024**3)
    bandwidth = total_gb / (avg_time / 1000)

    # Verification
    expected = a + scalar_val * b
    assert torch.allclose(res, expected), "Verification failed!"
