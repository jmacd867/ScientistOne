import torch

def triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    """Compute c = a + scalar * b for CUDA tensors already on the GPU.

    To maximize memory bandwidth (GB/s), we must minimize the total bytes 
    transferred across the global memory bus per operation element. 
    The naive `a + scalar * b` performs three passes:
    1. Load 'b', multiply by scalar, write to intermediate tensor T' (~3N transfers).
    2. Load 'a' and load 'T'', add them, write to result C (~3N transfers).

    To optimize for STREAM-style bandwidth (Memory Bound), we use torch.compile 
    with a focus on kernel fusion. This allows the compiler to generate an 
    elementwise Triton or CUDA kernel that performs:
      c[i] = a[i] + scalar * b[i]
    This single-pass approach ensures each element of 'a' and 'b' is loaded exactly once,
    and the result for index i is written back to memory only once. 

    Total transfers per element: Load A (1), Load B (1), Store C (1) = 3 elements total.
    """
    # We use torch.compile which uses Triton under the hood on CUDA devices.
    # This fuses the operations into a single kernel, achieving optimal bandwidth utilization.
    @torch_compile_optimized
    def fused_triad(a: torch.Tensor, b: torch.Tensor, s: float) -> torch.Tensor:
        return a + (s * b)

    return fused_triad(a, b, scalar)


# Helper to ensure the function is compiled for maximum performance during execution.
def torch_compile_optimized(fn):
    """Uses TorchInductor/Triton to fuse operations into a single-pass kernel."""
    try:
        return torch.compile(fn, mode="reduce-overhead")
    except Exception:
        # Fallback for environments where torch.compile might not be available or supported 
        # (though it is standard in modern PyTorch). The logic remains the same conceptually.
        return fn

if __name__ == "__main__":
    import time

    # Setup parameters to measure bandwidth-bound performance
    N = 1024 * 1024 * 512  # ~536M elements (approx 2GB per tensor)
    device = "cuda"
    dtype = torch.float3_t if hasattr(torch, 'float32') else torch.float32 # Ensure float32 usage

    a = torch.randn(N, device=device, dtype=torch.float32)
    b = torch.randn(N, device=device, dtype=torch.float32)
    scalar = 1.5

    # Warmup for compilation and CUDA kernel launch overheads
    for _ in range(5):
        _ = triad(a, b, scalar)
    torch.cuda.synchronize()

    # Timing loop to measure steady-state bandwidth performance
    iters = 20
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        # We perform the operation and ensure it is not optimized away by 
        # performing a simple reduction on result if needed, though here we return C directly.
        res = triad(a, b, scalar)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event) / iters

    # Bandwidth Calculation: 
    # In a single-pass kernel (c[i] = a[i] + s*b[i]):
    # We read 'a' (4 bytes), Read 'b' (4 bytes), Write 'res' (4 bytes). Total = 12 bytes per element.
    bytes_per_element = 3 * 4 # float32 is 4 bytes, we touch A, B and write C
    total_bytes_transferred = N * bytes_per_element * iters / iters  # average over iterations

    bandwidth_gbs = (total_bytes_transferred) / (elapsed_time_ms / 1000.0) / 1e9

    print(f"Measured Bandwidth: {bandwidth_gbs:.2f} GB/s")
