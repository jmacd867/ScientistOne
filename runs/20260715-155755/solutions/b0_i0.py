import torch

def triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    """Compute c = a + scalar * b for CUDA tensors already on the GPU.

    The goal is to maximize achieved bandwidth by reducing memory traffic (DRAM R/W).
    A naive implementation `a + scalar * b` performs two separate passes over 
    the data and allocates an intermediate tensor:
       1. Load 'b', multiply, write temporary result T = s*b  (Read B, Write T)
       2. Load 'a' and 'T', add them, write C           (Read A, Read T, Write C)

    By using torch.compile with the default Inductor backend (which uses Triton), 
    the operations are fused into a single kernel:
       1. Single pass: Load B, load A; compute res = a + s*b; write C  (Read A/B, Write C)

    This reduction in memory traffic from two reads and one write per element to 
    one read of each input and one final write maximizes the effective bandwidth (GB/s).
    """
    # We use torch.compile which invokes Triton under the hood on CUDA tensors.
    # This performs kernel fusion, transforming multiple operations into a single loop.
    static_triad = torch.compile(lambda x, y, s: x + s * y)

    return static_triad(a, b, scalar)

if __name__ == "__main__":
    import time

    # Setup parameters for benchmarking (Large enough to saturate bandwidth/hide overheads)
    N = 10**8  # ~400MB per tensor if float32. Total data movement is significant.
    device = "cuda"
    dtype = torch.float32
    scalar_val = 1.5

    a = torch.randn(N, device=device, dtype=dtype)
    b = torch.randn(N, device=device, dtype=dtype)

    # Warmup for JIT/Compilation overhead (torch.compile is lazy)
    for _ in range(3):
        _ = triad(a, b, scalar_val)
    torch.cuda.synchronize()

    # Benchmark loop
    iters = 50
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        res = triad(a, b, scalar_val)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event) / iters # average ms per op
    avg_seconds = (elapsed_time_ms / 1000.0)

    # Calculate Bandwidth: Total Bytes Read/Written divided by time taken
    # In a single-pass fused kernel, we read 'a' and 'b', then write result to memory space of size N.
    # Note: The function returns c = a + s*b which is essentially an assignment into the output buffer. 
    # We assume for bandwidth calculation that C was pre-allocated or overwriting existing logic,
    # but even with allocation cost included in 'time', we calculate based on input/output traffic.
    bytes_per_element = a.element_size() * 2 + (a.element_size()) # Read A, B; Write C
    total_data_gb = (N * bytes_per_element) / (1024**3)

    bandwidth_gbs = total_data_gb / avg_seconds

    print(f"Average time per iteration: {avg_seconds*1000:.4f} ms")
    print(f"Achieved Bandwidth Estimate: {bandwidth_gbs:.2f} GB/s")
