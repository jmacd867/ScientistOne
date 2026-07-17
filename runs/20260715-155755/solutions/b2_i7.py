import torch
import triton
import triton.language as tl

# The goal is to maximize achieved bandwidth for c = a + scalar * b (triad).
# To achieve maximum throughput, we must minimize memory traffic: 1 read of 'a', 
# 1 read of 'b', and 1 write of 'c'. Total bytes moved per element: size(dtype) * 3.
# The naive implementation performs an extra load/store for the intermediate (scalar * b).

@triton.jit
def triad_kernel(
    a_ptr, b_ptr, c_ptr, 
    scalar, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    """Triton kernel for fused triadic operation."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data from global memory into SRAM (Registers/L1 cache handled by hardware/Triton)
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # Perform fused computation: a + scalar * b 
    # This avoids the intermediate memory round-trip for (scalar * b).
    res = a + scalar * b

    # Store result back to global memory
    tl.store(c_ptr + offsets, res, mask=mask)

def triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    """Compute c = a + scalar * b for CUDA tensors already on the GPU using Triton fusion."""
    # Ensure output tensor is allocated once (as per STREAM requirement of maximizing bandwidth 
    # by avoiding unnecessary allocations within the kernel logic).
    c = torch.empty_like(a)
    n_elements = a.numel()

    if n_elements == 0:
        return c

    # Heuristics for block size to ensure high occupancy and coalesced access.
    # We use powers of two compatible with GPU warp sizes/memory bus widths (128, 512, 1024...).
    # Triton's autotuner can be used here if we were running a benchmark loop, but for single call:
    BLOCK_SIZE = 1024

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch the fused kernel
    triad_kernel[grid](
        a, b, c, 
        scalar, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )

    return c

# Note on Optimization: 
# The achieved bandwidth (GB/s) is calculated as: Total Bytes / Time.
# For Triad(a + s*b), the minimum data movement per element of float32 is 12 bytes.
# If we see ~60-70% efficiency relative to theoretical peak, it's a success in real environments.
# The primary bottleneck for STREAM operations (memory bound) isn't compute but memory latency/bandwidth.
# By using Triton with fused loads and stores, we ensure the hardware performs 1 Load(a), 1 Load(b), 1 Store(c).

if __name__ == "__main__":
    # Testing implementation integrity
    size = 50_000_000
    scalar_val = 2.5
    A = torch.randn(size, device='cuda')
    B = torch.randn(size, device='cuda')

    C_custom = triad(A, B, scalar_val)
    C_naive = A + scalar_val * B

    # Check correctness
    if torch.allclose(C_custom, C_naive):
        print("Verification Successful: Custom Triton kernel matches naive implementation.")
    else:
        print("Verification Failed!")
