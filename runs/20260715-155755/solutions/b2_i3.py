import torch
import triton
import triton.language as tl

# The goal is to maximize achieved bandwidth for c = a + scalar * b. 
# This operation (triad) involves reading 'a' and 'b', then writing 'c'. 
# Total bytes moved per element: sizeof(dtype)*3.
# Bandwidth calculation formula used by benchmarks typically assumes the throughput of data movement relative to total size read/written.

@triton.jit
def triad_kernel(
    a_ptr, b_ptr, c_ptr, 
    scalar, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    """
    A fused Triton kernel that performs the operation in a single pass over memory.
    This avoids intermediate allocations and maximizes coalesced access patterns.
    By using larger block sizes (up to hardware limits), we maximize throughput via pipelining 
    inherent in modern GPU architectures when handling large contiguous chunks of data.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load a and b into registers/SRAM in one pass per block to minimize memory pressure
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # Perform fused compute: c[i] = a[i] + scalar * b[i]
    res = a + (scalar * b)

    # Write back the result to memory once per element
    tl.store(c_ptr + offsets, res, mask=mask)


def triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    """Compute c = a + scalar * b for CUDA tensors already on the GPU using Triton fusion."""
    # Ensure output tensor is pre-allocated to avoid allocation overhead in benchmark loop if possible, 
    # though here we must return it as per interface requirement.
    c = torch.empty_like(a)
    n_elements = a.numel()

    if n_elements == 0:
        return c

    # Heuristics for block size to ensure high occupancy and coalesced access.
    # Using power-of-two blocks that fit well into the L1/SRAM cache hierarchy.
    BLOCK_SIZE = 4096 if n_elements > 4096 else triton.next_power_of_2(n_elements)

    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )

    triad_kernel[grid](
        a, b, c, 
        scalar, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )

    return c


# Note: To achieve peak performance in a real benchmark environment (like STREAM):
# One should use torch.compile() on the function if using standard PyTorch ops or run Triton kernels directly as shown above.
# The implementation provided uses an autotuned-style approach via direct kernel invocation 
# which bypasses Python overhead and intermediate allocations entirely for a single pass over memory.

