import torch
import triton
import triton.language as tl

def triad_kernel(a_ptr, b_ptr, c_ptr, scalar, n_elements, BLOCK_SIZE: int):
    """
    Triton kernel for fused triplet operation (c = a + s * b).
    This implementation minimizes memory traffic by performing exactly one 
    read of 'a', one read of 'b', and one write to 'c' per element.
    The Triton compiler handles the software pipelining/prefetching logic, 
    ensuring that loads for next blocks are overlapped with computation.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data from global memory into SRAM/Registers
    a = tl.load(a_ptr + offsets, mask=mask, interpolation="nearest") # placeholder logic for standard load
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute fused operation: a[i] + scalar * b[i] 
    # This occurs in registers/SRAM without intermediate global memory writes.
    res = a + (scalar * b)

    # Store result back to destination tensor 'c'
    tl.store(c_ptr + offsets, res, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['n_elements'],
)
def triad_kernel_autotuned(a_ptr, b_ptr, c_ptr, scalar, n_elements, BLOCK_SIZE: int):
    """The autotuned version of the kernel."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    res = a + (scalar * b)
    tl.store(c_ptr + offsets, res, mask=mask)

def triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    """Compute c = a + scalar * b for CUDA tensors already on the GPU using Triton fusion.

    This implementation uses an autotuned Triton kernel to fuse elementwise 
    multiplication and addition into a single pass over global memory. This reduces
    the total number of bytes moved across the GDDR/HBM bus from (4*N_a + 8*N_{intermediate} + 2*N_c) 
    to exactly one read for 'a', one read for 'b', and one write for 'c'.

    Achieves maximum bandwidth by minimizing memory traffic per element.
    """
    n_elements = a.numel()
    # Pre-allocate output tensor to avoid allocation overhead inside the kernel call, 
    # though in common usage this is provided as part of an existing buffer or created once.
    c = torch.empty_like(a)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    triad_kernel_autotuned[grid](
        a, b, c, 
        scalar, 
        n_elements, 
        BLOCK_SIZE=1024 # This value is actually overridden by autotune config during runtime execution if called via triton decorators properly.
    )

    return c

# Note: The function signature 'triad' expects to return a new tensor or handle the result of an operation. 
# To maximize bandwidth in STREAM benchmarks, we use Triton-based fusion which effectively turns
# (a + scalar * b) into exactly one load(A), one load(B), and one store(C).

