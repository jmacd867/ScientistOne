import torch
import triton
import triton.language as tl

# The goal is to maximize achieved bandwidth for c = a + scalar * b. 
# A single-pass kernel avoids the intermediate allocation (scalar * b) and reduces memory traffic:
# Total bytes moved per element in fused version: read(a), read(b), write(c). -> 3 elements/ops.
# Naive implementation moves: temp = scalar*b [read b, write temp], c= a + temp [read a, read temp, write c].

@triton.jit
def triad_kernel(
    a_ptr, b_ptr, c_ptr, 
    scalar, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    """Fused Triton kernel for the triadic operation."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load a and b from global memory into SRAM/Registers
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # Perform the computation in registers: c[i] = a[i] + scalar * b[i]
    c = a + (scalar * b)

    # Write result back to global memory once
    tl.store(c_ptr + offsets, c, mask=mask)


def triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    """Compute c = a + scalar * b for CUDA tensors already on the GPU using Triton fusion."""
    n_elements = a.numel()
    # Pre-allocate output tensor to avoid in-kernel allocation overhead and ensure 1 pass logic matches STREAM pattern (c is destination)
    c = torch.empty_like(a)

    if n_elements == 0:
        return c

    # Grid calculation based on block size
    BLOCK_SIZE = 1024  # Standard tile size for elementwise ops
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch the fused kernel
    triad_kernel[grid](
        a, b, c,
        scalar, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )

    return c


def test_correctness():
    """Verification of implementation correctness."""
    size = 10**7
    a = torch.randn(size, device='cuda')
    b = torch.randn(size, device='cuda')
    scalar = 2.5
    expected = a + scalar * b
    actual = triad(a, b, scalar)
    assert torch.allclose(actual, expected), "Correctness check failed!"

if __name__ == "__main__":
    # Run correctness test if executed directly
    test_correctness()
