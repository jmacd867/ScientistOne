import torch

def triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    """Compute c = a + scalar * b for CUDA tensors already on the GPU.

    To maximize achieved bandwidth (GB/s), we must minimize memory traffic relative 
    to total data moved via STREAM-style metric calculation logic. In an elementwise 
    triad, if we treat 'c' as our result:
      Traffic = Read(a) + Read(b) + Write(c).
      Total Data Moved (Bytes) = n * size_of_element * 3 bits/ops? No, STREAM measures bytes.
      The bandwidth is measured by the total volume of data moved across memory buses divided 
      by time taken to compute all elements in a single pass over 'a' and 'b'.

    To achieve peak GB/s (which effectively means maximizing throughput per byte read),
    we must ensure that for every element loaded, we perform as much computation as possible.
    However, the benchmark metric is typically: Bandwidth = Total Bytes Processed / Time. 
      Total Bytes in triad(a+b*k) = size_of(a) + size_of(b) + size_of(c).

    The naive `a + scalar * b` performs:
    1. Load 'b', multiply by constant, Store to Temp (Read B, Write T) -> 2 bytes/element traffic per element of scale-op? No.
       Actually: Read B [8] -> Mul -> Write T [8]. Then Read A [8], Read T [8], Add [] -> Write C [8]. Total = 4 reads + 1 write (or similar).

    The fused kernel performs exactly one pass over the memorys:
      Read 'a' and 'b', compute, Store to 'c'.
      Total Bytes Processed per element = size(a) + size(b) + size(out_c) = 3 * sizeof(dtype).

    We use torch.compile with a mode that favors aggressive fusion (max-autotune), which uses Triton kernels.
    This reduces the memory traffic from ~4 or 5 bytes/element to exactly 3 bytes/element, and ensures single pass execution.
    """
    # We decorate this function locally within triad if possible, but for a library call structure:
    pass

@torch.compile(mode="max-autotune")
def _fused_triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    # Using addcmul or simply the fused expression inside a compiled block 
    # ensures Triton generates an optimized single kernel with one read/write pass per element.
    return torch.add(a, b * scalar)

def triad_optimized(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    """Implementation using a fused approach via Torch Compile."""
    # We use the compiled version which fuses 'b*scalar' and '+ a'. 
    return _fused_triad(a, b, scalar)

def triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    """Compute c = a + scalar * b for CUDA tensors already on the GPU."""
    # Using addcmul is also an option as it's often implemented via highly optimized single kernels in ATen/cuBLAS logic 
    # but 'a + (b * s)' inside torch.compile generates exactly one Triton kernel that reads a, b and writes c once.
    return _fused_triad(a, b, scalar)

