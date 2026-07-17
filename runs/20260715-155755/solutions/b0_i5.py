import torch


def triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    """Compute c = a + scalar * b for CUDA tensors already on the GPU using kernel fusion.

    To maximize achieved bandwidth (GB/s), we must minimize total DRAM traffic. 
    The naive implementation `a + scalar * b` performs two memory passes:
    1. Read 'b', multiply by scalar, write temporary tensor T to VRAM.
    2. Read 'a' and 'T', add them, write result 'c' back to VRAM.

    Total data moved (Bytes): 3*size(b) + 2*size(a). For large tensors of same size: ~5 * elements_size.

    By using torch.compile with the default mode='default', TorchDynamo and
    TorchInductor perform loop fusion via Triton kernels. This fuses both operations into a single
    kernel pass that reads 'a' once, reads 'b' once, performs math in registers/L1, 
    and writes result to memory exactly once.

    Total data moved (Bytes): size(a) + size(b) + size(c). For large tensors: ~3 * elements_size.
    This reduction from a factor of 5x load-store operations to 3x is the key to maximizing GB/s,
    as bandwidth measurements are calculated based on (Total Bytes Read + Total Bytes Written / Time),
    and reducing denominator bytes while keeping numerator time constant or lower increases efficiency.

    We use 'torch._dynamo' and specifically decorate with torch.compile if not already compiled 
    to ensure the Triton kernel is generated for this specific operation pattern.
    """
    # We define a local fused function to be compiled by TorchInductor (Triton).
    # The compilation happens once on first call or when shapes/types change significantly, 
    # but in standard benchmarks like STREAM-style tests, the tensors are reused across iterations.

    @torch.compile(mode="default")
    def fused_triad(a: torch.Tensor, b: torch.Tensor, s: float) -> torch.Tensor:
        return a + (s * b)

    # Note: In production/benchmarking environments, the function should be 
    # compiled outside to avoid measuring compilation overhead during timing loops.
    # However, within this interface scope, we rely on TorchInductor's ability 
    # to optimize elementwise patterns into a single Triton kernel loop.

    return fused_triad(a, b, scalar)


