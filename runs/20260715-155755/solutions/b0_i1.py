import torch


def triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    """Compute c = a + scalar * b for CUDA tensors already on the GPU.

    Approach: Kernel Fusion via `torch.compile`. 
    The baseline implementation performs two separate memory passes (one to compute 
    `scalar * b` and write it, another to add 'a' to that result). By using 
    `torch.compile`, we trigger Triton-based kernel fusion which merges these operations 
    into a single CUDA kernel pass over the data. This reduces total DRAM traffic:
    instead of (Read B + Write Temp) + (Read A + Read Temp + Write C), it becomes
    (Read A + Read B + Write C).

    For STREAM-style benchmarks, bandwidth is measured as 3 * size_in_bytes / time.
    Reducing the number of memory operations directly increases achievable GB/s.
    """
    # We use torch.compile to fuse these elementwise ops into a single kernel pass.
    # 'reduce-scale' or default mode works well for fused point-wise kernels in Triton.
    @torch_compiled_triad
    def compute(a, b, s):
        return a + s * b

    return compute(a, b, scalar)


# Global cache to avoid the overhead of compiling during every function call execution 
# (compilation happens on first call). In real-world scenarios/benchmarks, we want 
# only the kernel performance measured.
_compiled_fn = None

def torch_compiled_triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    global _compiled_fn
    if _compiled_fn is None:
        # We define a simple function to be compiled by TorchDynamo/Triton
        def core_logic(t1, t2, s):
            return t1 + s * t2
        try:
            # mode="max-autotune" provides the most aggressive kernel fusion and 
            # tuning for Triton kernels (e.g., loop unrolling/tiling).
            _compiled_fn = torch.compile(core_logic, mode="max-autotune")
        except Exception:
            # Fallback if compile fails in specific environments
            return a + scalar * b

    try:
        return _compiled_fn(a, b, scalar)
    except Exception:
        return a + scalar * b


