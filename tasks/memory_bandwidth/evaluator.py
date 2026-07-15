import importlib.util

import torch

_SIZES = [32_000_000, 128_000_000, 512_000_000]  # elements per tensor (float32)
_WARMUP = 5
_REPS = 30
_SCALAR = 3.0
_SEED = 0


def _load_solution(solution_path: str):
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.triad


def _check_solution(triad, a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    c = triad(a, b, scalar)
    if c.shape != a.shape:
        raise ValueError(f"output shape {tuple(c.shape)} != input shape {tuple(a.shape)}")
    if c.dtype != a.dtype:
        raise ValueError(f"output dtype {c.dtype} != input dtype {a.dtype}")
    if c.device != a.device:
        raise ValueError(f"output device {c.device} != input device {a.device}")
    expected = a + scalar * b
    if not torch.allclose(c, expected, rtol=1e-4, atol=1e-4):
        raise ValueError("triad result does not match a + scalar * b within tolerance")
    return c


def evaluate(solution_path: str, workdir: str) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("memory_bandwidth task requires a CUDA-capable GPU")
    triad = _load_solution(solution_path)
    device = torch.device("cuda")
    gbps_by_size, lines = [], []
    for n in _SIZES:
        gen = torch.Generator(device=device).manual_seed(_SEED)
        a = torch.rand(n, device=device, dtype=torch.float32, generator=gen)
        b = torch.rand(n, device=device, dtype=torch.float32, generator=gen)
        _check_solution(triad, a, b, _SCALAR)
        for _ in range(_WARMUP):
            triad(a, b, _SCALAR)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(_REPS):
            triad(a, b, _SCALAR)
        end.record()
        torch.cuda.synchronize()
        elapsed_s = start.elapsed_time(end) / 1000.0
        bytes_per_call = 3 * n * a.element_size()  # read a, read b, write c
        gbps = (bytes_per_call * _REPS) / elapsed_s / 1e9
        gbps_by_size.append(gbps)
        lines.append(f"n={n}: {gbps:.2f} GB/s")
    score = sum(gbps_by_size) / len(gbps_by_size)
    lines.append(f"mean over {len(_SIZES)} sizes: {score:.2f} GB/s")
    return {"score": round(score, 2), "log": "\n".join(lines)}
