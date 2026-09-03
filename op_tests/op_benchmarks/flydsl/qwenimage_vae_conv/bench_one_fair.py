#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Three-tier fair comparison for one VAE conv shape.

Tier A (end-to-end, strictly like-for-like): every implementation produces the
        same output from the same NCHW inputs and pays all of its own costs.
Tier B (GEMM quality): FlyDSL's GEMM kernel against hipBLASLt on the same
        M/N/K. Not like-for-like — hipBLASLt reads a materialised M*K matrix
        contiguously while implicit GEMM gathers from the ~9x smaller input —
        so the im2col materialisation cost is reported separately rather than
        folded in either direction. FlyDSL gets its best per-shape tile because
        hipBLASLt selects a per-shape kernel.
Tier C (design ceiling): FlyDSL with a hypothetical zero-cost transpose.

FLOPs are counted on useful work (Cin*R*S), not on FlyDSL's padded Cin, so its
channel padding shows up as reduced throughput rather than being washed out.
All times are profiler on-device times.

Run via ``drive_fair.py`` (one fresh process per shape).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shapes import flydsl_root  # noqa: E402

sys.path.insert(0, str(flydsl_root()))

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from kernels.conv.conv3d_implicit import DEFAULT_TILE, conv3d_implicit

sid, cin, cout, hin, stride, padding, freq, path = (
    sys.argv[1],
    int(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
    int(sys.argv[5]),
    int(sys.argv[6]),
    int(sys.argv[7]),
    sys.argv[8],
)
R = S = 3
CANDIDATES = [
    tuple(DEFAULT_TILE),
    (128, 256, 2, 4),
    (256, 128, 2, 4),
    (256, 256, 4, 4),
    (128, 128, 4, 2),
]
ITERS = 10
TRIALS = 2
NAN = float("nan")


def gpu(call):
    """(total_us, {kernel: us}) on-device, best of TRIALS."""
    best, best_per = None, None
    for _ in range(TRIALS):
        for _ in range(4):
            call()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(ITERS):
                call()
            torch.cuda.synchronize()
        per = {}
        for e in prof.key_averages():
            if (
                e.device_type != torch.autograd.DeviceType.CUDA
                or e.self_device_time_total <= 0
            ):
                continue
            k = e.key.split("(")[0][:40]
            per[k] = per.get(k, 0.0) + e.self_device_time_total / ITERS
        t = sum(per.values())
        if best is None or t < best:
            best, best_per = t, per
    return best, best_per


def split_fly(per):
    g = sum(v for k, v in per.items() if "conv3d_implicit_kernel" in k)
    t = sum(v for k, v in per.items() if "transpose_kernel" in k)
    o = sum(
        v
        for k, v in per.items()
        if "conv3d_implicit_kernel" not in k and "transpose_kernel" not in k
    )
    return g, t, o


torch.manual_seed(0)
os.environ.setdefault("FLYDSL_CONV3D_AUTOTUNE", "0")
x4 = torch.randn((1, cin, hin, hin), device="cuda", dtype=torch.bfloat16)
w4 = torch.randn((cout, cin, R, S), device="cuda", dtype=torch.bfloat16)
b = torch.randn((cout,), device="cuda", dtype=torch.float32)
bbf = b.to(torch.bfloat16)

ref = F.conv2d(x4, w4, bias=bbf, stride=stride, padding=padding)
p = ref.shape[2]
M, N, K = p * p, cout, cin * R * S
flops = 2.0 * M * N * K


def check(y):
    assert y.shape == ref.shape, f"{sid}: {tuple(y.shape)} vs {tuple(ref.shape)}"
    e = ((y.float() - ref.float()).abs().mean() / ref.float().abs().mean()).item()
    assert e < 2e-2, f"{sid}: rel err {e:.2e}"


def fly(tile):
    return conv3d_implicit(
        x4, w4, bias=b, stride=stride, padding=padding, tile=tile
    )


check(fly(tuple(DEFAULT_TILE)))
t_def, per_def = gpu(lambda: fly(tuple(DEFAULT_TILE)))
g_def, tr_def, o_def = split_fly(per_def)

best = (g_def, t_def, tr_def, o_def, tuple(DEFAULT_TILE))
for tile in CANDIDATES[1:]:
    try:
        check(fly(tile))
        t, per = gpu(lambda: fly(tile))
        g, tr, o = split_fly(per)
        if g < best[0]:
            best = (g, t, tr, o, tile)
    except Exception:
        continue
g_best, t_best, tr_best, o_best, tile_best = best

t_mm = t_im2col = t_unfold_mm = NAN
try:
    A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    Bm = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
    Cm = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    t_mm, _ = gpu(lambda: torch.mm(A, Bm, out=Cm))
    del A, Bm, Cm
    torch.cuda.empty_cache()
except torch.OutOfMemoryError:
    torch.cuda.empty_cache()

try:
    t_im2col, _ = gpu(lambda: F.unfold(x4, (R, S), padding=padding, stride=stride))
    torch.cuda.empty_cache()
except torch.OutOfMemoryError:
    torch.cuda.empty_cache()

try:
    w2 = w4.reshape(cout, K).t().contiguous()

    def unfold_mm():
        a = F.unfold(x4, (R, S), padding=padding, stride=stride).squeeze(0).t()
        return (a @ w2 + bbf).t().reshape(1, cout, p, p)

    check(unfold_mm())
    t_unfold_mm, _ = gpu(unfold_mm)
    del w2
    torch.cuda.empty_cache()
except torch.OutOfMemoryError:
    torch.cuda.empty_cache()

t_mio, _ = gpu(lambda: F.conv2d(x4, w4, bias=bbf, stride=stride, padding=padding))

print(
    f"RESULT\t{sid}\t{cin}\t{cout}\t{hin}\t{stride}\t{freq}\t{path}\t{M}\t{N}\t{K}\t{flops:.6e}\t"
    f"{t_def:.2f}\t{g_def:.2f}\t{tr_def:.2f}\t"
    f"{t_best:.2f}\t{g_best:.2f}\t{tr_best:.2f}\t{'x'.join(str(v) for v in tile_best)}\t"
    f"{t_mm:.2f}\t{t_im2col:.2f}\t{t_unfold_mm:.2f}\t{t_mio:.2f}"
)
