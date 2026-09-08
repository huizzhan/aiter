#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen-Image VAE conv: per-shape (tile, WGM) sweep for the BF16 implicit-GEMM kernel.

Stage A of the 384 optimization plan. Measures the on-device ``conv3d_implicit_kernel``
time for every legal (tile, wgm) candidate and compares against both the shipped default
path (``tile=None``) and hipBLASLt's equivalent ``torch.mm``.

Each shape runs in a fresh process so JIT state cannot leak across candidates. Writes
``tile_wgm_sweep.json`` next to this script.

Usage::

    FLYDSL_ROOT=/workspace/FlyDSL python .../bench_tile_wgm.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "tile_wgm_sweep.json"

# sid, cin, cout, hin, stride, padding, freq, path
SHAPES = [
    ("enc_conv_in", 3, 96, 1024, 1, 1, 1, "1024"),
    ("enc_e0_res__dec_d3_res", 96, 96, 1024, 1, 1, 10, "1024"),
    ("enc_e1_res1", 96, 192, 512, 1, 1, 1, "1024"),
    ("enc_e1_res2__dec_d2_res", 192, 192, 512, 1, 1, 9, "1024"),
    ("enc_e2_res1__dec_d1_res1", 192, 384, 256, 1, 1, 2, "1024"),
    ("enc_e2_res2__dec_d1_res", 384, 384, 256, 1, 1, 8, "1024"),
    ("enc_e3_mid__dec_mid_d0", 384, 384, 128, 1, 1, 18, "1024"),
    ("enc_conv_out", 384, 32, 128, 1, 1, 1, "1024"),
    ("dec_conv_in", 16, 384, 128, 1, 1, 1, "1024"),
    ("dec_conv_out", 96, 3, 1024, 1, 1, 1, "1024"),
    ("enc_e0_downsample", 96, 96, 1025, 2, 0, 1, "1024"),
    ("enc_e1_downsample_spatial", 192, 192, 513, 2, 0, 1, "1024"),
    ("enc_e2_downsample_spatial", 384, 384, 257, 2, 0, 1, "1024"),
    ("dec_d0_upsample", 384, 192, 256, 1, 1, 1, "1024"),
    ("dec_d1_upsample", 384, 192, 512, 1, 1, 1, "1024"),
    ("dec_d2_upsample", 192, 96, 1024, 1, 1, 1, "1024"),
    ("dec_bottleneck_1328", 384, 384, 166, 1, 1, 18, "1328"),
    ("dec_d3_res_hot_1328", 96, 96, 1328, 1, 1, 10, "1328"),
]

R = S = 3
ITERS, TRIALS = 10, 2
# The box throttles ~17% under sustained load; idle between candidates so the sweep
# does not rank tiles by when they happened to run.
IDLE_S = 1.5
NAN = float("nan")


def flydsl_root() -> Path:
    return Path(os.environ.get("FLYDSL_ROOT", "/workspace/FlyDSL")).resolve()


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    root = str(flydsl_root())
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env["FLYDSL_CONV3D_AUTOTUNE"] = "0"
    return env


def run_one(sid, cin, cout, hin, stride, padding, freq, path) -> None:
    sys.path.insert(0, str(flydsl_root()))
    os.environ.setdefault("FLYDSL_CONV3D_AUTOTUNE", "0")

    import time as _time

    import torch
    import torch.nn.functional as F
    from kernels.conv.conv3d_autotune import BF16_CANDIDATES, WGM_VALUES
    from kernels.conv.conv3d_implicit import conv3d_implicit
    from torch.profiler import ProfilerActivity, profile

    cin, cout, hin, stride, padding, freq = map(
        int, (cin, cout, hin, stride, padding, freq)
    )

    def gpu(call, want="conv3d_implicit_kernel"):
        """On-device time of the kernels matching `want`, min over TRIALS."""
        best = None
        for _ in range(TRIALS):
            _time.sleep(IDLE_S)
            for _ in range(4):
                call()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(ITERS):
                    call()
                torch.cuda.synchronize()
            t = 0.0
            for e in prof.key_averages():
                if (
                    e.device_type != torch.autograd.DeviceType.CUDA
                    or e.self_device_time_total <= 0
                ):
                    continue
                if want is None or want in e.key:
                    t += e.self_device_time_total / ITERS
            if t > 0 and (best is None or t < best):
                best = t
        return NAN if best is None else best

    torch.manual_seed(0)
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

    def fly(tile=None, wgm=None):
        return conv3d_implicit(
            x4, w4, bias=b, stride=stride, padding=padding, tile=tile, wgm=wgm
        )

    # Shipped default: tile=None -> _pick_tile, wgm=1.
    check(fly())
    t_default = gpu(lambda: fly())

    rows = []
    for tile in BF16_CANDIDATES:
        for wgm in WGM_VALUES:
            call = partial(fly, tile, wgm)
            try:
                check(call())
                t = gpu(call)
            except Exception:  # noqa: BLE001  an illegal tile fails anywhere from compile to launch
                torch.cuda.empty_cache()
                continue
            if not math.isnan(t):
                rows.append({"tile": list(tile), "wgm": wgm, "gemm": t})

    def probe_mm():
        """hipBLASLt on the equivalent M/N/K, twice, as a thermal check.

        Operands live in this frame so they are freed before the caller returns.
        """
        a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        bm = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
        cm = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        mm = partial(torch.mm, a, bm, out=cm)
        return gpu(mm, want=None), gpu(mm, want=None)

    try:
        t_mm, t_mm2 = probe_mm()
    except torch.OutOfMemoryError:
        t_mm = t_mm2 = NAN
    torch.cuda.empty_cache()

    print(
        "RESULT"
        + json.dumps(
            {
                "sid": sid,
                "cin": cin,
                "cout": cout,
                "hin": hin,
                "stride": stride,
                "freq": freq,
                "path": path,
                "M": M,
                "N": N,
                "K": K,
                "flops": flops,
                "default": t_default,
                "mm": t_mm,
                "mm2": t_mm2,
                "cands": rows,
            }
        )
    )


def drive() -> None:
    env = child_env()
    cwd = str(flydsl_root())
    rows = []
    hdr = (
        f"{'shape':20s} {'x':>3s} {'M':>8s} {'K':>5s} | {'default':>8s} {'best':>8s} "
        f"{'gain':>6s} {'best (tile, wgm)':>22s} | {'mm':>8s} {'def/mm':>7s} {'best/mm':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for sp in SHAPES:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--one", *[str(v) for v in sp]],
            capture_output=True,
            text=True,
            env=env,
            cwd=cwd,
            check=False,
        )
        line = next(
            (l for l in proc.stdout.splitlines() if l.startswith("RESULT")), None
        )
        if line is None:
            print(f"FAILED {sp[0]}\n{proc.stderr[-1500:]}", flush=True)
            continue
        r = json.loads(line[len("RESULT") :])
        rows.append(r)

        best = min(r["cands"], key=lambda c: c["gemm"]) if r["cands"] else None
        shp = f"{r['cin']}->{r['cout']} @{r['hin']}" + ("" if r["stride"] == 1 else " s2")
        if best is None:
            print(f"{shp:20s} {r['freq']:3d} -- no legal candidate", flush=True)
            continue
        tag = f"{'x'.join(str(v) for v in best['tile'])} w{best['wgm']}"
        mm = r["mm"]
        print(
            f"{shp:20s} {r['freq']:3d} {r['M']:8d} {r['K']:5d} | "
            f"{r['default']:8.1f} {best['gemm']:8.1f} "
            f"{r['default'] / best['gemm']:5.2f}x {tag:>22s} | "
            f"{mm:8.1f} {mm / r['default']:6.2f}x {mm / best['gemm']:7.2f}x",
            flush=True,
        )

    OUT_JSON.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {OUT_JSON}")

    for path in ("1024", "1328"):
        sel = [r for r in rows if r["path"] == path and r["cands"]]
        if not sel:
            continue
        w_def = sum(r["default"] * r["freq"] for r in sel) / 1e3
        w_best = sum(min(c["gemm"] for c in r["cands"]) * r["freq"] for r in sel) / 1e3
        w_mm = sum(r["mm"] * r["freq"] for r in sel) / 1e3
        n_call = sum(r["freq"] for r in sel)
        print(
            f"\n{path} 路径 · {len(sel)} shapes / {n_call} 次调用\n"
            f"  GEMM 默认  {w_def:.2f} ms ({w_mm / w_def:.3f}x vs mm)\n"
            f"  GEMM 最优  {w_best:.2f} ms ({w_mm / w_best:.3f}x vs mm)   可省 {w_def - w_best:.2f} ms\n"
            f"  hipBLASLt  {w_mm:.2f} ms"
        )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--one"]:
        run_one(*sys.argv[2:])
    else:
        drive()
