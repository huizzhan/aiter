#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A/B the shipped tile/WGM selection against the previous ladder-only selection.

Verification gate for stage A of the 384 plan: the new ``_pick_tile`` / ``_pick_wgm``
must not regress any shape. "Old" is reproduced by forcing the tile the ladder-only
heuristic would have returned, with wgm=1 -- so both arms run the same kernel source
and differ only in the launch config.

hipBLASLt on the equivalent M/N/K is measured in the same rounds, so the 口径 B ratio
comes out of one interleaved run rather than a cross-batch join.

All arms are measured in the same process, alternating, min over rounds. Each shape
gets a fresh process. Writes ``pick_ab.json`` next to this script.

Usage::

    FLYDSL_ROOT=/workspace/FlyDSL python .../bench_pick_ab.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "pick_ab.json"

from bench_tile_wgm import SHAPES

R = S = 3
ITERS, ROUNDS = 10, 3
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


def old_pick_tile(npq, kg, groups, num_cu, ladder, min_fill, min_waves):
    """The pre-change selection: ladder only, no wide-N case, wgm always 1."""
    legal = [t for t in ladder if kg >= t[1] * min_fill] or [ladder[-1]]
    target = min_waves * num_cu
    for tile_m, tile_n, wave_m, wave_n in legal:
        blocks = ((npq + tile_m - 1) // tile_m) * groups * ((kg + tile_n - 1) // tile_n)
        if blocks * wave_m * wave_n >= target:
            return (tile_m, tile_n, wave_m, wave_n)
    return legal[-1]


def run_one(sid, cin, cout, hin, stride, padding, freq, path) -> None:
    sys.path.insert(0, str(flydsl_root()))
    os.environ.setdefault("FLYDSL_CONV3D_AUTOTUNE", "0")

    import time as _time

    import torch
    import torch.nn.functional as F
    from kernels.conv import conv3d_implicit as mod
    from torch.profiler import ProfilerActivity, profile

    cin, cout, hin, stride, padding, freq = map(
        int, (cin, cout, hin, stride, padding, freq)
    )

    torch.manual_seed(0)
    x4 = torch.randn((1, cin, hin, hin), device="cuda", dtype=torch.bfloat16)
    w4 = torch.randn((cout, cin, R, S), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((cout,), device="cuda", dtype=torch.float32)
    bbf = b.to(torch.bfloat16)
    ref = F.conv2d(x4, w4, bias=bbf, stride=stride, padding=padding)
    p = ref.shape[2]
    M, N, K = p * p, cout, cin * R * S

    num_cu = mod._num_cu(x4.device)
    old_tile = old_pick_tile(
        M, cout, 1, num_cu, mod.TILE_LADDER, mod.TILE_MIN_N_FILL, mod.TILE_MIN_WAVES_PER_CU
    )
    new_tile = mod._pick_tile(M, cout, 1, x4.device)
    new_wgm = mod._pick_wgm(M, cout, 1, new_tile, x4.device)

    def new_call():
        return mod.conv3d_implicit(x4, w4, bias=b, stride=stride, padding=padding)

    def old_call():
        return mod.conv3d_implicit(
            x4, w4, bias=b, stride=stride, padding=padding, tile=old_tile, wgm=1
        )

    for y in (new_call(), old_call()):
        assert y.shape == ref.shape, f"{sid}: {tuple(y.shape)} vs {tuple(ref.shape)}"
        e = ((y.float() - ref.float()).abs().mean() / ref.float().abs().mean()).item()
        assert e < 2e-2, f"{sid}: rel err {e:.2e}"

    def gemm_us(call, want="conv3d_implicit_kernel"):
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
                e.device_type == torch.autograd.DeviceType.CUDA
                and e.self_device_time_total > 0
                and (want is None or want in e.key)
            ):
                t += e.self_device_time_total / ITERS
        return t if t > 0 else NAN

    # hipBLASLt on the equivalent GEMM. Skipped (NaN) when the M x K operand does not fit.
    mm_call = None
    try:
        A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        Bm = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
        Cm = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

        def mm_call():
            return torch.mm(A, Bm, out=Cm)

        mm_call()
        torch.cuda.synchronize()
    except torch.OutOfMemoryError:
        mm_call = None
        torch.cuda.empty_cache()

    best = {"new": None, "old": None, "mm": None}
    for i in range(ROUNDS):
        # Alternate the order every round so a drifting clock cannot favour one arm.
        arms = [("new", new_call, "conv3d_implicit_kernel"), ("old", old_call, "conv3d_implicit_kernel")]
        if mm_call is not None:
            arms.append(("mm", mm_call, None))
        if i % 2:
            arms.reverse()
        for name, call, want in arms:
            t = gemm_us(call, want)
            if math.isnan(t):
                continue
            best[name] = t if best[name] is None else min(best[name], t)
    t_new, t_old, t_mm = best["new"], best["old"], best["mm"]

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
                "flops": 2.0 * M * N * K,
                "old_tile": list(old_tile),
                "new_tile": list(new_tile),
                "new_wgm": new_wgm,
                "old": NAN if t_old is None else t_old,
                "new": NAN if t_new is None else t_new,
                "mm": NAN if t_mm is None else t_mm,
            }
        )
    )


def drive() -> None:
    env = child_env()
    cwd = str(flydsl_root())
    rows = []
    hdr = (
        f"{'shape':20s} {'x':>3s} {'old tile':>13s} {'new tile':>13s} {'w':>2s} | "
        f"{'old us':>8s} {'new us':>8s} {'delta':>7s} {'ratio':>6s} | "
        f"{'mm us':>8s} {'old/mm':>7s} {'new/mm':>7s}"
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
        shp = f"{r['cin']}->{r['cout']} @{r['hin']}" + ("" if r["stride"] == 1 else " s2")
        ratio = r["old"] / r["new"]
        flag = "  <-- REGRESSION" if ratio < 0.97 else ""
        mm = r["mm"]
        mm_cols = (
            f"{mm:8.1f} {mm / r['old']:6.2f}x {mm / r['new']:6.2f}x"
            if not math.isnan(mm)
            else f"{'-':>8s} {'-':>7s} {'-':>7s}"
        )
        print(
            f"{shp:20s} {r['freq']:3d} {'x'.join(map(str, r['old_tile'])):>13s} "
            f"{'x'.join(map(str, r['new_tile'])):>13s} {r['new_wgm']:2d} | "
            f"{r['old']:8.1f} {r['new']:8.1f} {r['new'] - r['old']:+7.1f} {ratio:5.2f}x | "
            f"{mm_cols}{flag}",
            flush=True,
        )

    OUT_JSON.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {OUT_JSON}")

    for path in ("1024", "1328"):
        sel = [r for r in rows if r["path"] == path]
        if not sel:
            continue
        w_old = sum(r["old"] * r["freq"] for r in sel) / 1e3
        w_new = sum(r["new"] * r["freq"] for r in sel) / 1e3
        mmsel = [r for r in sel if not math.isnan(r["mm"])]
        w_mm = sum(r["mm"] * r["freq"] for r in mmsel) / 1e3
        w_old_m = sum(r["old"] * r["freq"] for r in mmsel) / 1e3
        w_new_m = sum(r["new"] * r["freq"] for r in mmsel) / 1e3
        print(
            f"\n{path} 路径 · {len(sel)} shapes / {sum(r['freq'] for r in sel)} 次调用\n"
            f"  GEMM 旧选核 {w_old:.2f} ms   新选核 {w_new:.2f} ms   "
            f"省 {w_old - w_new:+.2f} ms ({w_old / w_new:.3f}x)\n"
            f"  vs hipBLASLt {w_mm:.2f} ms:  旧 {w_mm / w_old_m:.3f}x -> 新 {w_mm / w_new_m:.3f}x"
        )
    worst = min(rows, key=lambda r: r["old"] / r["new"]) if rows else None
    if worst:
        print(
            f"\n最差 shape: {worst['cin']}->{worst['cout']} @{worst['hin']} "
            f"{worst['old'] / worst['new']:.3f}x (x{worst['freq']})"
        )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--one"]:
        run_one(*sys.argv[2:])
    else:
        drive()
