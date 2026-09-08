#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Stage B gate: how much of the 384 gap vs hipBLASLt is kernel structure, not gather.

Stage A closed what the launch config could reach; `384->384` still runs at 0.68-0.88x
of hipBLASLt on the equivalent GEMM. This measures whether a structural change actually
moved the GEMM, by running the same kernel on a 1x1 conv with the SAME M/N/K -- same
tile, same wave grid, same K depth, but a dense contiguous A instead of a gather. A cut
that improves 3x3 without moving 1x1 is a gather or measurement effect, not structure.

The 1x1 arm cannot go through ``conv3d_implicit``: a stride-1 unpadded 1x1 is
short-circuited to ``torch.matmul`` there, which would measure hipBLASLt twice. It drives
``compile_conv3d_implicit`` directly instead, on a (1, 1, H, W, Cin*R*S) NDHWC input, and
picks its tile/WGM with the shipped ``_pick_tile`` / ``_pick_wgm`` so the probe isolates
structure rather than selection.

Three arms (3x3, 1x1, hipBLASLt ``mm``) are measured in the same process in alternating
rounds. Each shape gets a fresh process. Writes ``stage_b.json`` next to this script.

Usage::

    FLYDSL_ROOT=/workspace/FlyDSL python .../bench_stage_b.py
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
OUT_JSON = HERE / "stage_b.json"

# The per-cut set from the plan's measurement protocol: the three 384 layers stage A
# could not move, plus the two 96->96 layers whose bandwidth advantage must not be
# spent on them. sid, cin, cout, hin, freq, role
SHAPES = [
    ("mid_384_128", 384, 384, 128, 18, "target"),
    ("bottleneck_384_166", 384, 384, 166, 18, "target"),
    ("e2_res_384_256", 384, 384, 256, 8, "target"),
    ("e0_res_96_1024", 96, 96, 1024, 10, "guard"),
    ("d3_res_96_1328", 96, 96, 1328, 10, "guard"),
]

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


def run_one(sid, cin, cout, hin, freq, role) -> None:
    sys.path.insert(0, str(flydsl_root()))
    os.environ.setdefault("FLYDSL_CONV3D_AUTOTUNE", "0")

    import time as _time

    import torch
    import torch.nn.functional as F
    from kernels.conv import conv3d_implicit as mod
    from torch.profiler import ProfilerActivity, profile

    cin, cout, hin, freq = map(int, (cin, cout, hin, freq))
    dev = torch.device("cuda")
    torch.manual_seed(0)

    M, N, K = hin * hin, cout, cin * R * S
    assert K % 8 == 0, f"{sid}: K={K} is not a whole number of LDG vectors"

    tile = mod._pick_tile(M, cout, 1, dev)
    wgm = mod._pick_wgm(M, cout, 1, tile, dev)

    def gpu(call, want="conv3d_implicit_kernel"):
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

    # ---- 3x3 arm: the shipped path, gather included.
    x4 = torch.randn((1, cin, hin, hin), device=dev, dtype=torch.bfloat16)
    w4 = torch.randn((cout, cin, R, S), device=dev, dtype=torch.bfloat16)
    ref = F.conv2d(x4, w4, stride=1, padding=1)
    conv3x3 = partial(mod.conv3d_implicit, x4, w4, stride=1, padding=1)
    y = conv3x3()
    assert y.shape == ref.shape, f"{sid}: 3x3 {tuple(y.shape)} vs {tuple(ref.shape)}"
    e3 = ((y.float() - ref.float()).abs().mean() / ref.float().abs().mean()).item()
    assert e3 < 2e-2, f"{sid}: 3x3 rel err {e3:.2e}"
    del ref, y

    # ---- 1x1 arm: same M/N/K, dense A. Driven below conv3d_implicit on purpose; see
    # the module docstring. The NDHWC input IS the im2col matrix, so no gather happens.
    a1 = torch.randn((1, 1, hin, hin, K), device=dev, dtype=torch.bfloat16)
    w1 = torch.randn((cout, K), device=dev, dtype=torch.bfloat16)
    bias_arg = torch.empty(1, device=dev, dtype=torch.float32)
    y1 = torch.empty((1, cout, 1, hin, hin), device=dev, dtype=torch.bfloat16)
    exe = mod.compile_conv3d_implicit(
        1, K, 1, hin, hin, cout, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, "zeros", False, 1, tile, wgm, 1, False
    )

    def conv1x1():
        mod._dispatch(exe, y1, a1, w1, bias_arg, stream=torch.cuda.current_stream())

    conv1x1()
    torch.cuda.synchronize()
    ref1 = a1.reshape(M, K).float() @ w1.t().float()
    e1 = ((y1.reshape(cout, M).t().float() - ref1).abs().mean() / ref1.abs().mean()).item()
    assert e1 < 2e-2, f"{sid}: 1x1 rel err {e1:.2e} -- the direct dispatch is wired wrong"
    del ref1

    # ---- hipBLASLt on the same M/N/K, against the same B the 1x1 arm used.
    am = a1.reshape(M, K)
    cm = torch.empty((M, N), device=dev, dtype=torch.bfloat16)
    mm = partial(torch.mm, am, w1.t(), out=cm)
    mm()
    torch.cuda.synchronize()

    # ---- FlyDSL's own gfx950 bf16 GEMM on the same M/N/K. This separates "conv3d_implicit
    # is behind" from "FlyDSL is behind": whatever gap survives here is not conv-specific.
    # It rejects shapes it cannot tile (a K tail, say); those just report as missing.
    fly = None
    try:
        from kernels.gemm.gemm_a16w16_gfx950 import gemm_a16w16

        cg = torch.empty((M, N), device=dev, dtype=torch.bfloat16)
        fly = partial(gemm_a16w16, am, w1.t(), out=cg, layout="nt")
        fly()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001  an unsupported shape can fail anywhere from validate to launch
        print(f"# {sid}: gemm_a16w16 unavailable ({type(exc).__name__}: {exc})", file=sys.stderr)
        fly = None
    if fly is not None:
        eg = ((cg.float() - cm.float()).abs().mean() / cm.float().abs().mean()).item()
        assert eg < 2e-2, f"{sid}: gemm_a16w16 rel err {eg:.2e}"

    arms = [
        ("t3x3", conv3x3, "conv3d_implicit_kernel"),
        ("t1x1", conv1x1, "conv3d_implicit_kernel"),
        ("mm", mm, None),
    ]
    if fly is not None:
        arms.append(("flygemm", fly, None))
    best = dict.fromkeys([a[0] for a in arms])
    for i in range(ROUNDS):
        # Alternate the order every round so a drifting clock cannot favour one arm.
        for name, call, want in (arms[::-1] if i % 2 else arms):
            t = gpu(call, want)
            if math.isnan(t):
                continue
            best[name] = t if best[name] is None else min(best[name], t)

    print(
        "RESULT"
        + json.dumps(
            {
                "sid": sid,
                "cin": cin,
                "cout": cout,
                "hin": hin,
                "freq": freq,
                "role": role,
                "M": M,
                "N": N,
                "K": K,
                "tile": list(tile),
                "wgm": wgm,
                **{k: (NAN if v is None else v) for k, v in best.items()},
            }
        )
    )


def drive() -> None:
    env = child_env()
    cwd = str(flydsl_root())
    rows = []
    hdr = (
        f"{'shape':18s} {'x':>3s} {'tile':>13s} {'w':>2s} | {'mm':>8s} {'flygemm':>8s} {'1x1':>8s} "
        f"{'3x3':>8s} | {'fly/mm':>7s} {'1x1/mm':>7s} {'3x3/mm':>7s} {'gather':>7s}"
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
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT")), None)
        if line is None:
            print(f"FAILED {sp[0]}\n{proc.stderr[-2000:]}", flush=True)
            continue
        r = json.loads(line[len("RESULT") :])
        rows.append(r)
        shp = f"{r['cin']}->{r['cout']} @{r['hin']}"
        mm, fg = r["mm"], r.get("flygemm", NAN)
        has_fg = not math.isnan(fg)
        print(
            f"{shp:18s} {r['freq']:3d} {'x'.join(map(str, r['tile'])):>13s} {r['wgm']:2d} | "
            f"{mm:8.1f} {fg:8.1f} {r['t1x1']:8.1f} {r['t3x3']:8.1f} | "
            f"{f'{mm / fg:6.3f}x' if has_fg else '-':>7s} "
            f"{mm / r['t1x1']:6.3f}x {mm / r['t3x3']:6.3f}x {r['t3x3'] / r['t1x1']:6.3f}x",
            flush=True,
        )

    OUT_JSON.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {OUT_JSON}")

    # The gate: targets to >=0.90x on 1x1/mm, guards not worse than 5% vs their own history.
    tg = [r for r in rows if r["role"] == "target"]
    if tg:
        avg = sum(r["mm"] / r["t1x1"] for r in tg) / len(tg)
        print(f"\n目标 1x1/mm 均值 {avg:.3f}x（阶段 B 成功线 >= 0.90x）")
        for r in tg:
            fg = r.get("flygemm", NAN)
            head = f"  {r['cin']}->{r['cout']} @{r['hin']:4d}  {r['mm'] / r['t1x1']:.3f}x"
            print(head if math.isnan(fg) else f"{head}   (FlyDSL 自家 GEMM {r['mm'] / fg:.3f}x)")
    gd = [r for r in rows if r["role"] == "guard"]
    if gd:
        print("\n守门 3x3 GEMM（µs，只与本机上一刀比，不看 mm）：")
        for r in gd:
            print(f"  {r['cin']}->{r['cout']} @{r['hin']:4d}  {r['t3x3']:8.1f}  (x{r['freq']})")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--one"]:
        run_one(*sys.argv[2:])
    else:
        drive()
