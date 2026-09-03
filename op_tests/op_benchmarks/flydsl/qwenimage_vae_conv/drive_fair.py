#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Aggregate the three-tier fair comparison over the 18 VAE conv shapes.

Each shape runs in a fresh process (``bench_one_fair.py``) so JIT / autotune
state cannot leak. Writes ``fair_baseline.json`` next to this script.

Usage::

    FLYDSL_ROOT=/path/to/FlyDSL python op_tests/op_benchmarks/flydsl/qwenimage_vae_conv/drive_fair.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from shapes import SHAPES, child_env, flydsl_root  # noqa: E402

FIELDS = [
    ("sid", str),
    ("cin", int),
    ("cout", int),
    ("hin", int),
    ("stride", int),
    ("freq", int),
    ("path", str),
    ("M", int),
    ("N", int),
    ("K", int),
    ("flops", float),
    ("fly_def", float),
    ("gemm_def", float),
    ("tr_def", float),
    ("fly_best", float),
    ("gemm_best", float),
    ("tr_best", float),
    ("tile", str),
    ("mm", float),
    ("im2col", float),
    ("unfold_mm", float),
    ("mio", float),
]

BENCH = HERE / "bench_one_fair.py"
OUT_JSON = HERE / "fair_baseline.json"


def main() -> None:
    env = child_env()
    cwd = str(flydsl_root())
    rows = []
    print(
        f"{'shape':18s} {'x':>3s} {'M':>8s} {'K':>5s} | {'hipBLASLt':>9s} {'FlyGEMM':>8s} {'比':>5s} "
        f"{'BLASt T/s':>9s} {'Fly T/s':>8s} | {'im2col':>7s} {'unfold+mm':>10s} {'Fly 全':>7s} {'MIOpen':>8s} {'最优tile':>13s}"
    )
    for sp in SHAPES:
        proc = subprocess.run(
            [sys.executable, str(BENCH), *[str(v) for v in sp]],
            capture_output=True,
            text=True,
            env=env,
            cwd=cwd,
        )
        line = next(
            (l for l in proc.stdout.splitlines() if l.startswith("RESULT")), None
        )
        if line is None:
            print(f"FAILED {sp[0]}\n{proc.stderr[-800:]}")
            continue
        vals = line.split("\t")[1:]
        r = {name: cast(v) for (name, cast), v in zip(FIELDS, vals)}
        rows.append(r)
        shp = f"{r['cin']}->{r['cout']} @{r['hin']}" + (
            "" if r["stride"] == 1 else " s2"
        )
        print(
            f"{shp:18s} {r['freq']:3d} {r['M']:8d} {r['K']:5d} | {r['mm']:9.1f} {r['gemm_best']:8.1f} "
            f"{r['mm'] / r['gemm_best']:4.2f}x {r['flops'] / r['mm'] / 1e6:9.0f} "
            f"{r['flops'] / r['gemm_best'] / 1e6:8.0f} | {r['im2col']:7.1f} {r['unfold_mm']:10.1f} "
            f"{r['fly_best']:7.1f} {r['mio']:8.1f} {r['tile']:>13s}",
            flush=True,
        )

    OUT_JSON.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {OUT_JSON}")

    for path in ("1024", "1328"):
        sel = [r for r in rows if r["path"] == path]
        if not sel:
            continue
        W = lambda k: sum(r[k] * r["freq"] for r in sel) / 1e3  # noqa: E731
        fl = sum(r["flops"] * r["freq"] for r in sel)
        print(
            f"\n{'=' * 78}\n{path} 路径 · {len(sel)} shapes / {sum(r['freq'] for r in sel)} 次调用"
            f" · 等价 GEMM 共 {fl / 1e12:.2f} TFLOP\n{'=' * 78}"
        )
        print("A 端到端（严格同类：同输入同输出，各付全部成本）")
        print(f"  unfold + mm (hipBLASLt)   {W('unfold_mm'):7.2f} ms")
        print(
            f"  F.conv2d (MIOpen)         {W('mio'):7.2f} ms   {W('unfold_mm') / W('mio'):5.2f}x vs unfold+mm"
        )
        print(
            f"  FlyDSL 默认 tile          {W('fly_def'):7.2f} ms   {W('mio') / W('fly_def'):5.3f}x vs MIOpen"
        )
        print(
            f"  FlyDSL 最优 tile          {W('fly_best'):7.2f} ms   {W('mio') / W('fly_best'):5.3f}x vs MIOpen"
        )
        print("\nB GEMM 质量（非同类：hipBLASLt 读物化矩阵，FlyDSL gather 原张量）")
        print(
            f"  hipBLASLt mm              {W('mm'):7.2f} ms   {fl / W('mm') / 1e9:6.0f} TFLOP/s"
        )
        print(
            f"  FlyDSL GEMM 默认 tile     {W('gemm_def'):7.2f} ms   {fl / W('gemm_def') / 1e9:6.0f} TFLOP/s"
            f"   {W('mm') / W('gemm_def'):5.3f}x"
        )
        print(
            f"  FlyDSL GEMM 最优 tile     {W('gemm_best'):7.2f} ms   {fl / W('gemm_best') / 1e9:6.0f} TFLOP/s"
            f"   {W('mm') / W('gemm_best'):5.3f}x"
        )
        print(
            f"  hipBLASLt 跳过的 im2col   {W('im2col'):7.2f} ms   (若计入则 {W('mm') + W('im2col'):.2f} ms,"
            f" {(W('mm') + W('im2col')) / W('gemm_best'):.3f}x)"
        )
        print("\nC 设计上限（转置零成本时的 FlyDSL）")
        print(
            f"  FlyDSL 最优 tile 无转置   {W('gemm_best'):7.2f} ms   {W('mio') / W('gemm_best'):5.3f}x vs MIOpen,"
            f" {W('mm') / W('gemm_best'):5.3f}x vs hipBLASLt"
        )
        print(
            f"  转置现占 FlyDSL           {W('tr_best'):7.2f} ms   ({100 * W('tr_best') / W('fly_best'):.0f}%)"
        )
        print(
            f"\n  vs hipBLASLt 赢的 shape: GEMM only {sum(1 for r in sel if r['mm'] >= r['gemm_best'])}/{len(sel)}"
            f", 含转置 {sum(1 for r in sel if r['mm'] >= r['fly_best'])}/{len(sel)}"
        )


if __name__ == "__main__":
    main()
