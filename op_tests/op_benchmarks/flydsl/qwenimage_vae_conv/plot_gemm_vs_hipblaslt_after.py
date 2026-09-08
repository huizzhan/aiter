#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Plot 口径 B after stage A of the 384 plan, from pick_ab.json.

Same axis as plot_gemm_vs_hipblaslt.py (hipBLASLt µs / FlyDSL GEMM µs), but the two
bar segments are the old and new *shipped* selection rather than default vs best-of-sweep
-- every number here is what the production path actually does.

Usage::

    python op_tests/op_benchmarks/flydsl/qwenimage_vae_conv/plot_gemm_vs_hipblaslt_after.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
WIN, LOSE, OLD_WIN, OLD_LOSE, NEUTRAL, GAIN = (
    "#2a6f6f",
    "#b44a3c",
    "#9ec9c9",
    "#e8b4ad",
    "#5c5c5c",
    "#1d4f4f",
)


def lab(r):
    conv = f"{r['cin']}→{r['cout']} @{r['hin']}²"
    if r["stride"] != 1:
        conv += " s2"
    return f"{conv}  {r['M']}×{r['N']}×{r['K']}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, default=HERE / "pick_ab.json")
    p.add_argument(
        "--out",
        type=Path,
        default=HERE / "figures" / "gemm_vs_hipblaslt_after.png",
    )
    args = p.parse_args()
    rows = [r for r in json.loads(args.json.read_text()) if r["mm"] == r["mm"]]

    plt.rcParams.update(
        {
            "font.sans-serif": ["WenQuanYi Zen Hei", "DejaVu Sans"],
            "font.size": 11,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 160,
        }
    )

    max_f = max(r["freq"] for r in rows)

    def bar_h(freq):
        return 0.22 + 0.72 * (freq / max_f)

    fig, ax = plt.subplots(figsize=(13.4, 10.8))
    gap = 0.22
    centers = []
    cursor = 0.0
    for r in rows:
        h = bar_h(r["freq"])
        cursor += h / 2
        centers.append(cursor)
        cursor += h / 2 + gap
    centers = np.array(centers)

    for yc, r in zip(centers, rows):
        h = bar_h(r["freq"])
        s_new = r["mm"] / r["new"]
        s_old = r["mm"] / r["old"]
        win = s_new >= 1
        # Longer segment first, so the shorter one stays visible on top of it.
        ax.barh(yc, max(s_new, s_old), height=h, color=WIN if win else LOSE, zorder=2)
        ax.barh(yc, min(s_new, s_old), height=h, color=OLD_WIN if win else OLD_LOSE, zorder=3)
        gained = s_new / s_old - 1
        note = f"{s_new:.2f}  ×{r['freq']}"
        if gained > 0.02:
            note += f"   +{100 * gained:.0f}%"
        ax.text(
            max(s_new, s_old) + 0.03,
            yc,
            note,
            va="center",
            fontsize=10,
            color=GAIN if gained > 0.02 else (WIN if win else LOSE),
            fontweight="bold" if gained > 0.02 else "normal",
            zorder=5,
        )

    ax.axvline(1.0, color=NEUTRAL, ls="--", lw=1.2, zorder=4)
    ax.set_yticks(centers)
    ax.set_yticklabels([lab(r) for r in rows], fontsize=10)
    ax.set_ylim(centers[-1] + bar_h(rows[-1]["freq"]) / 2 + gap, -gap)
    ax.set_xlabel(
        "GEMM 加速比  hipBLASLt on-device µs / FlyDSL GEMM kernel on-device µs"
    )
    ax.set_xlim(0, 3.35)
    ax.set_title(
        "阶段 A 后：出厂选核对标 hipBLASLt（条宽 = 调用次数；同 M/N/K，同 FLOP）"
    )
    fig.subplots_adjust(left=0.32)
    ax.legend(
        handles=[
            Patch(facecolor=WIN, label="新选核，≥1 跑赢 hipBLASLt"),
            Patch(facecolor=LOSE, label="新选核，<1 落后"),
            Patch(facecolor=OLD_WIN, label="旧选核（赢，浅青绿）"),
            Patch(facecolor=OLD_LOSE, label="旧选核（输，浅红）"),
        ],
        loc="center right",
        bbox_to_anchor=(1.0, 0.34),
        frameon=True,
        framealpha=0.95,
        edgecolor="none",
        fontsize=10,
    )

    w = lambda rs, k: sum(r[k] * r["freq"] for r in rs)
    p1024 = [r for r in rows if r["path"] == "1024"]
    fig.text(
        0.01,
        0.012,
        "两段都是出厂路径（非扫核最优）：深色为新 _pick_tile / _pick_wgm，浅色为旧 ladder 选核。"
        f"1024 路径加权 {w(p1024, 'mm') / w(p1024, 'old'):.3f}× → {w(p1024, 'mm') / w(p1024, 'new'):.3f}×。",
        color=NEUTRAL,
        fontsize=8,
    )
    fig.text(
        0.01,
        -0.004,
        "收益集中在 Cout=192（单 n-tile 宽 N tile）；384 家族仅 @256² 因 WGM 受益，@128²/@166² 无 tile 红利可取。"
        "非同类对比：hipBLASLt 读已物化的 M×K 矩阵，物化代价未计入。",
        color=NEUTRAL,
        fontsize=8,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
