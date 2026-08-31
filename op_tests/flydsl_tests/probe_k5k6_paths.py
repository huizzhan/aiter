# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which K5+K6 implementation should the GDN prefill pipeline use on this card?

``sweep_k5k6_compare.py`` answers a narrower question -- fused against fused,
opus ``gdn_k2_kernel`` against FlyDSL ``chunk_gdn_fwd_h_o_flydsl_vk_*`` -- and
on gfx950 FlyDSL wins it outright.  That result on its own is easy to misread as
a routing recommendation, because neither fused kernel is the fastest way to
cross this boundary on gfx950: FlyDSL's own *separate* K5 plus the Triton K6
beats FlyDSL's fused kernel there.

So this probe puts all four candidates on one footing.  Every number is the
profiler device time of the K5+K6 stage inside a full pipeline run, so the
front end -- which differs between opus and the FlyDSL/Triton path -- is
excluded on both sides and the columns are comparable.

seqlen is held at 8192 and only B*H varies, matching probe_fused_variant.py:
the grid sweep showed B*H is the only variable that moves these kernels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_gdn_block_ws_vs_flydsl as B
import sweep_k5k6_compare as S

# Profiler symbols that make up the K5+K6 stage on each path.  The FlyDSL
# separate K5 is ``chunk_gdn_fwd_h_flydsl_opt``; the fused one carries an
# extra ``_o_`` and a ``_vk_<variant>`` suffix, so the two do not collide.
PATHS = {
    "opus_wf": ("gdn_k2_kernel",),
    "opus_ws": ("chunk_gated_delta_rule_fwd_h_hip_kernel", "gdn_k2_out_kernel"),
    "fly_fused": ("chunk_gdn_fwd_h_o_flydsl",),
    "fly_sep": ("chunk_gdn_fwd_h_flydsl", "chunk_fwd_kernel_o"),
}
LABEL = {
    "opus_wf": "opus WF (fused)",
    "opus_ws": "opus WS (separate)",
    "fly_fused": "FlyDSL fused VK",
    "fly_sep": "FlyDSL sep K5 + Triton K6",
}


def flydsl_callable(t: dict, front: str, fusion: str):
    from aiter.ops.triton.gated_delta_net import chunk_gated_delta_rule_opt_vk

    def run():
        return chunk_gated_delta_rule_opt_vk(
            q=t["q"],
            k=t["k"],
            v=t["v"],
            g=t["g"],
            beta=t["beta"],
            initial_state=t["h0"],
            output_final_state=True,
            cu_seqlens=t["cu"],
            prefill_metadata=t["meta"],
            use_chunk_flydsl=True,
            use_prepare_flydsl=front == "flydsl",
            fusion=fusion,
        )

    return run


def stage_us(run, syms: tuple[str, ...]) -> tuple[float, dict]:
    run()
    torch.cuda.synchronize()
    kernels = B.profile_kernels(run)
    us = sum(v for n, v in kernels.items() if any(s in n for s in syms))
    return us, kernels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default=str(Path(__file__).with_name("k5k6_paths_gfx950.json"))
    )
    ap.add_argument("--front", choices=("auto", "flydsl", "triton"), default="auto")
    ap.add_argument("--seqlen", type=int, default=8192)
    args = ap.parse_args()

    front = args.front
    if front == "auto":
        try:
            import flydsl.expr.gpu as _fx_gpu

            front = "flydsl" if hasattr(_fx_gpu, "shuffle") else "triton"
        except ImportError:
            front = "triton"

    props = torch.cuda.get_device_properties(0)
    arch = props.gcnArchName.split(":")[0]
    cus = props.multi_processor_count

    out = {
        "gfx": props.gcnArchName,
        "cus": cus,
        "device": torch.cuda.get_device_name(0),
        "front": front,
        "seqlen": args.seqlen,
        "prof_iters": B.PROF_ITERS,
        "rows": [],
    }

    for tp, n_seqs in ((8, 1), (8, 2), (8, 4), (8, 8), (4, 8), (2, 8), (1, 8)):
        hg, h = 16 // tp, 64 // tp
        B.HK, B.HV, B.TP, B.HG, B.H = 16, 64, tp, hg, h
        B.FULL_PROMPT_LEN = args.seqlen
        t = B.build_inputs(n_seqs)
        bh = h * n_seqs
        want = S.cu_scaled_variant(bh, cus, arch=arch)

        row = {"tp": tp, "H": h, "n_seqs": n_seqs, "bh": bh, "variant": want, "us": {}}
        runs = {
            "opus_wf": lambda: B.make_callable("wf", t),
            "opus_ws": lambda: B.make_callable("ws", t),
            "fly_fused": lambda: S.fly_fixed_callable(t, want, front),
            "fly_sep": lambda: flydsl_callable(t, front, "never"),
        }
        for name, factory in runs.items():
            try:
                us, kernels = stage_us(factory(), PATHS[name])
                if us <= 0.0:
                    raise RuntimeError(
                        f"no {PATHS[name]} kernel ran: {sorted(kernels)}"
                    )
                row["us"][name] = us
            except Exception as exc:  # noqa: BLE001
                row.setdefault("errors", {})[name] = f"{type(exc).__name__}: {exc}"
            finally:
                torch.cuda.synchronize()

        ok = row["us"]
        if ok:
            best = min(ok, key=ok.get)
            row["best"] = best
            print(
                f"B*H={bh:5d} TP={tp} H={h:2d} B={n_seqs}  "
                + "  ".join(f"{k}={ok[k]:8.1f}" if k in ok else "" for k in PATHS)
                + f"   best={LABEL[best]}",
                flush=True,
            )
        else:
            print(f"B*H={bh:5d}  all failed: {row.get('errors')}", flush=True)
        out["rows"].append(row)

        del t
        torch.cuda.empty_cache()

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
