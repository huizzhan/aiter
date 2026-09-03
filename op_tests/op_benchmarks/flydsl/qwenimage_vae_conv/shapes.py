# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen-Image VAE conv shapes for one T2I encode+decode at 1024 / 1328.

Each tuple is (sid, cin, cout, hin, stride, padding, freq, path).
``hin`` is the *input* spatial size. ``freq`` is hook call count for that
resolution. ``path`` is ``1024`` or ``1328``.
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

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


def flydsl_root() -> Path:
    return Path(os.environ.get("FLYDSL_ROOT", "/workspace/FlyDSL")).resolve()


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    root = str(flydsl_root())
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env["FLYDSL_CONV3D_AUTOTUNE"] = "0"
    return env
