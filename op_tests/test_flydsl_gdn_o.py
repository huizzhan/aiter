# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for the standalone FlyDSL GDN K6 output kernel.

Both K6 implementations are fed the SAME ``h`` snapshot and ungated ``v_new``,
produced by the pure-PyTorch K5 reference. That isolates K6: any difference is
the output stage's own, not K5's. The Triton ``chunk_fwd_o_opt_vk`` is the gold
standard, as it is for the fused kernel's tests.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import torch

from aiter.ops.flydsl.utils import is_flydsl_available

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL K6 tests.",
        allow_module_level=True,
    )

try:
    from aiter.ops.flydsl.gdn_o_kernels import chunk_fwd_o_flydsl, flydsl_k6_supported
    from aiter.ops.triton.gated_delta_net.gated_delta_rule import chunk_fwd_o_opt_vk
    from op_tests.gdn_common import (
        _make_inputs,
        _rmse_ratio,
        ref_chunk_gated_delta_rule_fwd_h,
    )
except ImportError as exc:  # pragma: no cover
    pytest.skip(
        f"Unable to import FlyDSL K6 dependencies: {exc}",
        allow_module_level=True,
    )

# Both kernels consume identical bf16 operands and accumulate in f32, so they
# should agree far more tightly than either agrees with an fp32 reference. The
# residual is MFMA-vs-tl.dot accumulation order plus one bf16 rounding of A.
_K6_RMSE_TOL = 2e-2


def _k6_operands(inp):
    """Pure-PyTorch K5 reference -> the bf16 (h, v_new) pair K6 consumes."""
    h_ref, v_new_ref, _ = ref_chunk_gated_delta_rule_fwd_h(
        k=inp["k"],
        w=inp["w_tm"],
        u=inp["u_tm"],
        g=inp["g_ref"],
        gk=inp["gk"],
        initial_state=inp["h0"],
        output_final_state=False,
        cu_seqlens=inp["cu"],
        g_head_major=True,
    )
    dtype = inp["u_tm"].dtype
    # K5 drains v_new token-major [B, T, H, V]; K6 reads it head-major.
    v_hm = v_new_ref.permute(0, 2, 1, 3).contiguous().to(dtype)
    return h_ref.to(dtype).contiguous(), v_hm


def _run_both(inp, *, scale, use_exp2, BV=None):
    h, v_hm = _k6_operands(inp)
    common = dict(
        q=inp["q"],
        k=inp["k"],
        v=v_hm,
        h=h,
        g=inp["g"],
        scale=scale,
        cu_seqlens=inp["cu"],
        use_exp2=use_exp2,
    )
    o_triton = inp["u_tm"].new_empty(inp["u_tm"].shape)
    chunk_fwd_o_opt_vk(o=o_triton, **common)

    o_flydsl = inp["u_tm"].new_empty(inp["u_tm"].shape)
    chunk_fwd_o_flydsl(o=o_flydsl, BV=BV, **common)
    return o_flydsl, o_triton


def _skip_if_unsupported(inp, BV):
    h = inp["h0"].to(inp["u_tm"].dtype)
    if not flydsl_k6_supported(q=inp["q"], h=h, K=inp["K"], V=inp["V"], BV=BV):
        pytest.skip("the FlyDSL K6 kernel does not support this device / shape")


@pytest.mark.parametrize("gate", ["g", "gk"])
@pytest.mark.parametrize(
    "H,Hg",
    [(12, 12), (4, 2), (24, 24)],  # MHA, GQA, MHA-wide
)
@pytest.mark.parametrize(
    "seq_lens",
    [
        [512],  # dense, BT-aligned
        [500],  # dense, tail chunk
        [512, 512],  # varlen, aligned
        [640, 384, 500],  # varlen, tail chunk on the last sequence
    ],
)
@pytest.mark.parametrize("BV", [32, 64, 128])
def test_k6_matches_triton(gate, H, Hg, seq_lens, BV):
    """FlyDSL K6 matches Triton K6 given identical h / v_new."""
    K = V = 128
    T_flat = sum(seq_lens)
    scale = K**-0.5
    use_exp2 = False  # _make_inputs builds natural-log gates

    inp = _make_inputs(H, Hg, K, V, T_flat, seq_lens, gate)
    _skip_if_unsupported(inp, BV)

    o_flydsl, o_triton = _run_both(inp, scale=scale, use_exp2=use_exp2, BV=BV)

    rmse = _rmse_ratio(o_flydsl, o_triton)
    assert rmse < _K6_RMSE_TOL, f"o RMSE ratio {rmse:.3e} >= {_K6_RMSE_TOL:.0e}"


@pytest.mark.parametrize("use_exp2", [False, True])
def test_k6_exp2(use_exp2):
    """The log2-scaled gate path agrees with Triton's."""
    H = Hg = 12
    K = V = 128
    seq_lens = [512, 384]
    scale = K**-0.5

    inp = _make_inputs(H, Hg, K, V, sum(seq_lens), seq_lens, "g")
    _skip_if_unsupported(inp, 64)
    if use_exp2:
        # Triton's USE_EXP2 reads g as already log2-scaled; scale both the
        # kernel gate and the reference's the same way so the comparison is of
        # exp2-vs-exp evaluation, not of two different gates.
        import math

        inp = dict(inp)
        inp["g"] = (inp["g"] * math.log2(math.e)).contiguous()

    o_flydsl, o_triton = _run_both(inp, scale=scale, use_exp2=use_exp2)
    rmse = _rmse_ratio(o_flydsl, o_triton)
    assert rmse < _K6_RMSE_TOL, f"o RMSE ratio {rmse:.3e} >= {_K6_RMSE_TOL:.0e}"


def test_k6_strong_decay():
    """A steep gate must not blow up the pair term.

    The pair gate exp(g_i - g_j) is only ever evaluated under the causal mask,
    where g_i <= g_j bounds it by 1. This is the regression guard for that: a
    telescoping formulation keyed on g_last would overflow f32 here.
    """
    H = Hg = 12
    K = V = 128
    seq_lens = [512]
    scale = K**-0.5

    inp = _make_inputs(H, Hg, K, V, sum(seq_lens), seq_lens, "g")
    _skip_if_unsupported(inp, 64)
    # ~-5 per token, so a 64-token chunk accumulates ~-320: exp(+320) is inf.
    inp = dict(inp)
    g = torch.randn(H, sum(seq_lens), dtype=torch.float32, device="cuda").abs() * -5.0
    inp["g"] = g.cumsum(dim=1).contiguous()
    inp["g_ref"] = inp["g"].unsqueeze(0).contiguous()

    o_flydsl, o_triton = _run_both(inp, scale=scale, use_exp2=False)
    assert torch.isfinite(o_flydsl).all(), "FlyDSL K6 produced non-finite output"
    rmse = _rmse_ratio(o_flydsl, o_triton)
    assert rmse < _K6_RMSE_TOL, f"o RMSE ratio {rmse:.3e} >= {_K6_RMSE_TOL:.0e}"


if __name__ == "__main__":
    torch.manual_seed(0)
    K = V = 128
    scale = K**-0.5
    rows = []
    for gate in ("g", "gk"):
        for H, Hg in ((12, 12), (4, 2)):
            for seq_lens in ([512], [500], [640, 384, 500]):
                for BV in (32, 64):
                    inp = _make_inputs(H, Hg, K, V, sum(seq_lens), seq_lens, gate)
                    try:
                        o_f, o_t = _run_both(inp, scale=scale, use_exp2=False, BV=BV)
                        rmse = _rmse_ratio(o_f, o_t)
                        status = "PASS" if rmse < _K6_RMSE_TOL else "FAIL"
                    except Exception as exc:  # noqa: BLE001
                        rmse, status = float("nan"), f"ERROR: {type(exc).__name__}"
                    rows.append((gate, f"{H}/{Hg}", str(seq_lens), BV, rmse, status))

    print(f"\n{'gate':>5} {'H/Hg':>6} {'seq_lens':>20} {'BV':>3} {'RMSE':>10} status")
    for gate, hhg, sl, bv, rmse, status in rows:
        print(f"{gate:>5} {hhg:>6} {sl:>20} {bv:>3} {rmse:10.3e} {status}")
    n_pass = sum(1 for r in rows if r[5] == "PASS")
    print(f"\n{n_pass}/{len(rows)} passed")
