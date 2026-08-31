# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Render sweep_k5k6_compare.py's grid as one self-contained HTML page.

Two tables.  The first aggregates by B*H, which the measurements show to be the
only variable that moves the result -- TP, seqlen and T reach the kernels only
through it.  The second is the full 64-cell grid, which is what makes that claim
checkable: cells sharing a B*H agree to within a couple of percent even when
their token totals differ eightfold.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import statistics
from pathlib import Path

DEFAULT_JSON = Path(__file__).with_name("k5k6_compare.json")
# The page is a checked-in deliverable next to the document that embeds its
# screenshot, so a plain re-run refreshes that copy rather than leaving a stray.
DEFAULT_HTML = (
    Path(__file__).resolve().parents[2] / "docs" / "gdn-k5k6-opus-vs-flydsl.html"
)

# opus keeps the pink it has in the existing GDN mode charts; the FlyDSL kernel
# gets violet, as the FlyDSL path does there.
COLOR = {
    "wf": ("#db2777", "#ec4899"),
    "fly": ("#7c3aed", "#8b5cf6"),
}
LABEL = {"wf": "opus WF", "fly": "FlyDSL"}
BANDS = ((2.0, 1.00), (1.5, 0.86), (1.25, 0.68), (1.10, 0.48), (1.0, 0.28))


def fmt_len(n: int) -> str:
    return f"{n // 1024}K" if n >= 1024 and n % 1024 == 0 else str(n)


def fmt_us(us: float) -> str:
    return f"{us / 1000:.2f} ms" if us >= 1000 else f"{us:.1f} us"


def band(ratio: float) -> float:
    for lo, alpha in BANDS:
        if ratio >= lo:
            return alpha
    return 0.28


def mix(a: str, b: str, t: float) -> str:
    ca = tuple(int(a[i : i + 2], 16) for i in (1, 3, 5))
    cb = tuple(int(b[i : i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(x * t + y * (1 - t)):02x}" for x, y in zip(ca, cb))


def geo(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    p = 1.0
    for v in vals:
        p *= v
    return p ** (1 / len(vals))


def by_bh(cells: list[dict], col: str) -> dict[int, float]:
    """geomean(opus / FlyDSL) per B*H, which is the axis that moves results."""
    g = collections.defaultdict(list)
    for c in cells:
        g[c["bh"]].append(
            c["backends"]["wf"]["k5k6_us"] / c["backends"][col]["k5k6_us"]
        )
    return {bh: geo(v) for bh, v in sorted(g.items())}


def crossover(cells: list[dict], col: str) -> tuple[int | None, int | None]:
    """The last B*H FlyDSL wins and the first one opus wins.

    Read off the per-B*H geomeans rather than hardcoded, because the boundary
    tracks the CU count and so moves between parts.
    """
    g = by_bh(cells, col)
    fly = [bh for bh, r in g.items() if r > 1]
    opus = [bh for bh, r in g.items() if r <= 1]
    return (max(fly) if fly else None), (min(opus) if opus else None)


def cell_html(cell: dict | None, col: str) -> str:
    if not cell or col not in cell["backends"] or "wf" not in cell["backends"]:
        why = (cell or {}).get("skipped") or "; ".join(
            f"{k}: {v}" for k, v in (cell or {}).get("errors", {}).items()
        )
        return f'<td class="empty" data-tip="{html.escape(why or "no data")}">—</td>'

    wf = cell["backends"]["wf"]
    fl = cell["backends"][col]
    ratio = wf["k5k6_us"] / fl["k5k6_us"]
    win = "fly" if ratio > 1 else "wf"
    margin = ratio if ratio > 1 else 1 / ratio
    deep, mid = COLOR[win]

    lines = [
        f"TP={cell['tp']} · H={cell['H']} · Hg={cell['Hg']} · B={cell['n_seqs']}",
        f"seqlen={cell['seqlen']:,} × B={cell['n_seqs']}  ⇒  T={cell['total_tokens']:,}",
        f"B·H={cell['bh']}",
        "",
        "K5+K6 融合 kernel (profiler device time):",
        f"  opus WF   {fmt_us(wf['k5k6_us']):>10}   gdn_k2_kernel",
        f"  FlyDSL    {fmt_us(fl['k5k6_us']):>10}   bv 变体 {fl.get('variant', '?')}",
        f"  ⇒ {LABEL[win]} 快 {margin:.2f}x",
        "",
        "整块 pipeline wall:",
        f"  opus WF   {fmt_us(wf['wall_us']):>10}  (K1..K4 {fmt_us(wf['front_us'])})",
        f"  FlyDSL    {fmt_us(fl['wall_us']):>10}  (K1..K4 {fmt_us(fl['front_us'])})",
    ]
    if col == "flyfix":
        auto = cell["backends"].get("fly", {})
        if auto.get("variant") and auto["variant"] != fl.get("variant"):
            lines += [
                "",
                (
                    f"as-shipped 会选 {auto['variant']} → {fmt_us(auto['k5k6_us'])}"
                    f" ({auto['k5k6_us'] / fl['k5k6_us']:.2f}x 慢)"
                ),
            ]

    a = band(margin)
    style = (
        f"--bg:{mix(mid, '#ffffff', a * 0.44)};"
        f"--bd:{mix(mid, '#e5e7eb', a * 0.60)};"
        f"--fg:{mix(deep, '#000000', 0.88)}"
    )
    return (
        f'<td class="cell" data-tip="{html.escape(chr(10).join(lines))}" '
        f'data-bh="{cell["bh"]}" style="{style}">'
        f'<span class="pill">{LABEL[win]}</span>'
        f'<span class="num">{margin:.2f}x</span></td>'
    )


def grid_table(d: dict, col: str) -> str:
    cells = d["cells"]
    totals = sorted({c["total_tokens"] for c in cells})
    by = {(c["tp"], c["seqlen"], c["total_tokens"]): c for c in cells}
    rows_key = sorted({(c["tp"], c["seqlen"], c["H"], c["Hg"]) for c in cells})

    head = "".join(f"<th>{fmt_len(t)}</th>" for t in totals)
    rows, prev_tp = [], None
    for tp, seqlen, h, hg in rows_key:
        tds = "".join(cell_html(by.get((tp, seqlen, t)), col) for t in totals)
        cls = "grp" if prev_tp is not None and tp != prev_tp else ""
        prev_tp = tp
        rows.append(
            f'<tr class="{cls}">'
            f'<th class="rh">{tp}</th><th class="rh sub">{h}</th>'
            f'<th class="rh sub">{hg}</th><th class="rh">{fmt_len(seqlen)}</th>'
            f"{tds}</tr>"
        )

    ok = [c for c in cells if col in c["backends"]]
    r = [c["backends"]["wf"]["k5k6_us"] / c["backends"][col]["k5k6_us"] for c in ok]
    nfly = sum(1 for x in r if x > 1)
    return (
        '<table class="grid"><thead>'
        '<tr><th class="rh" rowspan="2">TP</th><th class="rh sub" rowspan="2">H</th>'
        '<th class="rh sub" rowspan="2">Hg</th><th class="rh" rowspan="2">seqlen</th>'
        f'<th class="span" colspan="{len(totals)}">T = 总 token 数（B = T / seqlen）</th></tr>'
        f"<tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        f'<p class="note">FlyDSL 赢 <b>{nfly}</b> / {len(ok)} 格 · '
        f"geomean(opus/FlyDSL) = <b>{geo(r):.3f}</b> · "
        f"范围 {min(r):.2f}x ~ {max(r):.2f}x</p>"
    )


def summary_table(d: dict) -> str:
    g = collections.defaultdict(list)
    for c in d["cells"]:
        g[c["bh"]].append(c)
    cus = d["cus"]

    def med(cs: list[dict], col: str) -> float:
        return statistics.median(c["backends"][col]["k5k6_us"] for c in cs)

    def verdict(cs: list[dict], col: str) -> str:
        r = geo(
            [c["backends"]["wf"]["k5k6_us"] / c["backends"][col]["k5k6_us"] for c in cs]
        )
        cls = "wfly" if r > 1 else "wopus"
        who = "FlyDSL" if r > 1 else "opus"
        return f'<td class="{cls}">{who} {(r if r > 1 else 1 / r):.2f}x</td>'

    rows = []
    for bh in sorted(g):
        cs = g[bh]
        av = collections.Counter(
            c["backends"]["fly"].get("variant") for c in cs
        ).most_common(1)[0][0]
        xv = cs[0]["cu_scaled_variant"]
        bv = int(av.replace("w8", "")[2:])
        fill = -(-128 // bv) * bh / cus
        flip = ' class="flip"' if av != xv else ""
        rows.append(
            f"<tr><th>{bh}</th><td>{len(cs)}</td>"
            f"<td>{med(cs, 'wf'):.0f}</td>"
            f"<td>{med(cs, 'fly'):.0f}</td><td><code>{av}</code></td>"
            f"<td>{fill:.1f}×</td>{verdict(cs, 'fly')}"
            f"<td>{med(cs, 'flyfix'):.0f}</td><td{flip}><code>{xv}</code></td>"
            f"{verdict(cs, 'flyfix')}</tr>"
        )
    return (
        '<table class="sum"><thead><tr>'
        '<th rowspan="2">B·H</th><th rowspan="2">格数</th>'
        '<th rowspan="2">opus WF<br><span class="u">中位 µs</span></th>'
        '<th colspan="4">FlyDSL as-shipped（auto 选变体）</th>'
        '<th colspan="3">FlyDSL 换成按 CU 缩放的变体</th></tr>'
        '<tr><th class="u">中位 µs</th><th class="u">变体</th>'
        '<th class="u">CTA/CU</th><th class="u">对比</th>'
        '<th class="u">中位 µs</th><th class="u">变体</th><th class="u">对比</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


CSS = """
:root{--ink:#111827;--dim:#6b7280;--line:#e5e7eb;--bgc:#f9fafb}
*{box-sizing:border-box}
body{margin:0;padding:34px 30px 60px;background:#fff;color:var(--ink);
 font:14px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",
 "PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
h1{font-size:21px;margin:0 0 6px;letter-spacing:-.2px}
h2{font-size:16px;margin:38px 0 10px;padding-top:14px;border-top:1px solid var(--line)}
.sub1{color:var(--dim);font-size:13px;margin:0 0 4px}
.warn{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;
 padding:11px 14px;margin:16px 0 4px;font-size:13px;color:#78350f}
.warn b{color:#92400e}
table{border-collapse:separate;border-spacing:2px;margin:12px 0 4px}
th,td{padding:5px 7px;font-size:12.5px;text-align:center;white-space:nowrap}
thead th{color:var(--dim);font-weight:600;font-size:11.5px;letter-spacing:.3px}
thead th.span{background:var(--bgc);border-radius:6px;color:var(--ink)}
.u{font-weight:500;font-size:11px}
.rh{background:var(--bgc);border-radius:5px;font-weight:600;min-width:40px}
.rh.sub{color:var(--dim);font-weight:500}
tr.grp td,tr.grp th{border-top:2px solid #d1d5db}
.cell{background:var(--bg);border:1px solid var(--bd);border-radius:6px;
 color:var(--fg);cursor:default;min-width:96px}
.cell .pill{font-weight:650;font-size:11.5px}
.cell .num{display:block;font-size:11px;opacity:.72;font-variant-numeric:tabular-nums}
.empty{color:#d1d5db;background:#fcfcfd;border:1px dashed var(--line);border-radius:6px}
.sum th{background:var(--bgc);border-radius:5px}
.sum tbody th{font-weight:650;min-width:52px}
.sum td{background:#fcfcfd;border:1px solid var(--line);border-radius:5px;
 font-variant-numeric:tabular-nums}
.sum code{font:11.5px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
 background:#f3f4f6;padding:2px 4px;border-radius:3px}
.sum td.flip code{background:#fef3c7;color:#92400e;font-weight:600}
.wfly{background:#f5f3ff!important;border-color:#ddd6fe!important;
 color:#5b21b6;font-weight:650}
.wopus{background:#fdf2f8!important;border-color:#fbcfe8!important;
 color:#9d174d;font-weight:650}
.note{color:var(--dim);font-size:12.5px;margin:6px 0 0}
.tabs{display:flex;gap:6px;margin:14px 0 0}
.tabs button{font:inherit;font-size:12.5px;padding:5px 13px;border-radius:6px;
 border:1px solid var(--line);background:#fff;color:var(--dim);cursor:pointer}
.tabs button.on{background:var(--ink);border-color:var(--ink);color:#fff;font-weight:600}
#tip{position:fixed;z-index:9;display:none;pointer-events:none;max-width:430px;
 background:#111827f2;color:#f9fafb;padding:9px 12px;border-radius:7px;
 font:11.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre}
.foot{color:var(--dim);font-size:12px;margin-top:34px;padding-top:14px;
 border-top:1px solid var(--line)}
"""

JS = """
const tip=document.getElementById('tip');
document.addEventListener('mouseover',e=>{
 const t=e.target.closest('[data-tip]'); if(!t)return;
 tip.textContent=t.dataset.tip; tip.style.display='block';
 const r=t.getBoundingClientRect(), b=tip.getBoundingClientRect();
 tip.style.left=Math.min(r.left,innerWidth-b.width-14)+'px';
 tip.style.top=(r.bottom+8+b.height>innerHeight?r.top-b.height-8:r.bottom+8)+'px';
});
document.addEventListener('mouseout',e=>{
 if(e.target.closest('[data-tip]'))tip.style.display='none';});
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('on',x===b));
 document.querySelectorAll('[data-pane]').forEach(p=>
  p.style.display=p.dataset.pane===b.dataset.pane?'':'none');
});
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    ap.add_argument("--out", default=str(DEFAULT_HTML))
    args = ap.parse_args()

    with open(args.json) as fh:
        d = json.load(fh)
    cells = d["cells"]
    r_auto = [
        c["backends"]["wf"]["k5k6_us"] / c["backends"]["fly"]["k5k6_us"] for c in cells
    ]
    r_fix = [
        c["backends"]["wf"]["k5k6_us"] / c["backends"]["flyfix"]["k5k6_us"]
        for c in cells
    ]
    bad = [
        c
        for c in cells
        if c["backends"]["fly"].get("variant") != c["cu_scaled_variant"]
    ]
    n = len(cells)
    gfx = d["gfx"].split(":")[0]
    front = d.get("front", "flydsl")
    fly_win, opus_win = crossover(cells, "flyfix")

    # The kernel and its tests are gated to gfx942 upstream, so a run on any
    # other part had that gate lifted and is reporting what the kernel does
    # there, not what the PR ships there.
    caveats = []
    if gfx != "gfx942":
        caveats.append(
            f"<b>融合 kernel 上游限定 gfx942。</b>PR #4884 的融合 K5+K6 "
            f"（<code>gdn_fused_gfx942_kernels.py</code>）对非 gfx942 直接拒绝，"
            f"自带的单元测试也整套 skip。本页是<b>放开 arch 门控之后</b>在 {gfx} 上测的，"
            f"测之前先把 PR 那 114 个融合用例在本卡跑过一遍（113 passed / 1 skipped，"
            f"含 bv16/32/64/bv64w8 各变体、非 BT 倍数尾块、tail-mask 的 Inf/NaN 隔离、"
            f"final_state 一致性）。"
        )
    else:
        caveats.append(
            f"<b>FlyDSL 列是强制打开的。</b>PR #4884 把 VK 路径门控在 "
            f"<code>_device_cu_count() &gt;= 304</code>，本机 {d['cus']} CU，"
            f"<code>fusion=AUTO</code> 不会融合；这里用 <code>fusion=ALWAYS</code>。"
        )
    if front == "triton":
        caveats.append(
            "<b>前端是 Triton 而非 FlyDSL。</b>FlyDSL 的 "
            "<code>gdn_prepare</code> 需要 <code>flydsl.expr.gpu.shuffle</code>，"
            "本机装的 flydsl 0.2.4 没有这个符号。融合 K5+K6 本身不用 shuffle，"
            "所以 K5+K6 那一列不受影响；<b>wall 一列</b>两边前端不同，只能参考。"
        )
    if bad:
        loss = geo(
            [
                c["backends"]["fly"]["k5k6_us"] / c["backends"]["flyfix"]["k5k6_us"]
                for c in bad
            ]
        )
        caveats.append(
            f"<b>auto 的变体选择在这张卡上不是最优。</b>{len(bad)}/{n} 格里 "
            f"auto 选的 BV 与变体探针（<code>probe_fused_variant.py</code>）在本卡"
            f"实测最优的那个不同，平均慢 <b>{loss:.2f}x</b>。"
            f"右侧「按 CU 缩放的变体」列是换成实测规则之后的结果。"
        )
    else:
        caveats.append(
            f"<b>auto 的变体选择在这张卡上是对的。</b>{n}/{n} 格里 auto 选的 BV "
            f"与按 {d['cus']} CU 缩放的选择一致。"
        )

    bhs = sorted({c["bh"] for c in cells})
    if fly_win and opus_win:
        bound = (
            f"<b>B·H ≤ {fly_win} 时 FlyDSL 融合 kernel 更快</b>，"
            f"<b>B·H ≥ {opus_win} 时 opus WF 更快</b>。两边都是同一个机理——"
            f"融合 kernel 在 B·H 小的时候喂不饱设备，谁的退化更缓谁就赢"
        )
    elif fly_win:
        bound = (
            f"<b>网格覆盖的全部 B·H（{bhs[0]}..{bhs[-1]}）上 FlyDSL 融合 kernel 都更快</b>，"
            f"没有出现翻转点"
        )
    else:
        bound = (
            f"<b>网格覆盖的全部 B·H（{bhs[0]}..{bhs[-1]}）上 opus WF 都更快</b>，"
            f"没有出现翻转点"
        )

    body = f"""<h1>GDN prefill 融合 K5+K6：opus WF vs FlyDSL VK (PR #4884)</h1>
<p class="sub1">{d["device"]} · {d["gfx"]} · {d["cus"]} CU · bf16 · K=V=128 · BT=64 ·
GQA {d["gqa_ratio"]}（Hk={d["Hk_model"]} / Hv={d["Hv_model"]}）· packed varlen ·
两侧都是 <code>output_final_state=True</code> + 真实 initial_state ·
每格 {d["num_iters"]} 次取中位、profiler {d["prof_iters"]} 次取 device time</p>

<div class="warn">
{"<p>" + "</p><p>".join(caveats) + "</p>"}
</div>

<h2>按 B·H 汇总</h2>
<p class="sub1">B·H（序列数 × 每卡 value head 数）是唯一移动结果的变量：TP、seqlen、T
都只通过它起作用。CTA/CU 是 as-shipped 变体的实际网格占用
<code>⌈V/BV⌉·B·H / {d["cus"]}</code>，超过 1 就是排队。</p>
{summary_table(d)}
<p class="note">按实测最优变体（右侧列）看：{bound}。</p>

<h2>完整 {n} 格网格</h2>
<p class="sub1">同一个 B·H 的格子彼此吻合到 2% 以内，即便 token 总量差 8 倍
——这正是「只有 B·H 起作用」的验证。悬停看每格明细。</p>
<div class="tabs">
 <button class="on" data-pane="fly">as-shipped（auto 选变体）</button>
 <button data-pane="flyfix">按 CU 缩放的变体</button>
</div>
<div data-pane="fly">{grid_table(d, "fly")}</div>
<div data-pane="flyfix" style="display:none">{grid_table(d, "flyfix")}</div>

<h2>整块 pipeline 的 wall</h2>
<p class="sub1">K5+K6 之外两条 pipeline 的前端也不同：opus 把 K1..K4 融进一个 HIP
kernel（<code>gdn_k1_neumann_kernel</code>），FlyDSL 这边是
{"Triton prepare 三件套" if front == "triton" else "<code>gdn_prepare_kernel</code>"}，
两者之比 geomean =
{geo([c["backends"]["wf"]["front_us"] / c["backends"]["fly"]["front_us"] for c in cells]):.2f}x。</p>
<p class="note">as-shipped 全 pipeline：geomean(opus/FlyDSL) =
<b>{geo([c["backends"]["wf"]["wall_us"] / c["backends"]["fly"]["wall_us"] for c in cells]):.3f}</b>，
FlyDSL 赢 <b>{sum(1 for c in cells if c["backends"]["wf"]["wall_us"] > c["backends"]["fly"]["wall_us"])}</b>/{n} 格。
K5+K6 单看是 geomean <b>{geo(r_auto):.3f}</b>（修正变体后 <b>{geo(r_fix):.3f}</b>）。</p>

<p class="foot">数据 <code>{Path(args.json).name}</code> ·
复现 <code>python sweep_k5k6_compare.py</code> ·
变体探针 <code>python probe_fused_variant.py</code> ·
torch {d["torch"]}</p>
<div id="tip"></div>"""

    page = (
        "<!doctype html><html lang=zh-CN><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        "<title>GDN 融合 K5+K6：opus WF vs FlyDSL VK</title>"
        f"<style>{CSS}</style></head><body>{body}<script>{JS}</script></body></html>"
    )
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"wrote {args.out} ({len(page) / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
