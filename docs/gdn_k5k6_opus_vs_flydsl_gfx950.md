# GDN prefill 融合 K5+K6：opus WF vs FlyDSL VK，gfx950 (PR #4884)

AMD Radeon Graphics · gfx950 · 256 CU · bf16 · K=V=128 · BT=64 · GQA 4（Hk=16 / Hv=64）
· packed varlen · 2026-08-28

这是 `gdn_k5k6_opus_vs_flydsl.md`（MI308X · gfx942 · 80 CU）的同口径重测。对比的还是两个
**融合同一个边界**的实现：inter-chunk 状态扫描（K5）和输出投影（K6）在一次 dispatch 里完成。

| | opus WF | FlyDSL VK（PR #4884） |
| --- | --- | --- |
| 融合 kernel | `gdn_k2_kernel`（`gdn_k2_fused_traits`） | `chunk_gdn_fwd_h_o_flydsl_vk_{bv16,bv32,bv64,bv64w8}` |
| 入口 | `opus_gdn_wu_prefill_fwd(k2_mode=OPUS_GDN_K2_WU_FUSED)` | `chunk_gated_delta_rule_fwd_h_o_flydsl`，或 `chunk_gated_delta_rule_opt_vk(use_chunk_flydsl=True, fusion="always")` |

---

## 1. 结论

**换到 gfx950，胜负整体翻了过来。** gfx942 上 opus WF 从 B·H=64 起反超并一直领先到
B·H=4096；gfx950 上这个反超点消失了——只要选对 BV 变体，**FlyDSL 融合 kernel 在 64 格里
全部 64 格都更快**，幅度 1.15–2.28x（geomean 1.461）。

按 B·H 看融合 kernel 的 device time（右两列是按本卡实测最优变体，见第 4 节）：

| B·H | 格数 | opus WF | FlyDSL as-shipped | 谁快 | FlyDSL 最优变体 | 谁快 |
| ---: | ---: | ---: | ---: | :--- | ---: | :--- |
| 8 | 1 | 767 µs | 333 µs (`bv16`) | **FlyDSL 2.30x** | 336 µs (`bv16`) | **FlyDSL 2.28x** |
| 16 | 3 | 767 | 331 (`bv16`) | **FlyDSL 2.30x** | 335 (`bv16`) | **FlyDSL 2.28x** |
| 32 | 6 | 574 | 322 (`bv32`) | **FlyDSL 1.80x** | 254 (`bv16`) | **FlyDSL 2.25x** |
| 64 | 10 | 390 | 289 (`bv64`) | **FlyDSL 1.42x** | 212 (`bv16`) | **FlyDSL 1.78x** |
| 128 | 12 | 442 | 295 (`bv64w8`) | **FlyDSL 1.52x** | 298 (`bv32`) | **FlyDSL 1.48x** |
| 256 | 12 | 250 | 301 (`bv64w8`) | opus 1.21x | 204 (`bv64`) | **FlyDSL 1.19x** |
| 512 | 10 | 493 | 599 (`bv64w8`) | opus 1.18x | 394 (`bv64`) | **FlyDSL 1.21x** |
| 1024 | 6 | 777 | 904 (`bv64w8`) | opus 1.18x | 662 (`bv64`) | **FlyDSL 1.19x** |
| 2048 | 3 | 1087 | 1225 (`bv64w8`) | opus 1.12x | 932 (`bv64`) | **FlyDSL 1.16x** |
| 4096 | 1 | 2175 | 2419 (`bv64w8`) | opus 1.11x | 1885 (`bv64`) | **FlyDSL 1.15x** |

（µs 是该 B·H 下所有格子的中位数；不同格子的 seqlen 不同，所以列内绝对值不可横向比较，
比值是逐格算出来后取的 geomean。）

| 口径 | geomean(opus/FlyDSL) | FlyDSL 赢 |
| --- | ---: | ---: |
| K5+K6 融合 kernel，as-shipped 变体 | 1.170 | 32 / 64 |
| K5+K6 融合 kernel，本卡实测最优变体 | **1.461** | **64 / 64** |

as-shipped 只赢一半，不是 kernel 的问题，是变体选择的问题：60/64 格选到了非最优的 BV，
平均慢 1.27x，正好把 B·H ≥ 256 那半边的胜负颠倒过来（见第 4 节）。

**但「FlyDSL 融合赢 opus 融合」不等于「gfx950 上就该用融合」。** 把两边的分离实现也放进来
之后，最优解在低 B·H 段并不是融合 kernel（见第 5 节）：**B·H ≤ 16 用 FlyDSL 分离 K5 +
Triton K6，B·H ≥ 32 用 FlyDSL 融合**，而 opus 的两种模式在任何 B·H 都不是最优。生产上关心
的那一点——TP=8、H=8、B=1、seqlen=8192，即 B·H=8——最快的是 FlyDSL 分离路径
（190 µs），比 opus WS（229 µs，本卡当前默认落点）快 1.21x，比 opus WF 快 4.04x。

---

## 2. 三件影响解读的口径问题

**融合 kernel 上游限定 gfx942，本页是放开门控之后测的。** PR #4884 的融合 K5+K6
（`aiter/ops/flydsl/gdn_fused_gfx942_kernels.py`）有两道 arch 门控：
`is_fused_k5k6_gfx942_unsupported()` 对非 gfx942 直接返回不支持，
`chunk_gated_delta_rule_fwd_h_o_flydsl` 入口再抛一次 `NotImplementedError`；
`op_tests/test_flydsl_gdn_fused_k5k6.py` 的每个用例也都 `pytest.skip`。PR 最新版
（本文测的是 cherry-pick 到本分支的 7 个新提交，含 `Fix gfx950 bug for sequence lengths
that are not multiples of chunk size`）给 gfx950 加的是**分离 K5（VK K5）**的支持，融合
K5+K6 仍是 gfx942 专属。本页把这两道门控放开到 `("gfx942", "gfx950")` 之后测，测之前先把
PR 自带的 114 个融合用例在本卡跑了一遍：

```
113 passed, 1 skipped in 109.05s
```

覆盖 `bv16/bv32/bv64/bv64w8` 各变体与 auto、dense 多 batch、非 BT 倍数的尾块、tail-mask
的 Inf/NaN 隔离、`final_state` 一致性、以及整条 pipeline 的 `fusion=ALWAYS` 对齐纯 Triton
基线。唯一那个 skip 是 `w8/w16` 波宽轴的 gfx942 专属用例。所以下面的性能数字是在数值正确
的前提下取的，但**这仍然是「把 gfx942 的 kernel 搬到 gfx950 会怎样」，不是 PR 当前在
gfx950 上的行为**（当前行为：融合不启用）。

**前端是 Triton，不是 FlyDSL。** FlyDSL 的前端 kernel `gdn_prepare` 调
`flydsl.expr.gpu.shuffle`，本机装的 flydsl 0.2.4 没有这个符号
（`/workspace/FlyDSL` 源码树有，但没有构建产物）：

```
AttributeError: module 'flydsl.expr.gpu' has no attribute 'shuffle'
```

融合 K5+K6 本身不用 shuffle，所以把它配上 Triton 的 prepare 三件套
（`fused_chunk_local_cumsum_scaled_dot_kkt_fwd_kernel` + `merge_16x16_to_64x64_inverse_kernel`
+ `recompute_w_u_head_major_kernel`）就能测，**K5+K6 那一列不受影响**。受影响的是 wall：
opus 的 `gdn_k1_neumann_kernel` 比这三件套快 2.0x（geomean 0.499），所以 wall 口径
（geomean(opus/FlyDSL) = 0.879，FlyDSL 只赢 19/64）对 FlyDSL 是不利的，**不能当作
pipeline 结论**。gfx942 上 FlyDSL 前端反而比 opus 快 1.30x。

**`output_final_state` 两侧对齐。** 两列都是 `output_final_state=True` 加真实
`initial_state`，final state 的写回成本两边都算进去了。

---

## 3. 和 gfx942 基线对照：为什么反超点消失了

同一份 64 格网格，两张卡的融合 kernel（都取各自最优变体）：

| B·H | gfx942 opus | gfx942 FlyDSL | gfx942 谁快 | gfx950 opus | gfx950 FlyDSL | gfx950 谁快 |
| ---: | ---: | ---: | :--- | ---: | ---: | :--- |
| 8 | 1181 | 504 | FlyDSL 2.35x | 767 | 336 | FlyDSL 2.28x |
| 32 | 902 | 558 | FlyDSL 1.62x | 574 | 254 | FlyDSL 2.25x |
| 64 | 617 | 747 | **opus 1.21x** | 390 | 212 | FlyDSL 1.78x |
| 256 | 1306 | 1434 | opus 1.10x | 250 | 204 | FlyDSL 1.19x |
| 4096 | 8683 | 10977 | opus 1.26x | 2175 | 1885 | FlyDSL 1.15x |

换卡带来的加速比（gfx942 → gfx950，CU 数 80 → 256，比值 3.2x）：

| B·H | opus WF 提速 | FlyDSL 提速 |
| ---: | ---: | ---: |
| 8 | 1.54x | 1.50x |
| 64 | 1.58x | 3.52x |
| 256 | 5.22x | 7.03x |
| 4096 | 3.99x | **5.82x** |

低 B·H 段（B·H=8）两边都只拿到 1.5x，远低于 CU 比——这一段是串行递归受限，加宽设备帮不上，
两边的相对关系因此保持不变（FlyDSL 稳定领先 2.3x 左右）。高 B·H 段两边都进入
throughput-bound，但 **FlyDSL 从新硬件拿到的收益明显更多**（5.82x vs 3.99x，前者超过 CU 比，
说明还吃到了单 CU 的提升），gfx942 上 opus 那 1.1–1.3x 的领先就是在这一段被抹掉的。

---

## 4. 变体选择规则在 gfx950 上要换，而且换的方向和 gfx942 相反

单点探针（`probe_fused_variant.py`，seqlen=8192，强制四种变体，时间是融合 kernel 的 wall）：

| B·H | auto 选 | bv16 | bv32 | bv64 | bv64w8 | 最优 | auto 罚分 |
| ---: | :--- | ---: | ---: | ---: | ---: | :--- | ---: |
| 8 | `bv16` | **336.6** | 401.0 | 528.0 | 442.9 | `bv16` | 1.00x |
| 16 | `bv16` | **335.2** | 402.2 | 529.9 | 445.0 | `bv16` | 1.00x |
| 32 | `bv32` | **341.0** | 403.6 | 531.0 | 446.5 | `bv16` | 1.18x |
| 64 | `bv64` | **493.3** | 510.8 | 582.2 | 545.2 | `bv16` | 1.18x |
| 128 | `bv64w8` | 970.5 | 600.0 | 603.9 | **567.3** | `bv64w8` | 1.00x |
| 256 | `bv64w8` | 1936.9 | 1192.6 | **775.8** | 1128.5 | `bv64` | **1.45x** |
| 512 | `bv64w8` | 3837.2 | 2356.1 | **1565.9** | 2242.0 | `bv64` | **1.43x** |

两条和 gfx942 相反的结论：

**波宽加倍（`w8`）在 gfx950 上是负优化。** gfx942 上 `bv64w8` 是大 B·H 段的最优选择
（那边 LDS 把一个 CU 钉在一个 workgroup 上，多出来的常驻 wave 有用）；gfx950 上它在
B·H ≥ 256 反而比 `bv64` 慢 1.43–1.45x。auto 在这一段一律选 `bv64w8`，这就是第 1 节里
as-shipped 曲线在 B·H ≥ 256 输给 opus 的全部原因。

**最优点从「一个 CTA wave」变成「两个」。** gfx942 报告里的规则是「网格 CTA 数最接近但不
超过 CU 数」；gfx950 上最优变体对应的 CTA 数稳定在 2×CU 附近：

| B·H | 最优变体 | CTA 数 | CTA / 256 CU |
| ---: | :--- | ---: | ---: |
| 32 | `bv16` | 256 | 1.0× |
| 64 | `bv16` | 512 | 2.0× |
| 256 | `bv64` | 512 | 2.0× |
| 512 | `bv64` | 1024 | 4.0× |

写成规则（`sweep_k5k6_compare.py:cu_scaled_variant`，本页第 1 节右两列用的就是它）：

```python
if arch == "gfx950":
    for tag, bv in (("bv16", 16), ("bv32", 32), ("bv64", 64)):
        if -(-V // bv) * bh <= 2 * cus:
            return tag
    return "bv64"
```

7 个探针点里 6 个精确命中，B·H=128 那点选 `bv32`（600.0）而实测最优是 `bv64w8`（567.3），
差 5%。

**给上游的建议**：`_fused_bv_for_shape` 的 `_GFX942_MIN_FILL = 0.37` 和 `w8` 的启用条件
（`_FUSED_W8_MIN_FILL = 0.55`）都是 gfx942 标定值，搬到 gfx950 会同时选错 BV 和错误启用
`w8`。如果要在 gfx950 上开融合路径，这两个常数需要按本卡重标；`w8` 轴建议在 gfx950 上直接
关掉。

---

## 5. gfx950 上 K5+K6 这一段到底该走哪条路

第 1 节只比了两个融合实现。把两边的分离实现也放进来（`probe_k5k6_paths.py`，seqlen=8192，
每个数字都是完整 pipeline 里 K5+K6 段的 profiler device time，所以前端差异不计入）：

| B·H | opus WF 融合 | opus WS 分离 | FlyDSL 融合 | FlyDSL 分离 + Triton K6 | 最优 |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 8 | 768.3 | 229.4 | 334.2 | **190.0** | FlyDSL 分离 |
| 16 | 768.6 | 273.0 | 336.5 | **209.0** | FlyDSL 分离 |
| 32 | 770.3 | 432.7 | **338.1** | 339.8 | 平手 |
| 64 | 854.7 | 561.3 | **491.0** | 545.0 | FlyDSL 融合 |
| 128 | 882.9 | 784.0 | **609.9** | 882.0 | FlyDSL 融合 |
| 256 | 936.4 | 1433.9 | **775.0** | 1382.2 | FlyDSL 融合 |
| 512 | 1883.6 | 3081.4 | **1543.1** | 2793.1 | FlyDSL 融合 |

（FlyDSL 分离 = `chunk_gdn_fwd_h_flydsl_opt` + Triton `chunk_fwd_kernel_o_opt_vk`，即
`fusion=NEVER` 时 gfx950 的实际落点；opus WS = `chunk_gated_delta_rule_fwd_h_hip_kernel`
+ `gdn_k2_out_kernel`。）

三条结论：

1. **opus 的两种模式在任何 B·H 都不是最优。** 低 B·H 段 opus WS 被 FlyDSL 分离压住
   （229 vs 190），高 B·H 段 opus WF 被 FlyDSL 融合压住（1884 vs 1543）。
2. **融合与分离的分界在 B·H ≈ 32。** 分离路径要把 `h` 写出再读回给 K6，这笔流量随 B·H
   线性增长，所以大 B·H 段融合稳赢；反过来融合 kernel 的并行度是 `⌈V/BV⌉·B·H` 个 CTA，
   B·H 小的时候喂不饱 256 个 CU，这时省掉一次 launch 换不回被闲置的设备。
3. **PR 现在 `fusion=AUTO` 在生产点不融合，这个决定是对的。**
   `should_use_fused_k5k6_gfx942(H=8, N=1, V=128)` 返回 `False`（`_FUSED_MIN_FILL=0.45`，
   而 `bv16` 在 B·H=8 只有 64/256 = 0.25 的 fill）。第 1 节的 2.28x 是「融合对融合」的
   比较，不是说这里该开融合——同一个点上分离比融合快 1.76x（190.0 vs 334.2）。

### 5.1 放开 arch 门控之后，AUTO 的判据还差一点

如果只把 arch 门控放开而不动别的，`fusion=AUTO` 在 gfx950 上的决策对 7 个探针点里的 6 个
是正确的，但 B·H=16 会选错：

| B·H | AUTO 选中的 BV | fill | AUTO 融合？ | 融合 | 分离 | AUTO 选对？ |
| ---: | :--- | ---: | :--- | ---: | ---: | :--- |
| 8 | `bv16` | 0.25 | 否 | 334.2 | **190.0** | 是 |
| 16 | `bv16` | 0.50 | **是** | 336.5 | **209.0** | **否，慢 1.61x** |
| 32 | `bv32` | 0.50 | 是 | **338.1** | 339.8 | 是（平手） |
| 64 | `bv64` | 0.50 | 是 | **491.0** | 545.0 | 是 |
| 128 | `bv64w8` | 1.00 | 是 | **609.9** | 882.0 | 是 |
| 256 | `bv64w8` | 2.00 | 是 | **775.0** | 1382.2 | 是 |
| 512 | `bv64w8` | 4.00 | 是 | **1543.1** | 2793.1 | 是 |

B·H=16 不是边角料：TP=8、H=8、B=2、seqlen=8192（两条 8K 序列，16K token）是常见档位。

根因是 `_FUSED_MIN_FILL` 拿**实际选中的那个 BV** 去算 fill。B·H=16 配 `bv16` 和 B·H=32 配
`bv32` 都是 128 个 CTA、fill 都是 0.50，但一个该融合、一个不该——同一个 fill 值对应相反的
结论，所以这个判据分不开它们。真正在动的是 B·H 本身：分离路径的时间随 B·H 快速上升
（190 → 209 → 340 → 545），融合 kernel 在 B·H ≤ 32 几乎是常数（334 → 336 → 338），
说明它在这一段完全 latency-bound。

改成拿**最小合法 BV**（`bv16`）算 fill 就能对齐实测分界，即「融合 iff
`⌈V/16⌉·B·H ≥ CU`」，在 256 CU 上就是 B·H ≥ 32：

| B·H | `bv16` 的 CTA 数 | / 256 CU | 该融合？ | 实测 |
| ---: | ---: | ---: | :--- | :--- |
| 8 | 64 | 0.25 | 否 | 分离快 1.76x |
| 16 | 128 | 0.50 | 否 | 分离快 1.61x |
| 32 | 256 | 1.00 | 是 | 平手 |
| 64 | 512 | 2.00 | 是 | 融合快 1.11x |

**所以在 gfx950 上开融合路径需要三处一起动**：放开 arch 门控、把融合与否的判据换成上面这条、
以及关掉 `w8` 轴（第 4 节）。只放开门控会在 B·H=16 引入 1.61x 的回归。

---

## 6. 复现

PR #4884 在本分支已 merge 过一版（`abfb959f7`），这次把 PR 之后的 7 个新提交
cherry-pick 上来（无冲突）：

```bash
git fetch origin users/vpietila/kda-prefill-chunk-gated-delta
git cherry-pick abfb959f7^2..FETCH_HEAD    # 7 个提交，无冲突
```

然后放开融合 kernel 的两道 arch 门控。**这两处改动没有入库**（融合 kernel 在 gfx950 上还没
标定，见第 4、5 节），复现时需要手动打上，都在
`aiter/ops/flydsl/gdn_fused_gfx942_kernels.py`：

```python
# 1) is_fused_k5k6_gfx942_unsupported()
-    if _host._ARCH != "gfx942":
-        return f"the fused K5+K6 kernel is gfx942-only; this device is {_host._ARCH}"
-    if _host._device_cu_count() < 304:
+    if _host._ARCH not in ("gfx942", "gfx950"):
+        return f"the fused K5+K6 kernel is gfx942/gfx950-only; this device is {_host._ARCH}"
+    if _host._ARCH == "gfx942" and _host._device_cu_count() < 304:

# 2) chunk_gated_delta_rule_fwd_h_o_flydsl() 入口
-    if _host._ARCH != "gfx942":
+    if _host._ARCH not in ("gfx942", "gfx950"):
         raise NotImplementedError(...)
```

之后：

```bash
# 正确性：PR 自带的融合用例（默认在非 gfx942 上整套 skip）
sed -i 's/if get_gfx() != "gfx942":/if get_gfx() not in ("gfx942", "gfx950"):/' \
  op_tests/test_flydsl_gdn_fused_k5k6.py
HIP_VISIBLE_DEVICES=7 PYTHONPATH=$PWD python -m pytest \
  op_tests/test_flydsl_gdn_fused_k5k6.py -q

# 64 格网格：opus WF / FlyDSL auto 变体 / FlyDSL 实测最优变体
HIP_VISIBLE_DEVICES=7 PYTHONPATH=$PWD python op_tests/flydsl_tests/sweep_k5k6_compare.py \
  --out op_tests/flydsl_tests/k5k6_compare_gfx950.json

# 变体探针：强制四种 BV
HIP_VISIBLE_DEVICES=7 PYTHONPATH=$PWD python op_tests/flydsl_tests/probe_fused_variant.py \
  --out op_tests/flydsl_tests/fused_variant_probe_gfx950.json

# 四条路径：融合 vs 分离，opus vs FlyDSL
HIP_VISIBLE_DEVICES=7 PYTHONPATH=$PWD python op_tests/flydsl_tests/probe_k5k6_paths.py

# 网页 + 截图
python op_tests/flydsl_tests/render_k5k6_compare.py \
  --json op_tests/flydsl_tests/k5k6_compare_gfx950.json \
  --out docs/gdn-k5k6-opus-vs-flydsl-gfx950.html
python op_tests/flydsl_tests/shot_gdn_mode_grid.py \
  --html docs/gdn-k5k6-opus-vs-flydsl-gfx950.html \
  --out docs/images/gdn-k5k6-opus-vs-flydsl-gfx950.png
```

`PYTHONPATH=$PWD` 是必须的：本机 aiter 以 editable 方式装在别处，不覆盖它就会跑到旧版本。

三个脚本都加了 `--front {auto,flydsl,triton}`：`auto` 检测
`flydsl.expr.gpu.shuffle` 是否存在，缺失就落到 Triton prepare，所以在 gfx942 上仍按原
口径（FlyDSL 前端）跑，不需要改参数。

产物：

| 文件 | 内容 |
| --- | --- |
| `op_tests/flydsl_tests/k5k6_compare_gfx950.json` | 64 格原始数据（三列 × wall/k5k6/front + 变体） |
| `op_tests/flydsl_tests/fused_variant_probe_gfx950.json` | 变体探针原始数据 |
| `op_tests/flydsl_tests/k5k6_paths_gfx950.json` | 四条路径对比原始数据 |
| `docs/gdn-k5k6-opus-vs-flydsl-gfx950.html` | 交互网页（悬停看每格明细，可切 as-shipped / 最优变体） |
| `docs/images/gdn-k5k6-opus-vs-flydsl-gfx950.png` | 网页截图 |

网格维度：TP ∈ {1,2,4,8}（H = 64/32/16/8，Hg = H/4）× seqlen ∈ {1K,2K,4K,8K}
× T ∈ {8K,16K,32K,64K}，B = T/seqlen，packed varlen。每格 wall 取 50 次中位、
per-kernel device time 取 profiler 20 次平均。

---

## 7. 相关文件

| 文件 | 说明 |
| --- | --- |
| `docs/gdn_k5k6_opus_vs_flydsl.md` | gfx942 · 80 CU 基线（本页的对照） |
| `docs/gdn_prefill_mode_gfx950.md` | 同一张卡上 opus 四种模式（WF/WS/CF/CS）的对比 |
| `aiter/ops/flydsl/gdn_fused_gfx942_kernels.py` | 融合 K5+K6 的 host 侧：arch 门控、BV/波宽选择、融合与否的路由 |
| `aiter/ops/flydsl/kernels/chunk_gated_delta_h_gfx942.py` | 融合 kernel 的 FlyDSL builder（LDS 预算、`MFMA_K=16` 等 gfx942 参数） |
