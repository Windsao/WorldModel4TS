# 让 video backbone for TS work：执行结果

执行 `VIDEO_TS_LITERATURE_AND_RESCUE_PLAN.md`。NU CS 集群（nyx / hemera，各 4×A40），零成本，
2026-09-05 23:30 – 09-06 09:50。判定标准不变：条件 (i) 预训练臂优于同样训练的随机初始化臂；
条件 (ii) 加入后让 ZL-051 blend（最佳无视频方法）变好；paired moving-block bootstrap，
origin 与 VisionTS / ZL-051 完全相同（pairs_sha1 断言）。

## 0. 结论

**路线 B 满足了条件 (ii)，这是本项目第一个过关的正面结果。** 在相同的时序继续预训练
（LOTSA 子集 2.78 亿观测、20k 步、四臂只差编码器起点）之后，Kinetics VideoMAE 初始化的模型
作为零样本预测器在六个基准数据集上 Q = 0.787（vs VisionTS）、0.860（vs ZL-051 blend）；作为
第五个候选加入 blend 后 Q = 0.858，四个数据集 CI 不含 1，两个 ETT 小时数据集持平。**初始化是
因果变量**：视频编码器 < 图像编码器 < 随机的顺序在像素损失、Stage F、Stage E 三处一致。

**路线 A 得到一个有边界的结论：** 自回归视频模型 MAGI-1 零样本地延续了渲染序列的运动（合成
外推门通过，ramp 0.17×、travel 0.20× copy-last 误差），在真实数据上也优于 snaive
（ETTh2 0.79×、ETTm2 0.89×），但输给四个滑动平均的 blend（1.15× / 1.09×），按计划在 G2 停止。

**对项目主张的修正：** 之前的三份报告说"视频预训练的表征不是预测需要的表征"，在**冻结零样本**
regime 里仍然成立；但在**继续预训练迁移** regime 里它是比 ImageNet 图像 MAE 更好的起点，
差距在高通道数据集上很大（traffic 上单独的视频模型比 VisionTS 好 40%）。VisionTS++ 报告的
"ImageNet 初始化值 30%"有了视频版：等预算下视频初始化对随机为 0.835，图像初始化对随机为 0.979。

## 1. 路线 B：视频初始化 vs 图像初始化 vs 随机初始化（等预算时序继续预训练）

### 1.1 设置

- 架构 VideoMAE-B（`MCG-NJU/videomae-base` 配置，`norm_pix_loss=False`），四臂：
  `vmae_full`（Kinetics 编码器 + 其预训练解码器）、`vmae_enc`（Kinetics 编码器 + 全新解码器）、
  `imae_enc`（VisionTS 所用 ImageNet MAE 编码器做 3D 膨胀 + 全新解码器；qkv 的 key bias 被丢弃，
  它在 softmax 里是每个 query 的常数）、`random`。后三臂共享 bit 级相同的解码器初始化；四臂数据
  流（同 seed、同 worker 数）、mask、优化器、步数完全相同。
- 数据：LOTSA 子集 52 个数据集、13.8 万条序列、2.78 亿观测，按名称排除 energy / solar / wind /
  road-traffic 域（`extended_web_traffic` 与 `kaggle_web_traffic_weekly` 因子串 "traffic" 被过度
  排除）；另混 20% 合成序列（7 月脚本的生成器）。六个基准数据集从未被读取。完整列表见
  `pilot/results_field/route_b/corpus_meta.json`。
- 渲染：frame f = period f 的平滑面积图（224×224，值 → 位置，上下文统计只来自可见周期）；
  预测 = 遮住最后 1 或 2 个 tubelet（2/4 个整帧），只在未来 token 上算原始像素 MSE；30% 的预测
  批次额外遮住前 1–3 个 tubelet（8–12 个可见周期，供评测的伪 origin 使用）；30% 原生 75% tube mask。
- 训练：batch 32，AdamW lr 1e-4（warmup 500，cosine），wd 0.05，bf16，20k 步，单卡每臂
  0.47 s/step（2.6 h），单个训练 seed。
- 评测 `pilot/eval_route_b.py`：模型用上下文的最后 12（或 14）个周期，输出 hp=4（或 2）帧解码；
  条件 (ii) 把模型当作第五个候选放进 ZL-051 的 evidence-scaled 稠密伪 origin blend（先验四个不变；
  纯先验 blend 与 ZL 的实现逐元素相等，已断言）。
- 测试：`tests/test_route_b.py` 14 项全过。

### 1.2 训练曲线（held-out 未来帧原始像素 MSE，8 个固定批次，四臂相同）

| arm | 1k | 2k | 5k | 10k | 15k | 20k |
|---|---:|---:|---:|---:|---:|---:|
| vmae_full | 0.0395 | 0.0382 | 0.0360 | 0.0350 | 0.0336 | **0.0333** |
| vmae_enc | 0.0497 | 0.0476 | 0.0408 | 0.0368 | 0.0356 | **0.0350** |
| imae_enc | 0.0500 | 0.0474 | 0.0434 | 0.0400 | 0.0382 | 0.0375 |
| random | 0.0516 | 0.0492 | 0.0447 | 0.0403 | 0.0390 | 0.0381 |

### 1.3 Stage F：六数据集零样本，基准测试窗口（`pilot/results_field/route_b/eval/stageF/`）

| 方法 | ETTh1 | ETTh2 | ETTm2 | electricity | traffic | solar | Q vs blend | Q vs VisionTS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| smean | 0.4026 | 0.3454 | 0.3301 | 0.2069 | 0.5106 | 0.1973 | 1.103 | 1.010 |
| ZL-051 blend | 0.4076 | 0.3026 | 0.2003 | 0.1980 | 0.5082 | 0.2137 | 1.000 | 0.915 |
| VisionTS | 0.4023 | 0.3130 | 0.2509 | 0.2014 | 0.5207 | 0.2729 | 1.093 | 1.000 |
| video_random | 0.4378 | 0.3286 | 0.1899 | 0.2139 | 0.4760 | 0.2284 | 1.030 | 0.943 |
| video_imae_enc | 0.4246 | 0.3286 | 0.2037 | 0.1998 | 0.4247 | 0.2324 | 1.009 | 0.924 |
| video_vmae_enc | 0.4341 | 0.3220 | 0.1762 | 0.1886 | 0.3567 | 0.2325 | 0.948 | 0.868 |
| **video_vmae_full** | 0.4289 | 0.3123 | **0.1698** | **0.1536** | **0.3122** | **0.1969** | **0.860** | **0.787** |
| blend + video_random | 0.4000 | 0.2994 | 0.1869 | 0.1848 | 0.4556 | 0.2154 | 0.956 | 0.875 |
| blend + video_imae_enc | 0.4009 | 0.2998 | 0.1882 | 0.1767 | 0.4189 | 0.2207 | 0.941 | 0.862 |
| blend + video_vmae_enc | 0.4052 | 0.2981 | 0.1788 | 0.1723 | 0.3657 | 0.2197 | 0.909 | 0.832 |
| **blend + video_vmae_full** | 0.4071 | 0.2975 | 0.1756 | 0.1528 | 0.3252 | 0.2008 | **0.858** | 0.785 |

Paired moving-block bootstrap（5000 次，block = ceil((L+H)/stride)，按 origin 配对）：

| 比较 | ETTh1 | ETTh2 | ETTm2 | electricity | traffic | solar |
|---|---|---|---|---|---|---|
| 条件 (ii) blend+vmae_full / blend | [0.985, 1.013] 平 | [0.958, 1.013] 平 | **[0.859, 0.903]** | **[0.722, 0.804]** | **[0.567, 0.713]** | **[0.884, 0.968]** |
| 条件 (i) vmae_full / random | [0.926, 1.033] 平 | [0.907, 1.010] 平 | **[0.841, 0.907]** | **[0.669, 0.755]** | **[0.581, 0.722]** | **[0.758, 0.954]** |
| vmae_enc / random | [0.955, 1.025] 平 | [0.954, 1.019] 平 | **[0.885, 0.936]** | **[0.853, 0.906]** | **[0.700, 0.801]** | [0.923, 1.098] 平 |
| imae_enc / random | **[0.951, 0.984]** | [0.976, 1.034] 平 | [1.017, 1.110] 更差 | **[0.913, 0.963]** | **[0.874, 0.921]** | [0.967, 1.089] 平 |
| vmae_full 单独 / blend | [1.004, 1.121] 更差 | [0.970, 1.127] 平 | **[0.808, 0.915]** | **[0.717, 0.802]** | **[0.551, 0.669]** | **[0.829, 0.974]** |

加粗 = CI 不含 1 且方向为更好。origin 数：ETTh1/ETTh2 349，ETTm2 1115，electricity 620，
traffic 425，solar 1017。

### 1.4 Stage E：历史审计窗口（训练范围内，与 Stage F 独立的分割）

同样的模式在另一组窗口上复现：blend+vmae_full / blend Q = 0.867，vmae_full / random 0.811，
vmae_enc / random 0.894，imae_enc / random 0.971；electricity / traffic / ETTm2 上 CI 均不含 1，
ETTh1 / ETTh2 / solar 上条件 (ii) 持平。完整表见 `route_b/eval/summary_E_20000.json`。

### 1.5 读法与保留

- 视频编码器初始化在 traffic 上把 blend 的 0.508 变成 0.325，electricity 0.198 → 0.153，
  ETTm2 0.200 → 0.176，solar 0.214 → 0.201。两个 ETT 小时数据集（349 个 origin，噪声最大）持平。
- 图像 MAE 初始化对随机在 electricity / traffic 上有小幅显著优势（约 5–10%），但 ETTm2 上
  更差；它与视频初始化之间的差距（traffic 0.425 vs 0.357）比它与随机的差距大。
- `vmae_full` 比 `vmae_enc` 又好一截（Q 0.860 vs 0.948），说明 Kinetics 预训练的**解码器**也有
  贡献，不只是编码器；这一臂与 imae 不是严格匹配的（imae 没有对应的预训练解码器可用）。
  严格匹配的三臂比较是 vmae_enc / imae_enc / random。
- 只有一个训练 seed；四臂之间的差异是否超过训练噪声，只能由 Stage E 的独立复现和每数据集
  CI 支持，多 seed 未跑。
- 数据泄漏：语料按名称排除，六个基准数据集及其同源数据（UCI electricity、PeMS、solar_AL）
  均不在 LOTSA 下载列表中。

## 2. 路线 A：自回归视频模型（MAGI-1 4.5B）作为像素级外推器

### 2.1 设置

- MAGI-1 4.5B base（64 步、cfg 3）与 distill（cfg 1），480×480，24 fps，
  `/nyx-storage1/hanliu/miniconda3/envs/magi`。
- 渲染 `pilot/arvideo_render.py`：值 → 位置。每个周期是一帧平滑轮廓（相位中心线性插值的填充
  区域），相邻周期之间插 3 帧过渡（每周期 4 帧）。顶部时钟条：红色方块每帧右移 3 px。
- MAGI 的 `process_prefix_video` 固定读 32 帧 prefix → 8 个上下文周期；周期 k 位于绝对帧 4k+3。
  MAGI 输出只含生成帧，第一帧对应绝对帧 32（全部 96 次运行由时钟读数验证为 31–33）。
- 解码：每列统计 R 通道低于 FG/BG 中点的行数 → 边界行 → z；按相位中心列采样。
- 第一版渲染（阶梯色块 + prompt 含 "skyline"）被 MAGI 续成了城市楼群纹理
  （`results_field/arvideo/g1_base/ramp_s*/sheet.png`）；改为平滑轮廓与 "rolling hills /
  water level" prompt 后消失。

### 2.2 G0 codec 门：PASS

纯 numpy 往返 ≤ 0.02 z；mp4（x264 crf 6）往返最大误差 0.0084 z（门限 0.03）；5×5 高斯模糊
+ σ=6 噪声 + 亮度 +8 下平滑形状 ≤ 0.05 z。

### 2.3 G1 合成外推门：PASS（`results_field/arvideo/g1_v2/summary.json`）

8 个上下文周期 → 8 个未来周期，每种探针 3 个 seed，对照 copy-last（snaive 的等价物）。

| 探针 | base（3 seed 几何均值） | distill | base 单次 |
|---|---:|---:|---|
| const | 无漂移（2.6e-5） | 无漂移 | — |
| ramp（每周期 +0.25 std） | **0.17** | **0.12** | 0.086 / 0.173 / 0.338 |
| sine_level（8 周期正弦） | 0.75（门限 0.7，边缘） | 0.79 | 1.39 / 0.53 / 0.58 |
| travel（形状每周期平移 1/8） | **0.20** | **0.15** | 0.29 / 0.08 / 0.35 |

门：const 无漂移，且 travel 或 sine ≤ 0.7（travel 0.20）。**这是本项目第一次观察到视频模型
零样本地延续数值序列的运动**：线性漂移与平移被外推，振荡外推较弱。模型播放时钟标记的速度
比真实快 4–23%（clock_rate 1.04–1.23）。distill 不弱于 base，G2 采用 distill（7.8 min/段，
base 14.5 min）。

### 2.4 G2 历史审计：FAIL（`results_field/arvideo/g2/summary.json`）

ETTh2、ETTm2 各 36 个不同窗口（审计分割，不读基准未来），MAGI 用最后 8 个周期生成 4（或 1）个
未来周期；blend 用完整上下文 L。

| 数据集 | MAGI | snaive | smean | blend | MAGI/snaive（CI） | MAGI/blend（CI） |
|---|---:|---:|---:|---:|---|---|
| ETTh2 | 0.573 | 0.721 | 0.453 | 0.497 | 0.79 [0.63, 1.21] | **1.15 [1.03, 1.39]** 更差 |
| ETTm2 | 0.412 | 0.463 | 0.494 | 0.379 | 0.89（block ≥ n，CI 不可算） | 1.09（同上） |

两个数据集都过 snaive 门（≤ 0.95）、都不过 blend 门（≤ 1.0）；ETTh2 上比 blend 差是显著的。
按计划 §3 的顺序门在此停止：MAGI 有真实的外推先验，但零样本精度不及四个滑动平均的 blend。
它只用 8 个周期的上下文（MAGI 的 32 帧 prefix 限制），blend 用 16 个；这是它的结构性劣势，
也是继续这条路线要解决的第一件事。

## 3. 路线 C

未运行（计划中为空闲卡时的旁路检查；两条主线占满了两台节点）。

## 4. 文献对照

- VisionTS++ 报告 ImageNet 初始化在时序 CPT 后仍值约 30%；本文在等预算下测得视频初始化对随机
  0.835、图像初始化对随机 0.979（Q vs random，Stage F）。
- SVTime / OccamVTS 说明冻结图像 MAE 的价值是局部插值先验；本文的冻结-零样本负结论与之一致，
  而 CPT 后视频初始化的优势说明视频预训练里有插值之外、对预测有用的结构，只是需要训练才能取出。
- Frozen-video-forecasting 评测里生成式模型远好于 masked 模型；本文 G1 与之一致（MAGI 外推
  运动，VideoMAE 从未做到），但 G2 说明这个先验的零样本精度还打不过强先验 blend。

## 5. 文件

```
pilot/pretrain_route_b.py          Route B corpus builder + matched four-arm trainer
pilot/eval_route_b.py              six-dataset zero-shot evaluation, conditions (i)/(ii)
pilot/summarize_route_b.py, pilot/run_route_b_eval.sh
pilot/arvideo_render.py            Route A renderer / decoder / clock / synthetic probes
pilot/run_arvideo_g1.py            G0 + G1 runner (MAGI v2v, clock alignment, rescore)
pilot/run_arvideo_g2.py            G2 historical audit runner
pilot/magi_run.sh, magi_4p5B_480*.json (on the cluster)
tests/test_arvideo_codec.py (11)   tests/test_route_b.py (14)
pilot/results_field/route_b/       eval JSONs (stage E/F), train logs, corpus meta
pilot/results_field/arvideo/       g1_base / g1_v2 / g1_v2_distill / g2 JSONs + contact sheets
cluster: /nyx-storage1/hanliu/wm4ts/{route_b (13G: corpus + 4 arms x 8 ckpts), arvideo, magi (42G), lotsa (6.4G)}
```
