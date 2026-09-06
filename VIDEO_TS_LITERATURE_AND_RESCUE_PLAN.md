# 文献回顾与"让 video backbone for TS work"的方案

> **执行结果见 `VIDEO_TS_RESCUE_RESULTS.md`（2026-09-06）。** 路线 B 满足条件 (ii)：Q = 0.858 vs
> ZL-051 blend，0.787 vs VisionTS，视频初始化 > 图像初始化 > 随机。路线 A 过 G1、止于 G2
> （优于 snaive，输给 blend）。路线 C 未跑。

日期 2026-09-05。前置：`ZEROSHOT_BACKBONE_FINAL.md`、`VIDEO_VISIONTS_RECOVERY_RESULTS.md`、
`VIDEO_VISIONTS_MATCHED_MASK_RESULTS.md`、`VIDEO_VISIONTS_CARRIER_SIGNAL_RESULTS.md`。
判定标准不变：任何"video 有用"的主张必须在相同 origin 上满足条件 (ii)，即加入后让最佳无视频
方法（ZL-051 blend）变好，paired CI 不含 1。

---

## 1. 文献告诉我们什么

| 文献 | 结论 | 对我们的含义 |
|---|---|---|
| **VisionTS++** (2508.04379) | 图像 MAE 要成为 SOTA 需在 LOTSA（2310 亿观测）上**全参数**继续预训练 10 万步；但去掉 ImageNet 初始化会差约 30% | 视觉预训练的价值是**初始化**，只在闭合模态差距之后才显现。零样本冻结是它最弱的使用方式 |
| **SVTime** (2510.09780) | VisionTS 的零样本能力可以用 3 条归纳偏置复现：周期间平滑、按 patch 行分组的周期一致性、随距离衰减的局部注意力。21.5 万参数的小模型在 Electricity/Traffic 上反超 VisionTS 2–3% | VisionTS 从视觉预训练里拿到的就是一个**局部插值先验**。这和我们 matched-mask 里测到的"历史区 0.39–0.52，未来区 0.97–1.22"完全一致 |
| **OccamVTS** (2508.01727) | 时序特征只与低层纹理对齐，高层语义反而有害；蒸馏到 1% 参数不掉点 | 骨干越"懂视频语义"（Kinetics 动作）越无关。VideoMAE 相对图像 MAE 多出来的正是这部分 |
| **RoMAE** (2505.20535) | BERT 式双向 MAE 编码器没有因果/前进的归纳偏置，适合插值与表征，不适合外推 | 我们第二层失败（插值 vs 外推）在文献里有独立印证 |
| **Frozen video forecasting 统一评测** (2507.13942) | 9 个冻结模型里，**生成式**视频模型 WALT 的像素级预测远好于所有 masked 模型（FD 5.46 vs 28–30）；图像模型显著弱于视频模型；语言监督无帮助 | 如果视频预训练里存在"外推先验"，它在**生成式 / 自回归**模型里，不在 MAE 里 |
| **ICI-Time** (2608.23855) | 冻结 LVM 做 grid 视觉 in-context inpainting：第一行放一个（历史→实现未来）示例对，第二行放 query，零样本可用；高度尺度从示例迁移到 query 带来 9–18% | in-context 示例是视觉模型零样本外推的一条已验证途径。视频的帧轴天然适合放示例对 |
| **MAGI-1 / Self-Forcing / Next Forcing** | 开源自回归视频模型，按 chunk 因果生成，原生支持 video continuation | 训练目标就是"给定过去帧生成未来帧"，是像素级外推器，无 `norm_pix_loss`，不需要 ridge readout |
| **TTA 系列** (PETSA 2506.23424, FAC 2605.17250) | 冻结 TSFM 只在输入/输出侧加小校准模块，频域参数化 | 与我们 R4 lens 同类，文献里增益也只有几个百分点，不是解决方案 |
| **Position: category error** (2602.05287) | 时序不是一个模态，通用 TSFM 退化为"通用滤波器" | 解释为什么 4 个滑动平均能打败 VisionTS：可迁移的公共先验本来就小 |

## 2. 我们的四层失败与文献的对应

| 层 | 我们的证据 | 文献印证 |
|---|---|---|
| 任务天花板低 | ZL-051 零参数 blend 打败 VisionTS，Q_all 0.9152 | SVTime 21.5 万参数打败 VisionTS；category-error 论文 |
| 学到的是插值不是外推 | 历史区 patch 0.39–0.52，未来区 0.97–1.22；打散未来 patch 反而 1.52 | RoMAE；SVTime 的 3 条偏置全是插值型 |
| `norm_pix_loss` 丢掉绝对水平 | 因果统计恢复比 oracle 差 3–10 倍；lens / 重训解码器 / 载波全败 | VisionTS 用 `norm_pix_loss=False` 的检查点；VisionTS++ 全参数训练解码器 |
| 表征有结构但不是预测的结构 | pt/random 0.66–0.70，B/A 1.01–1.09；V-JEPA 2 同样 | OccamVTS：高层语义有害；Frozen forecasting：masked 模型像素预测最差 |

结论：**继续在冻结 VideoMAE 上换 renderer / mask / lens 没有文献支持，也没有实验空间。**
要让 idea work 必须换掉失败的那个组件：换成有外推目标的模型（路线 A），或者换成允许闭合模态
差距的训练 regime（路线 B）。

---

## 3. 路线 A：自回归视频模型作为像素级外推器（零样本保留）

**假设** 自回归视频生成模型的预训练目标是"从过去帧外推未来帧"，因此含有 MAE 没有的外推先验；
只要把周期渲染成物理上合理的慢运动，它能零样本延续这个运动。

**它移除了哪几层失败** 第 2 层（目标是外推不是插值）、第 3 层（直接输出像素，没有归一化 cube，
没有 ridge readout）、第 4 层的一半（生成式模型在 frozen-forecasting 评测里是像素预测最好的一类）。

**为什么之前的 VACE 结果不算测过** `run_vace_ts.py` 用的是文本到视频的编辑模型做双向 inpainting
（又回到插值），值编码为灰度强度（OccamVTS 说强度不是视觉模型保留的量），未来帧填 0.5 常数（我们
在 M1.5 里测过，大块常数区域本身就是 OOD）。三个设计选择都踩在现在已知的坑上。

**模型** 首选 MAGI-1 4.5B（原生 V2V continuation，单张 A40 可跑 fp8/distill）；备选
Cosmos-Predict2.5 2B Video2World（多帧条件）。不用 T2V 编辑模型。

**渲染（值 = 位置，不是强度）** frame f = period f。每帧是 P 根柱子的"天际线"或一条水位线，高度
= 该周期在该相位的 z 值；相邻周期之间线性插值出 m 帧过渡（m=4–6），让运动慢且连续，符合自然视频
统计。背景加弱纹理（OccamVTS：时序对齐的是纹理特征）。固定相机、无文字。

**条件与生成** 条件帧 = 最近 G_ctx 个周期（G_ctx = L/P，ETTh 为 4–8），生成 H/P 个未来周期的帧。
每个 origin 采样 4 次取均值（顺便得到概率预测）。

**反演** 逐列取最高前景像素 → 高度 → z（同 ICI-Time 的 boundary extraction），去归一化用上下文
的 mean/std。这一步先过 codec gate。

**实现中确定的细节（2026-09-06）** MAGI-1 的 `process_prefix_video` 固定只读 32 帧 prefix，且输出
不含 prefix 重放，无法从重叠帧验证时间对齐；因此每帧顶部加一条"时钟"：红色方块每帧右移 3 px，
从生成帧里读方块位置即得绝对帧号，同时检验模型是否保持恒定帧率。第一版把相位画成阶梯状色块并
在 prompt 里写了 "skyline"，MAGI 把它续成了城市楼群纹理（见 `arvideo/g1_base/ramp_s*/sheet.png`），
所以改为相位中心线性插值的平滑轮廓，prompt 改为 "gentle rolling hills / water level"。解码按
相位中心列统计阈值以下的行数（对垂直模糊和 ±60 亮度漂移稳健）。

**顺序门（任一失败即停，写结论）**
- G0 codec：合成 z 网格往返误差 ≤ 0.02（我们已有这套检验）。
- G1 合成外推（最关键，一天内可得）：给模型正弦 / 线性斜坡 / 常数三种柱高运动，看生成的未来帧
  是延续运动还是复制最后一帧。判据：正弦延续 MSE 相对 copy-last ≤ 0.7。**这一步直接回答"外推
  先验存不存在"，不依赖任何真实数据集。**
- G2 历史审计 ETTh2 + ETTm2（不读测试集未来）：TS MSE 相对 snaive ≤ 0.95，且相对 blend ≤ 1.0。
- G3 六数据集正式评测：条件 (ii) vs ZL-051，paired moving-block bootstrap，pairs_sha1 与
  VisionTS manifest 一致。

**成本** 免费集群。G0–G1 约 2 GPU 天；G2 约 2 × 150 origins × 4 samples，每样本 1–2 分钟，
4 卡一天；G3 约 3–4 天。

**失败即结论** 若 G1 失败："当前开源自回归视频模型对单自由度慢运动没有可用的零样本外推先验"，
这是一个比现在所有报告都更强、更干净的负结论，因为它绕开了 MAE 的全部已知缺陷。

## 4. 路线 B：视频初始化 vs 图像初始化 vs 随机初始化（VisionTS++ 配方，regime 换成预训练迁移）

**假设** 视觉预训练的价值是初始化（VisionTS++ 的 30%）。问题变成：在相同的时序继续预训练之下，
**视频初始化是不是比图像初始化更好的起点**。这也是用户早先问过的"不同预训练数据会不会影响骨干"
在文献支持下的正确提法。

**设计** 三条完全对齐的臂，相同数据、步数、seed、渲染、mask：
1. VideoMAE-B（Kinetics-400 初始化）
2. 图像 MAE-B（VisionTS 用的 `mae_visualize_vit_base` 初始化），把同一 16 帧输入按帧独立编码
   或复制到时间轴，保证参数量与计算对齐
3. 随机初始化

训练数据用 LOTSA 的一个子集（确认 ETT / electricity / traffic / solar 不在其中），渲染 frame f =
period f 的 dense matrix，mask 固定为未来右块（不再是 tube 随机），解码器输出**原始像素**
（去掉 `norm_pix_loss`），全参数训练。ViT-B、batch 128、2 万步，4×A40 每臂 2–3 天。

**评测** 六数据集零样本，条件 (ii) vs ZL-051；同时报告公开 VisionTS++ 检查点作为上界参考。

**7 月已经做过的一半** `pilot/pretrain_vmae_ts.py`（commit 003d805，master README Phase 6/7）在
纯合成序列上对 VideoMAE 做过 20k 步 CPT（`norm_pix_loss=False`，LC/scroll 两种布局，50% 预测
mask）。零样本 ETTh1 0.740，远落后 VisionTS 0.355，且受当时的 P0 泄漏影响；配套的图像 MAE CPT
（`pretrain_mae_ts.py`，`ckpt_mae_ts`）只跑了 12k 步，没有做成等预算对照。所以 7 月的结论只是
"合成数据 CPT 不够"，三臂等预算比较从未做过。检查点仍在 `/nyx-storage1/hanliu/wm4ts/ckpt_*_ts`。

**为什么值得做** 无论结果如何都是一条干净的论文主张：要么"视频初始化 > 图像初始化 > 随机"，
要么"视频 = 图像"，都比现有的"冻结零样本不行"信息量大。VisionTS++ 已经证明这个 regime 里初始化
效应可测（30%）。

## 5. 路线 C：帧轴放 in-context 示例对（一天的旁路检查）

frames 1–7 = 过去的（上下文 + 已实现未来）完整可见，frame 8 = 当前上下文 + 未来 masked。用
非 tube 的 `bool_masked_pos`（机制上可行，但相对 tube masking 是 OOD）。ETTh2/ETTm2 历史审计，
比较 vs 当前 R5 冻结配置和 snaive。若有信号，路线 B 的 mask 直接改成这一种；若无，一天成本，
不影响主线。

## 6. 推荐顺序

1. 本周：路线 A 的 G0 + G1（合成外推测试）。这是全计划里信息量 / 成本比最高的一步。
2. 并行启动路线 B 的数据准备与臂 3（随机初始化），因为它不依赖 A 的结果。
3. A 过 G1 则接 G2/G3；不过则写负结论，B 成为主线。
4. C 只在有空闲卡时跑。

## 7. 不再做的事

- 冻结 VideoMAE 上任何新的 renderer / mask 拓扑 / 统计 lens / 载波编码。三份报告 + SVTime +
  OccamVTS 已经把这个空间关上了。
- 用"预训练 > 随机初始化"作为证据。只报告条件 (ii)。
- 文本到视频的编辑模型做双向 inpainting。

## 8. Sources

- VisionTS++: https://arxiv.org/abs/2508.04379
- SVTime: https://arxiv.org/abs/2510.09780
- OccamVTS: https://arxiv.org/abs/2508.01727
- Frozen video models for forecasting (unified evaluation): https://arxiv.org/abs/2507.13942
- ICI-Time, in-context inpainting: https://arxiv.org/abs/2608.23855
- Rotary Masked Autoencoders: https://arxiv.org/abs/2505.20535
- MAGI-1: https://arxiv.org/abs/2505.13211 · https://github.com/SandAI-org/MAGI-1
- Next Forcing: https://arxiv.org/abs/2606.11187
- PETSA: https://arxiv.org/abs/2506.23424 · FAC: https://arxiv.org/abs/2605.17250
- Position: category error: https://arxiv.org/abs/2602.05287
- VisionTS: https://arxiv.org/abs/2408.17253
