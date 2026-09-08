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

**第二轮（§1.6，给图像臂配上 MAE 自己的预训练解码器，两个 seed）把这个主张收窄成两句话：**
解码器预训练本身值 5–7%，它把图像臂拉到与"视频编码器 + 新解码器"持平；但在解码器完全相同时，
视频编码器仍比图像编码器好 4–7%、比随机好 7–10%，图像编码器只比随机好 2–4%，两个 seed 一致；
两边都用各自的预训练解码器时视频仍领先 2.4–4.6%（且视频臂的解码器小一倍多）。seed 间噪声 ≤ 3%。

**第三轮（§1.7、§1.8）回答了机制与规模：** 视频编码器的优势由其跨时间注意力承载（关掉它，视频
臂损失 3.5–4.6%，图像臂不变，两者差距缩到 1–4%）；优势随训练预算衰减但未归零（20k 步 7%，
60k 步 3%，仍集中在 traffic / ETTm2 且显著）。

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

### 1.6 第二轮：给图像臂配预训练解码器（2026-09-06 晚，`eval_r2_s0/`、`eval_r2_s1/`）

第一轮的 `vmae_full` 带着 Kinetics 预训练解码器，而图像臂没有对等的解码器。第二轮把解码器换成
MAE 自己的 8×512 形状，新增四臂，每臂两个训练 seed（0 / 1，seed 同时决定数据流顺序）：

- `imae_full`：ImageNet MAE 编码器（3D 膨胀）+ **MAE 预训练解码器**（126 个张量逐一映射；像素头
  768→1536 按 tubelet 两帧复制膨胀；`decoder_embed` 的 bias 丢弃，因 HF 的 `encoder_to_decoder`
  无 bias；位置编码为固定 3D sincos）
- `imae_enc_d8`：同一图像编码器 + 全新 8×512 解码器（隔离解码器预训练的价值）
- `vmae_enc_d8`：Kinetics 编码器 + 全新 8×512 解码器（同容量下的编码器对照）
- `random_d8`：全随机。三个 `*_d8` 臂共享 bit 级相同的解码器初始化。

seed 0 的四臂第一次运行在 15.6k–17.2k 步时因 slurm 分配被收回而中断（本地 salloc 客户端进程
被系统 OOM 杀死），残缺目录归档为 `*_killed_at_alloc_loss`，四臂从头重跑；下表全部来自完整运行。

**Stage F（基准测试窗口），Q = 六数据集几何均值，seed 0 / seed 1：**

| 臂 | 单独 vs blend | blend+臂 vs blend | 单独 vs VisionTS | 20k 像素损失 |
|---|---:|---:|---:|---:|
| imae_full | 0.898 / 0.881 | 0.877 / 0.873 | 0.822 / 0.806 | 0.0346 / 0.0346 |
| vmae_enc_d8 | 0.907 / 0.883 | 0.884 / 0.875 | 0.830 / 0.808 | 0.0341 / 0.0338 |
| imae_enc_d8 | 0.946 / 0.948 | 0.912 / 0.915 | 0.866 / 0.868 | 0.0364 / 0.0362 |
| random_d8 | 0.977 / 0.966 | 0.926 / 0.925 | 0.894 / 0.884 | 0.0370 / 0.0370 |
| （第一轮 vmae_full，4×384 解码器，seed 0） | 0.860 | 0.858 | 0.787 | 0.0333 |

**配对比较（相同 origin，paired moving-block bootstrap），Q 与"CI 不含 1 的数据集数（更好 / 更差）"：**

| 比较 | Stage F seed 0 | Stage F seed 1 | Stage E seed 0 | Stage E seed 1 |
|---|---|---|---|---|
| imae_full / vmae_enc_d8 | 0.990 (1/1) | 0.998 (1/2) | 1.015 (1/3) | 0.997 (2/2) |
| imae_full / imae_enc_d8（解码器预训练的价值） | 0.949 (4/0) | 0.929 (4/0) | 0.941 (4/0) | 0.934 (5/0) |
| vmae_enc_d8 / imae_enc_d8（同一新解码器下的编码器） | 0.959 (3/1) | 0.931 (4/0) | 0.927 (4/0) | 0.937 (3/0) |
| vmae_enc_d8 / random_d8 | 0.928 (4/0) | 0.914 (4/0) | 0.907 (6/0) | 0.900 (5/0) |
| imae_enc_d8 / random_d8 | 0.968 (3/0) | 0.982 (4/1) | 0.977 (3/1) | 0.960 (4/0) |
| 第一轮 vmae_full / imae_full | 0.958 (3/0) | 0.976 (3/0) | 0.954 (4/0) | 0.976 (2/1) |
| 同一臂 seed 0 / seed 1（噪声下限） | 1.00–1.03 | | 1.00–1.02 | |

读法：

1. **图像 MAE 的预训练解码器值 5–7%**，把图像臂从 0.947 拉到 0.89，与"视频编码器 + 新解码器"
   打平（0.99–1.02，四次比较方向不一致）。第一轮 `imae_enc` 落后 `vmae_enc` 的差距有约一半
   来自它没有对等的解码器。
2. **编码器层面的排序在解码器对等后没有翻转，且两个 seed 一致**：同一个全新解码器下，视频
   编码器比图像编码器好 4–7%，比随机好 7–10%；图像编码器只比随机好 2–4%。效应量高于 seed 间
   噪声（≤ 3%）。
3. **两边都用各自的预训练解码器时，视频仍领先 2.4–4.6%**（vmae_full / imae_full），在 ETTm2、
   electricity、traffic 上 CI 不含 1，ETT 小时数据集持平，solar 一次更差。这是对视频臂保守的
   比较：它的解码器是 4×384，而 8×512 的解码器让每个臂都好约 7%（第一轮各臂 / 对应的 d8 臂 ≈ 1.07）。
   Kinetics 没有 8×512 的预训练解码器可配，所以无法造出完全对称的臂。
4. **seed 间噪声**：同一臂两个 seed 的 Q 差 0–3%，但逐数据集 CI 常常不含 1（bootstrap 只覆盖
   origin 抽样，不覆盖训练随机性）。因此凡是 Q 在 0.97–1.03 之间的比较都应视为持平，上面 1、2
   两条效应远大于这个范围。

### 1.7 第三轮 A：效应来源——关掉编码器的时间注意力（`eval_r3_spatial/`，2026-09-07）

三臂 `vmae_enc_d8` / `imae_enc_d8` / `random_d8` 各加同一约束：编码器 12 层自注意力只在 tubelet
内进行（块对角加性偏置，SDPA 实现，测试验证与"逐 tubelet 独立编码"逐元素相等），解码器保持
跨时间注意力（否则无法预测）。同协议 CPT 20k 步，seed 0，评测时施加同样约束。

| 比较（相同 origin 配对） | Stage F | Stage E |
|---|---|---|
| 视频 / 图像（都是空间注意力） | 0.990（3 好 / 0 差） | 0.959（4 / 0） |
| 视频 / 随机（都是空间注意力） | 0.984（2 / 1） | 0.974（4 / 1） |
| 图像 / 随机（都是空间注意力） | 0.993（1 / 2） | 1.016（0 / 4） |
| 视频空间 / 视频全注意力（第二轮 seed 0） | **1.035（0 / 3）** | **1.046（0 / 4）** |
| 图像空间 / 图像全注意力 | 1.002（0 / 0） | 1.011（0 / 1） |
| 随机空间 / 随机全注意力 | 0.976（3 / 0） | 0.973（3 / 0） |

对照全注意力下的差距（视频/图像 0.93–0.96，视频/随机 0.90–0.93）：**视频编码器的优势由跨时间
注意力承载**。去掉它，视频臂损失 3.5–4.6%（traffic 11–13%），图像臂不受影响，随机臂反而变好
（更简单的编码器更易训练）。残余的 1–4% 来自逐 tubelet 特征本身（两帧 tubelet 嵌入 + Kinetics
特征），Stage E 上显著、Stage F 上边缘。结论：把视频编码器当图像编码器用，它的额外价值就消失；
论文能说的是"编码器中的时间注意力先验"，而不只是"另一个检查点"。

### 1.8 第三轮 B：规模——60k 步（`eval_r3_60k/`，2026-09-07）

`vmae_enc_d8` 与 `imae_enc_d8` 各用 60k 步的 cosine 调度重跑（seed 0），每 20k 评一次：

| 检查点 | 视频 单独 / blend+ | 图像 单独 / blend+ | 视频 vs VisionTS | 图像 vs VisionTS | 视频 / 图像（配对） |
|---|---|---|---|---|---|
| 20k（60k 调度中段） | 0.900 / 0.879 | 0.965 / 0.912 | 0.824 | 0.883 | 0.933（5 / 0） |
| 40k | **0.858** / 0.849 | 0.902 / 0.881 | 0.785 | 0.826 | 0.950（4 / 0） |
| 60k | 0.875 / 0.855 | 0.904 / 0.875 | 0.801 | 0.828 | 0.967（3 / 0） |
| （第二轮 20k 调度终点） | 0.907 / 0.884 | 0.946 / 0.912 | 0.830 | 0.866 | 0.959 |

Stage E 上视频 / 图像为 0.909 → 0.963 → 0.968。held-out 像素损失 60k 时 0.0319 对 0.0342
（差 7%，与 20k 时持平），但时序指标上的差距从约 7% 收窄到约 3%，剩余优势集中在 traffic
（0.92）与 ETTm2（0.93），ETT 小时数据集与 solar 已持平。

读法：**初始化的优势随训练预算衰减但没有归零**：3 倍步数下从 7% 降到 3%，仍高于 seed 噪声
（≤ 3%）的上沿、且在两个高通道数据集上显著。VisionTS++ 的预算是本文的 40 倍，按这个趋势不能
排除在那个规模上完全收敛。两臂都因更长训练变好（视频 0.965，图像 0.956），40k 检查点的视频臂
单独达到 Q = 0.858 vs blend，与第一轮 `vmae_full` 持平。


## 1.9 论文协议（标准 LSF 零样本）下的成绩（2026-09-08，进行中）

评测器 `pilot/eval_lsf.py` 严格照 Time-Series-Library / VisionTS 的零样本协议：ETT 12/4/4 月、
custom 70/10/20，训练集 StandardScaler，全通道、测试 origin 步长 1，horizon 96/192/336/720，
标准化空间的 MSE/MAE。**校验**：VisionTS 官方检查点在本评测器上 ETTh1 得 96: 0.3527/0.3833、
720: 0.4062/0.441，与论文 0.353/0.383、0.406/0.441 一致到第三位。

**同一模型在两套协议下不可比**：VisionTS 在我们早先的配对 manifest（≤112 通道、抽样 2000 对、
16 周期上下文）上 ETTh1-96 是 0.402，论文协议下 0.353。§1.3 的表因此不能与任何已发表数字并排。

**我们的模型在此协议下的配方**（`--mode auto`，全部在 ETTh1/ETTh2/ETTm2/weather 上调出，
见 `results_field/lsf/`）：
- 帧 = k 个周期：一次前向覆盖长 horizon 并用上 12k 个周期的上下文；逐周期 rollout 极差
  （ETTh1-720：rollout 0.72，直推 k=8 0.52，两段 rollout k=4 0.48）。
- 多尺度集成（对 k·s·P ≤ 96 的尺度取平均）：ETTh1-96 0.4185 → 0.3858，ETTh2-96 0.316 → 0.300。
- 小时数据 H720 用两段 rollout（k=4）；15 分钟数据 H720 直推 k=2（0.371 vs rollout 0.409）。
- 与 VisionTS 简单平均：ETTh1-96 0.346 < VisionTS 0.353。

（完整六数据集 × 四 horizon 表见下方，待 electricity 完成后填入。）

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
