# Zero-shot 实验运行手册

本文档用于单独运行 WorldModel4TS 的 zero-shot / frozen-transfer 实验。命令以仓库根目录为工作目录，并与 `field-video-clean` 分支的 `86ece3a` 提交对应。

## 1. 实验定义

本仓库里有三种容易被混称为 “zero-shot” 的设置：

1. **Frozen backbone（本文主实验）**：加载 Kinetics-400 预训练的 VideoMAE，冻结整个 encoder，只用时间序列训练集训练预测头。运行参数是 `--pretrained 1 --tune frozen`。严格术语应是 frozen-feature linear probing，而不是完全无标签的 zero-shot。
2. **Frozen-feature probe**：不训练神经网络，只提取冻结特征并拟合闭式 ridge probe。入口是 `pilot/probe_frozen.py`；它仍使用训练标签拟合 ridge。
3. **Strict zero-shot**：模型参数和预测头都不使用时间序列标签训练。当前仓库中对应 Wan-VACE 视频修复实验，见第 8 节。

如果目标是复现 README 和 `PROGRESS.md` 中的 “zero-shot / frozen transfer” 结论，请运行第 2–7 节。

## 2. 环境

需要 Python 3.9+、CUDA GPU，以及 `transformers<5`。`uni` 模式建议至少约 16 GB 显存。

```bash
conda activate wm4ts
pip install "transformers==4.46.3" torch torchvision pandas numpy einops requests

python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"
```

设置数据、模型缓存和结果目录。集群上的数据目录可直接使用第一种写法；本地运行时改用第二种。

```bash
# Northwestern 集群
export WM4TS_DATA_DIR=/nyx-storage1/hanliu/wm4ts/data

# 本地数据若放在仓库的 data/ 下，则使用：
# export WM4TS_DATA_DIR="$PWD/data"

export HF_HOME="$PWD/hf_cache"
export WM4TS_ZEROSHOT_OUT="$PWD/pilot/results_field/zeroshot"
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$HF_HOME" "$WM4TS_ZEROSHOT_OUT"
```

第一次运行会下载 `MCG-NJU/videomae-base`，约 350 MB。也可以指定已有 checkpoint：

```bash
export VMAE_CKPT=/absolute/path/to/videomae-base
```

检查数据文件：

```bash
for file in ETTh1.csv ETTh2.csv ETTm1.csv ETTm2.csv electricity.txt traffic.txt solar_AL.txt; do
  test -f "$WM4TS_DATA_DIR/$file" && echo "OK  $file" || echo "MISSING  $file"
done
```

## 3. 先跑一个 smoke test

下面的命令只跑少量训练窗口，用于检查环境、数据、checkpoint 和输出路径。它不能作为正式结果。

```bash
python pilot/run_field.py \
  --dataset ETTh1 \
  --data-dir "$WM4TS_DATA_DIR" \
  --out-dir "$WM4TS_ZEROSHOT_OUT/smoke" \
  --mode uni \
  --backbone video \
  --render period \
  --pretrained 1 \
  --tune frozen \
  --context-steps 384 \
  --horizon-steps 96 \
  --max-ch 7 \
  --stride 32 \
  --ft-cap 256 \
  --epochs 1 \
  --batch 16 \
  --lr 1e-4 \
  --seed 0
```

日志中必须出现以下信息：

- `tune=frozen`；
- trainable 参数只占很小比例；
- `snaive`、`smean` 和 `video_uni_frozen_s0` 三组指标；
- 最后生成 `field_ETTh1_uni_video_frozen_L384_h96_s0.json`。

## 4. 正式单数据集实验

以下配置复现项目已有的 frozen-backbone 对比协议：VideoMAE 预训练 encoder 全冻结，只训练预测头；最多使用 112 个通道；预测步长为 96；训练 5 个 epoch。

```bash
python pilot/run_field.py \
  --dataset electricity \
  --data-dir "$WM4TS_DATA_DIR" \
  --out-dir "$WM4TS_ZEROSHOT_OUT/frozen" \
  --mode uni \
  --backbone video \
  --render period \
  --pretrained 1 \
  --tune frozen \
  --context-steps 384 \
  --horizon-steps 96 \
  --max-ch 112 \
  --stride 8 \
  --ft-cap 40000 \
  --epochs 5 \
  --batch 16 \
  --lr 1e-4 \
  --seed 0
```

输出文件为：

```text
pilot/results_field/zeroshot/frozen/field_electricity_uni_video_frozen_L384_h96_s0.json
```

JSON 中应同时包含 `snaive`、`smean` 和 `video_uni_frozen_s0`，所有 MSE/MAE 都是在训练集统计量标准化后的尺度上计算。

## 5. 批量跑七个数据集

默认 context 是 `16 × 数据周期 P`：ETTh1/ETTh2/electricity/traffic 为 384，ETTm1/ETTm2 为 1536，solar 为 2304。除 solar 使用 h=144 外，其余数据集使用 h=96。

```bash
for dataset in ETTh1 ETTh2 ETTm1 ETTm2 electricity traffic; do
  python pilot/run_field.py \
    --dataset "$dataset" \
    --data-dir "$WM4TS_DATA_DIR" \
    --out-dir "$WM4TS_ZEROSHOT_OUT/frozen" \
    --mode uni \
    --backbone video \
    --render period \
    --pretrained 1 \
    --tune frozen \
    --horizon-steps 96 \
    --max-ch 112 \
    --stride 8 \
    --ft-cap 40000 \
    --epochs 5 \
    --batch 16 \
    --lr 1e-4 \
    --seed 0
done

python pilot/run_field.py \
  --dataset solar \
  --data-dir "$WM4TS_DATA_DIR" \
  --out-dir "$WM4TS_ZEROSHOT_OUT/frozen" \
  --mode uni \
  --backbone video \
  --render period \
  --pretrained 1 \
  --tune frozen \
  --horizon-steps 144 \
  --max-ch 112 \
  --stride 8 \
  --ft-cap 40000 \
  --epochs 5 \
  --batch 16 \
  --lr 1e-4 \
  --seed 0
```

重复相同配置和 seed 会覆盖同名 JSON。需要多 seed 时，将 `--seed 0` 分别改为 `0`、`1`、`2`；文件名会自动包含 seed。

## 6. Frozen-feature diagnostic

该脚本不做神经网络梯度训练，会比较三种渲染的冻结特征有效维度和 ridge-probe skill。先从 electricity 开始：

```bash
python pilot/probe_frozen.py \
  --dataset electricity \
  --data-dir "$WM4TS_DATA_DIR" \
  --max-ch 112 \
  --n 1500 \
  --horizon-p 4
```

结果固定写到 `pilot/results_field/probe/probe_electricity.json`。其中：

- `partic_ratio`：冻结特征的有效维度，越高表示特征坍缩越弱；
- `skill_ratio = probe_mse / const_mse`：越低越好，`1.0` 表示不优于常数预测；
- `uni_barcode`、`vts_2d`、`field_2d`：三种相同时间序列输入的渲染方式。

## 7. 结果核验与对照

正式结果至少检查以下项目：

1. JSON 的 `config.pretrained` 必须为 `1`，`config.tune` 必须为 `frozen`。
2. 主模型必须与同一 JSON 中的 `smean` 比较，不能引用另一套窗口上的 baseline。
3. 日志中的 frozen encoder 参数不应被列为 trainable。
4. 报告结果时使用 `model MSE / smean MSE`，并明确写作 **frozen-backbone supervised probe**。

已有 seed-0 结果可作为粗略 sanity check，而不是新的实验结论：electricity 约 `0.301`、traffic 约 `0.607`、ETTh1 约 `0.496`、solar（h=144）约 `0.226`。如果偏差很大，优先检查 `transformers` 版本、数据切分、context、horizon、通道数和 `--pretrained`。

要测量 Kinetics 预训练本身的贡献，可保持其余参数完全一致，仅增加随机初始化对照：

```bash
python pilot/run_field.py \
  --dataset electricity \
  --data-dir "$WM4TS_DATA_DIR" \
  --out-dir "$WM4TS_ZEROSHOT_OUT/random_control" \
  --mode uni --backbone video --render period \
  --pretrained 0 --tune frozen \
  --context-steps 384 --horizon-steps 96 \
  --max-ch 112 --stride 8 --ft-cap 40000 \
  --epochs 5 --batch 16 --lr 1e-4 --seed 0
```

## 8. Strict zero-shot：Wan-VACE

`pilot/run_vace_ts.py` 不在时间序列上训练任何模型或预测头，而是把 ETTh1 context 渲染成视频、遮住未来帧，再让预训练 Wan-VACE 直接补全。它目前固定使用 ETTh1、context=288、horizon=96。

这一实验需要单独的 VACE/Wan 环境、可用 checkpoint 和约 48 GB 显存。仓库中的 VACE fork 必须可导入；不要在普通 `wm4ts` 环境中直接运行。

```bash
conda activate open-sora
export VACE_DIR="$PWD/VACE"
export CUDA_VISIBLE_DEVICES=0

python pilot/run_vace_ts.py \
  --data-dir "$WM4TS_DATA_DIR" \
  --ckpt-dir /absolute/path/to/Wan2.1-VACE-1.3B \
  --model-name vace-1.3B \
  --out-dir "$WM4TS_ZEROSHOT_OUT/vace" \
  --n 6 \
  --fpp 4 \
  --steps 15 \
  --guide 1.0 \
  --context-scale 1.0 \
  --size 480p
```

结果写到 `vace_ETTh1_vace-1.3B_cs1.0.json`，并包含同一批窗口上的 `smean` 对照。先用 `--n 1 --steps 5` 做环境 smoke test；正式运行时换回目标 `n` 和 steps，并使用不同的 `--out-dir`，以免 smoke test 覆盖正式 JSON。

## 9. 常见问题

- `transformers>=5`：当前代码会拒绝运行；固定为 `4.46.3`。
- 找不到 CSV/TXT：确认 `--data-dir` 指向直接包含数据文件的目录，而不是其上一级。
- 首次启动卡住：通常正在下载 Hugging Face checkpoint，检查 `$HF_HOME` 和网络。
- CUDA OOM：先减小 `--batch`；不要修改数据协议中的 `max-ch`、context 或 horizon 后再与原结果直接比较。
- Slurm 任务被系统杀死：申请足够 CPU 内存；项目记录的稳定配置是 `-c 32 --mem=96G`。
- `--context-steps` 报错：它必须是数据周期 `P` 的整数倍。

