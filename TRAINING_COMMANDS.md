# 服务器训练命令

以下命令均在项目根目录执行。数据默认读取 `/data/ev-uav`。

## 0. 上传后先检查

```bash
git status --short
uv sync
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python -m unittest discover -s tests -v
nvidia-smi
```

如果服务器还没有编译 `HAIS_OP.so`，按项目原有方式执行：

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python scripts/build_ext.py
```

## 1. 一轮 CUDA 烟雾训练

先验证完整方法能够前向、反向、验证并保存 checkpoint。建议先用 4090 的空闲 GPU 编号替换 `0`：

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python train.py \
  --gpu 0 --data_dir /data/ev-uav \
  --run_name smoke_a4 --epochs 1 --train_workers 0 \
  --patch_attention sequence --motion_gd shallow --loss mbtc
```

检查最后输出不含 CUDA/spconv 错误，并确认新目录中存在：

```bash
find runs -maxdepth 3 -type f | sort | tail -n 20
```

## 2. 第一阶段五组单种子消融

可在 3090 和 4090 上并行运行不同配置，但同一配置的计时比较必须使用相同 GPU。

### A0：严格公开行为

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python train.py \
  --gpu 0 --data_dir /data/ev-uav --run_name a0_official_s37 \
  --seed 37 --epochs 50 --patch_attention legacy --motion_gd none --loss stc
```

### A1：仅修复 Patch Attention

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python train.py \
  --gpu 0 --data_dir /data/ev-uav --run_name a1_attention_s37 \
  --seed 37 --epochs 50 --patch_attention sequence --motion_gd none --loss stc
```

### A2：仅 MBTC 损失

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python train.py \
  --gpu 0 --data_dir /data/ev-uav --run_name a2_mbtc_s37 \
  --seed 37 --epochs 50 --patch_attention sequence --motion_gd none --loss mbtc
```

### A3：仅 MC-GDSC 结构

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python train.py \
  --gpu 0 --data_dir /data/ev-uav --run_name a3_mcgds_s37 \
  --seed 37 --epochs 50 --patch_attention sequence --motion_gd shallow --loss stc
```

### A4：完整方法

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python train.py \
  --gpu 0 --data_dir /data/ev-uav --run_name a4_full_s37 \
  --seed 37 --epochs 50 --patch_attention sequence --motion_gd shallow --loss mbtc
```

训练命令默认只使用 train/val，不会自动评估 test。

## 3. 找到每组最佳验证结果

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python scripts/summarize_runs.py
```

脚本会选择每个运行的最高验证 IoU，而不是最后一个 epoch；同时列出轨迹指标。

## 4. 最终配置运行三个种子

如果 A4 优于 A1/A2/A3，再运行：

```bash
for seed in 13 37 73; do
  UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python train.py \
    --gpu 0 --data_dir /data/ev-uav --run_name "a4_full_s${seed}" \
    --seed "$seed" --epochs 50 \
    --patch_attention sequence --motion_gd shallow --loss mbtc
done
```

基线 A1 也应运行相同三个种子，才能报告均值和标准差：

```bash
for seed in 13 37 73; do
  UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python train.py \
    --gpu 0 --data_dir /data/ev-uav --run_name "a1_attention_s${seed}" \
    --seed "$seed" --epochs 50 \
    --patch_attention sequence --motion_gd none --loss stc
done
```

## 5. 最终测试集评估

先从对应运行目录读取 `config.json`，确保评测参数与训练完全一致。下面以完整方法为例：

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python test.py \
  --gpu 0 --data_dir /data/ev-uav \
  --model_path runs/train_a4_full_s37_YYYYMMDD_HHMMSS/checkpoints/best_iou_seed37.pt \
  --patch_attention sequence --motion_gd shallow --loss mbtc
```

会输出 IoU、ACC、Pd、Fa、trajectory coverage、longest ratio 和 fragmentation。

如果评估旧的官方 checkpoint，必须使用其对应结构：

```bash
UV_CACHE_DIR=/tmp/ev-uav-uv-cache uv run python test.py \
  --gpu 0 --data_dir /data/ev-uav \
  --model_path /path/to/official_checkpoint.pt \
  --patch_attention legacy --motion_gd none --loss stc
```

## 6. 常见故障

- `No CUDA GPUs are available`：容器未挂载 GPU，检查 `nvidia-smi` 和启动参数。
- `HAIS_OP` 导入失败：重新运行 `scripts/build_ext.py`，确认 CUDA/GCC 与服务器环境匹配。
- DataLoader worker 崩溃：先加 `--train_workers 0`；稳定后再恢复 4 或 8。
- OOM：项目默认 batch size 为 1；仍 OOM 时降低 `--max_events_num`，但所有对照必须使用相同值。
- deterministic/spconv 报错：保留完整错误日志，不要直接关闭确定性设置后与旧结果混用。
