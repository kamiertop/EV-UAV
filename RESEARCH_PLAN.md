# EV-UAV 创新方案与实验记录

更新日期：2026-08-13

## 1. 研究定位

暂定题目：**面向事件微小目标检测的运动条件化稀疏卷积与双向轨迹一致性学习**

英文暂定：**Motion-Conditioned Sparse Convolution with Bidirectional Trajectory Consistency for Event-Based Tiny Object Detection**

核心问题：固定时空邻域难以同时适应慢速、快速、稀疏和转向轨迹。能否利用 EV-UAV 已有的实例与时间标注，在不引入持久同调的前提下，使稀疏卷积的尺度选择与目标运动状态耦合，同时直接抑制轨迹缺口和连续背景误检？

本方案有意避开同团队 CVPR 论文 SpTopoNet 的 TLM、SCM 和 EvTopoLoss。当前路线研究运动条件化尺度选择及双向困难事件，而不是持久同调拓扑匹配。

## 2. 已确认的代码与数据事实

- EV-UAV 位于 `/data/ev-uav`：99 个训练、24 个验证、24 个测试序列。
- 训练集约 932 万事件，前景比例约 2.94%；99 个训练序列中 56 个包含多个目标，单序列最多 27 个实例。
- 官方 GDSC 将通道固定分配给膨胀率 `(1,2,3,4)`，所有事件采用同一尺度分配。
- 公开 Patch Attention 的张量维度使序列长度为 1；当前 `sequence` 模式已按样本执行跨 patch 注意力，`legacy` 保留公开行为用于公平对照。
- 当前源码 `width=12` 的严格官方模式约 0.94M 可训练参数，与论文表中的 4.0M 不一致。论文写作前必须记录本仓库实测参数量和延迟，不照抄论文表格。
- 原论文表：EV-SpSegNet 的 IoU/ACC/Pd/Fa 为 55.18/65.02/77.53/1.63，SpTopoNet 为 66.62/74.44/83.37/1.29。这些只是论文报告值，必须与本仓库复现值分开列出。

## 3. 当前实现

### 3.1 Motion-Conditioned GDSC（MC-GDSC）

实现位置：`model/basemodel.py` 和 `model/evspsegnet.py`。

原 GDSC 的四个分组分支保留，以控制参数与计算量。新模块为每个活动体素预测局部速度与四分支路由：

\[
v_i=s\tanh(h_v(f_i)),\qquad
\alpha_i=\operatorname{softmax}(h_r(f_i,v_i,\|v_i\|)/\tau),
\]

\[
f'_i=\operatorname{Concat}_{r=1}^{4}
\left(4\alpha_{i,r} B_r(f^{(r)})_i\right).
\]

分支采用各向异性膨胀率：空间维默认 `(1,2,3,4)`，时间维默认 `(1,2,4,8)`。路由最后一层零初始化，初始时 `4 * softmax = 1`，与固定分支的输出尺度一致。速度头默认输出范围为 1.5 像素/ms，覆盖当前训练集由 50 ms 中心轨迹差分得到的速度范围；该值仍应在验证集做小范围消融。

默认只替换最高分辨率的 `encoder1`。原因：轨迹几何细节在浅层最完整，额外开销最低。`--motion_gd encoder/all` 只用于消融，不建议一开始作为主模型。

### 3.2 Motion-Bidirectional Trajectory Consistency（MBTC）

实现位置：`utils/mbtcloss.py`。

官方 STC 对局部支持低的正样本赋较小权重，又对局部支持高的负样本赋较小权重。这可能分别忽略轨迹缺口与连续假阳性。MBTC 改为：

\[
h_i=y_i(1-S_i^{gt})+(1-y_i)S_i^{pred},
\]

并用 `1 + lambda * h_i` 加权类平衡 focal loss。其余两项为：

- 轨迹置信度：同一实例相邻 50 ms 时间箱保持较高且平滑的置信度；
- 运动监督：从实例中心轨迹构造像素/毫秒速度，分别约束方向和速度。

结构路由采用弱批次均衡正则，避免所有事件塌缩到单个分支。实例 embedding/TACL 旧原型保留作历史消融，但不是当前主方案。

### 3.3 轨迹评价指标

实现位置：`utils/eval.py`。

- `trajectory_coverage`：成功检测的轨迹时间箱比例，越高越好；
- `trajectory_longest_ratio`：最长连续成功段占整条轨迹的比例，越高越好；
- `trajectory_fragmentation`：检测片段数/轨迹时间箱数，越低越好。

时间箱默认 50 ms。时间箱成功阈值只允许在验证集选择，固定后再运行测试集。

## 4. 公平实验矩阵

所有模型固定数据划分、种子、50 epoch、Adam、学习率 0.01 线性衰减到 0.001。先使用 seed 37 筛选，再对最终两种配置运行 seed 13/37/73。

| ID | Patch Attention | GDSC | Loss | 目的 |
|---|---|---|---|---|
| A0 | legacy | fixed | STC | 严格公开代码行为 |
| A1 | sequence | fixed | STC | 仅修复 Patch Attention |
| A2 | sequence | fixed | MBTC | 仅损失贡献 |
| A3 | sequence | MC-GDSC | STC | 仅结构贡献 |
| A4 | sequence | MC-GDSC | MBTC | 完整方法 |

第二阶段消融只在 A4 有明显收益后进行：

- `motion_gd = shallow/encoder/all`；
- 时间膨胀率 `(1,2,3,4)` 与 `(1,2,4,8)`；
- 分别令 `support/trajectory/motion/route` 权重为 0；
- 预测阈值和轨迹正确阈值只在验证集扫描一次并冻结。

## 5. 成败标准

以下标准针对**本仓库复现基线**，不能只与论文表中数字比较：

- 最低可继续：IoU 提升至少 1 个百分点，且轨迹覆盖或最长连续率相对提升 10%，Fa 不明显恶化；
- 稳妥投稿：IoU 提升 2 个百分点以上，ACC/Pd 至少一项提升，Fa 不恶化，轨迹指标改善；
- 如果 MC-GDSC 的 A3 不优于 A1，则结构动机未被支持，应停止扩大模块，保留 A2 作为损失方向；
- 如果 A2 有效而 A4 无额外收益，不能把动态卷积写入最终贡献；
- 如果只有修复 Attention 的 A1 提升，论文主张只能是复现/缺陷分析，不能宣称新模型有效。

## 6. 运行和结果管理

每个训练在 `runs/train_<run_name>_<timestamp>/` 保存：

- `config.json`：完整命令配置；
- `metrics.jsonl`：batch loss、验证 IoU/ACC/轨迹指标；
- `checkpoints/best_iou_seed*.pt`：验证集选出的 checkpoint。

训练默认不访问测试集。完成验证集模型选择后，用独立 `test.py` 命令评估一次测试集。不要依据测试结果继续调参。

## 7. 当前验证状态

已验证：Python 编译、参数解析、数据速度标签生成、Patch Attention 隔离性、MBTC 数值行为和轨迹指标，共 11 个 CPU 单元测试。

未验证：当前容器无法访问 NVIDIA 驱动，因此尚未完成 spconv CUDA 前向、反向、显存和真实训练收敛测试。上传服务器后必须先执行 1 epoch smoke test。
