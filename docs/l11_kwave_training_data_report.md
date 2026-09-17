# L11 k-Wave → neural_asp 数据适配报告

已完成 64 例真实 k-Wave RF 数据生成、PyTorch 打包和模型接口适配。没有通过标签叠加制造 B-mode 轮廓，也没有启动完整训练。

## 服务器位置

- 模型：`/home/zhuangyang/fmmodel/neural_asp`
- 配置：`/home/zhuangyang/fmmodel/neural_asp/configs/l11_kwave.yaml`
- 数据：`/data/zhuangyang/NumerialBreastPhantoms/l11_neural_asp_64`
- 就绪标记：数据目录中的 `READY`
- 详细检查：`dataset_report.json`、`validation_summary.json`
- smoke 运行：`/home/zhuangyang/fmmodel/neural_asp/runs/l11_kwave_smoke`

## 数据规格

| 项目 | 规格 |
|---|---|
| 划分 | train 48 / val 8 / test 8 |
| 训练病例 | Neg_07_Left 24 + Neg_35_Left 24 |
| 验证病例 | Neg_07_Left 4 + Neg_35_Left 4 |
| 测试病例 | 未见过的 Neg_47_Left 8 |
| 训练/验证层间隔 | 同病例至少 10 个原始层，约 2 mm |
| 仿真网格 | 50 μm，双尺度介质 |
| 发射 | 7.5 MHz，3 个平面波角度 −8°/0°/+8° |
| 接收 | 192 阵元，0.2 mm pitch，4–7.5 MHz |
| RF | float32 `[3,192,2401]`，40 MHz |
| D | complex64 `[3,53,192]`，`conj(FFT(rf))[band_idx]` |
| 真值网格 | `[216,192]`，0.2 mm |
| 真值 | `c`、`delta_s=1/c−1/1540`、complex64 `m` |
| shard 总大小 | 412,339,624 字节，约 412 MB |

宏观声速保留真实组织差异：脂肪 1455、腺体 1560、皮肤 1600、血管 1570 m/s。随机散射仅加入密度；腺体/脂肪/皮肤/血管相对 RMS 为 1.5%/0.3%/0.8%/0.15%，不加入随机声速散射。

均匀介质参考波场只计算一次并从总 RF 中减去，用于去除直达波。pilot 得到固定全局 RF 增益 `3.803263884986318e-9`，全部样本共用，不进行逐样本归一化。

## 验证结果

- 64/64 shard 的形状、有限性、声速/慢度范围、m RMS 和 RF→D FFT 一致性均通过。
- 最终项目测试：18 passed。
- RF RMS 范围：`4.0568e-7–5.6341e-7`。
- 声速范围：1455–1600 m/s；`|delta_s|max=3.7935e-5 s/m`；m RMS≈0.2。
- pilot 单步前向/反向成功，梯度全部有限；2.22 秒，峰值显存 0.52 GB。
- 正式 `train.py` 的 m/eta/joint 各 1 步成功，阶段恢复成功。
- 正式 `eval.py` 分批覆盖全部 8 个测试样本，成功输出指标与图像。

这些结果证明数据和模型流程可以运行，不证明已训练模型的重建精度。smoke checkpoint 只用于接口验证，不应作为正式模型。

## 完整训练命令（尚未执行）

在服务器模型目录执行：

```bash
cd /home/zhuangyang/fmmodel/neural_asp
CUDA_VISIBLE_DEVICES=0 /home/zhuangyang/miniconda3/envs/py310/bin/python train.py \
  --config configs/l11_kwave.yaml --stage all --out l11_kwave_64
```

配置默认每阶段 400 步、batch size 1。完整训练后可执行：

```bash
CUDA_VISIBLE_DEVICES=0 /home/zhuangyang/miniconda3/envs/py310/bin/python eval.py \
  --config configs/l11_kwave.yaml --ckpt runs/l11_kwave_64/joint.pt
```

## 重要限制与可恢复性

`m` 是归一化的高通 log-impedance 代理真值，而不是精确 Born 散射反演真值；真实全波 k-Wave RF 与 neural_asp 的 Born 前向模型仍存在模型失配，因此不能保证 RF 残差训练到零。

已保留原脚本备份（`*.pre_l11_dataset_20260915.bak`）。新增生成、打包、finalizer 和检查脚本位于模型目录 `scripts/`；原始 NPZ 与 PyTorch shard 都保留，支持跳过完成文件恢复生成。所有生成 worker 与 finalizer 已结束，没有启动完整训练任务。
