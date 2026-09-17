# L11 修复版使用说明

配置：`configs/l11_kwave_repaired.yaml`。原 `configs/l11_kwave.yaml` 和旧训练目录保留作为历史记录。

## 已修复

- 明确像素中心 `x0=-19.125 mm,z0=0.075 mm`、阵列中心及逐角度发射参考时刻。
- Born 发射、返回传播、伴随、展开残差、最终预测及慢度细化统一使用含固定系统响应的测量算子。
- 正确平面波延时和基带 IQ 的载频相位补偿；进入编码器前按样本 IQ RMS 归一化。
- 使用 210 个连续频率，加载时从 RF 重新计算 D，不改写旧 shard。
- m/eta 阶段显式冻结对应分支；eta 用声速监督；joint 使用完整声速权重、较小 RF 权重和学习率。
- 全验证集周期评估，并以初始化为候选保存最佳检查点。
- 检查点保存版本、配置及响应，拒绝旧算子检查点被悄悄重新解释。
- 主要图像指标只使用输入角度；全角度图和使用留出目标拟合增益的结果标为诊断。RF 叠图比较同频带波形，标题使用该单例指标。
- 新生成的 k-Wave raw metadata 保存发射时间、实际源中心频率和像素物理原点。

## 运行

在 `/home/zhuangyang/fmmodel/neural_asp` 中执行。以下命令用于独立的较长训练；短验证运行已另行完成，不代表全部阶段充分收敛。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 /home/zhuangyang/miniconda3/envs/py310/bin/python \
  scripts/calibrate_l11_response.py --config configs/l11_kwave_repaired.yaml

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 /home/zhuangyang/miniconda3/envs/py310/bin/python \
  train.py --config configs/l11_kwave_repaired.yaml --stage all --out l11_kwave_repaired_full

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 /home/zhuangyang/miniconda3/envs/py310/bin/python \
  eval.py --ckpt runs/l11_kwave_repaired_full/joint.pt
```

评估默认读取检查点中的配置和响应，避免错用原默认配置。版本不同的旧检查点须通过历史代码查看，不能继续作为新物理算子的兼容权重。

固定响应仅用 48 个训练样本的输入角度 −8°/0°估计，并记录 `data_cache/l11_kwave_response_v2.json`。它是基于代理散射标签的近似共享响应，仍不能替代严格的声学系统辨识。已停用对该代理 m 的直接复数 L1 监督，`lambda_I=0`；m 的相对幅值/结构仍在评估时检查。

短验证过程为 m 32 步、eta 400 步、joint 32 步。散射阶段位于 `runs/l11_kwave_repaired_check/stage_m.pt`，最终归一化声速训练及联合选优位于 `runs/l11_kwave_repaired_v2/`。16 个散射参数张量经逐项比较，在 eta 阶段完全保持不变。每次选择均覆盖 8 个验证样本，没有使用测试样本选模型。

备份目录：`backups/l11_consistency_v2_20260915_203750/`。原始数据和 `runs/l11_kwave_64/` 未覆盖。

最终 joint 32 步使验证声速 RMSE 从 43.10 升至 44.09 m/s，因此 `joint.pt` 的 `selected_step=-1`：它保存的是进入 joint 前的 eta 最佳权重。测试声速 RMSE 为 48.84 m/s，仍高于训练平均声速图基线 47.36；不能把该模型当作已解决声速结构恢复的最终结果。对同一组原 53 个频点计算的测试留出残差为 0.7248，原 joint 为 1.0806。

仍需扩充独立病例和角度；当前训练病例仅两例，验证来自同病例，测试切片均来自第三例。修复算子一致性不意味着声速结构和散射图已完全恢复。
