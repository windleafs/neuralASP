# UltraWave L11 独立500例、11角度

新数据目录：`/data/zhuangyang/NumerialBreastPhantoms/l11_ultrawave_500_11angle`

模型配置：`/home/zhuangyang/fmmodel/neural_asp/configs/l11_ultrawave_500_11angle.yaml`

旧64例数据及旧配置保持不变。新样本全部来自 UltraWave 二维线性声学全波，不是 Born 仿真，不含 BonA 非线性。11个角度−8°到+8°、间隔1.6°，全部RF保留；模型延续8角度训练/3角度留出验证。

## 规格与pilot

- 500个不同的病例/切片组合，无重复解剖补种子；但来源仅3个病例，不是500个独立患者。
- train400 / val50 / test50；训练和验证同病例最小切片间距10层，约2mm；测试只来自Neg_47_Left。
- 50μm仿真网格、8阶空间差分、2.5ns原生时间步长、24001点。
- 192阵元、0.2mm pitch、7.5MHz发射、4–7.5MHz接收、40MHz RF采样。
- RF float32 `[11,192,2401]`；D complex64 `[11,53,192]`；c/delta_s/m真值网格 `[216,192]`、0.2mm。
- 专属11角度参考波场和一次固定全局增益 `4.546855450439301e-9`，不逐样本归一化。
- 22项回归测试通过；pilot87.6秒；truth和joint单步反向通过，joint峰值显存4.33GB。
- 自然B-mode中5–35mm腺体/脂肪分层对比6.84–12.67dB，35–40mm约4.99dB。图像没有轮廓/标签叠加。

## 生成状态与完成标志

批量生成在GPU1/2后台运行，与SSH会话无关。pilot吞吐推算约5–6小时，批量实际时间可能不同。最终生成器完成后，finalizer自动打包500例、逐例校验、正式m/eta/joint各1步和全部50例测试评估smoke，生成`dataset_report.json`后才写`READY`。在`READY`出现前，不能把数据视作训练就绪。

检查进度：

```bash
cd /home/zhuangyang/fmmodel/neural_asp
/home/zhuangyang/miniconda3/envs/py310/bin/python scripts/generate_l11_ultrawave_raw.py --mode status
tail -n 10 /data/zhuangyang/NumerialBreastPhantoms/l11_ultrawave_500_11angle/finalizer.log
```

进程PID与日志在新数据目录`processes.json`、`worker0.log`、`worker1.log`和`finalizer.log`。错误记录在`raw/*.error.txt`或`FINALIZATION_FAILED.txt`。恢复前先检查错误和已运行进程，不要重复启动worker；完成文件会校验后跳过。

## 完整训练与eval命令（尚未执行）

确认`READY`存在后，选择空闲GPU运行：

```bash
cd /home/zhuangyang/fmmodel/neural_asp
CUDA_VISIBLE_DEVICES=1 /home/zhuangyang/miniconda3/envs/py310/bin/python train.py \
  --config configs/l11_ultrawave_500_11angle.yaml \
  --stage all --out l11_ultrawave_500_11angle
```

完整训练结束后：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zhuangyang/miniconda3/envs/py310/bin/python eval.py \
  --config configs/l11_ultrawave_500_11angle.yaml \
  --ckpt runs/l11_ultrawave_500_11angle/joint.pt \
  --out runs/l11_ultrawave_500_11angle/eval
```

完整模型训练不会自动启动。生成完成后的smoke checkpoint只用于接口验证，不是正式训练模型。`m`是高通log-impedance代理真值，神经模型Born前向与全波RF存在模型失配；50μm有限差分及二维离面缺失仍是精度限制。
