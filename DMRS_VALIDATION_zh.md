# 已知 DM-RS 输入与退化验证

当前架构为 `pdsch_dmrs_v1`：保留原有状态初始化、6 轮层间消息更新和符号/bit 双头，
把两层已知发送 DM-RS 直接作为输入。不增加信道估计网络，不恢复传统信道估计或 LMMSE。

## 输入与结构

每个空间流分支的 evidence 包括：

| 特征 | 默认通道数 |
| --- | ---: |
| 两根接收天线的接收 IQ | 4 |
| 两层已知发送 DM-RS 的 IQ | 4 |
| 两层 DM-RS 掩码 | 2 |
| 当前输出层的 one-hot 标识 | 2 |
| 当前层数据掩码 | 1 |
| 归一化噪声的对数 | 1 |
| Qm / 码率 / RB 条件嵌入 | 16 |
| 合计 | 30 |

原无导频版本为 24 通道。两层分支都能看到所有层参考，而不是只看到各自导频。
发送参考保留原始发送幅度；接收 IQ 和噪声沿用原有 RMS 归一化。接收导频 RE 不清零。
符号仍直接预测，损失仍为 masked BCE + 0.2 × 归一化符号 MSE，标签不作为输入。

训练/验证/推理均必须提供：

- `dmrs_grid`：`[N,S,K,L,2]`，最后一维为实部/虚部。
- `dmrs_mask`：`[N,S,K,L]`，二值；与同层 `data_mask` 不重叠。
- 每层至少有一个非零导频，参考值必须有限，掩码外必须为零。

加载器对分片做形状检查，并在取样时检查数值和掩码。缺字段会报错，不会默默回退到盲检测。
保留 MATLAB/Python 轴转换、按 RB 裁剪和 batch 补零。`channel_est` 仍不读取，训练也不读取 `traditional_eq`。
`matlab/writeNeuralRxH5.m` 已增加必需的 `dmrsGrid` / `dmrsMask` 字段。

## 重新训练

旧 CSI/LMMSE 及 `pdsch_no_csi_v1` 检查点不能续训当前模型。
新增 HDF5 字段也会改变数据指纹，因此使用新输出目录重新生成划分。
默认输出目录已改为 `runs/dmrs`，数据目录仍为 `dataset_h5`。

在工程目录用已安装的 PyTorch 环境运行：

```powershell
& D:/Anaconda_envs/envs/pytorch_env/python.exe train.py --data dataset_h5 --output-dir runs/dmrs --device cuda --batch-size 4
```

本次没有擅自修改学习率、训练轮数或损失权重，便于先比较增加导频输入的影响。
`history.csv` 新增 `train_neural_symbol_power_ratio` / `validation_neural_symbol_power_ratio`，
表示有效数据 RE 上预测符号总功率与标签总功率之比。若它接近 0 且 EVM 约 100%，
支持符号输出趋零的诊断；它接近 1 也不代表相位、层顺序或比特一定正确。

## 检查数据和导频依赖

```powershell
& D:/Anaconda_envs/envs/pytorch_env/python.exe inspect_dataset.py --data dataset_h5 --max-samples 30 --output results/dmrs_audit.json

# 同一模型、同一验证集，分别使用提供的参考、清零参考、90°旋转参考。
foreach ($mode in @('correct', 'zero', 'phase_rotate')) {
    & D:/Anaconda_envs/envs/pytorch_env/python.exe validate.py --data dataset_h5 --checkpoint runs/dmrs/best.pt --splits runs/dmrs/splits.npz --config runs/dmrs/config.json --split validation --dmrs-mode $mode --output-dir "results/dmrs_$mode" --device cuda --batch-size 4
}
```

`correct` 表示使用数据集提供的参考，并不代表程序已经证明其物理正确性。
`zero` 同时清零参考 IQ 与导频掩码；`phase_rotate` 保留位置，将参考乘以 j，保持接收信号和标签不变。
验证摘要会记录 `dmrs_mode`，避免混淆结果。

如果正确参考在留出数据上显著优于两种对照，说明模型利用了参考信息。
如果训练集改善而验证集仍近随机，优先排查过拟合、样本分布和序列对齐。
错误参考属于分布变化，性能变差本身不构成参考准确性的充分证明；还需要与发送端
slot、序列初始化、端口/OCC、预编码及功率缩放记录逐项对齐。
全局导频矩阵满秩也不保证每个局部信道区域都有足够的可分离导频。

## 有限步数的小样本对照

新增 `probe_dmrs.py`，从指定 Qm/RB/SNR 范围选择互不重叠的少量训练与验证样本。
从相同随机初始化出发，分别训练“使用参考”和“参考/掩码清零”的模型，训练步数、样本顺序和优化器一致。
两者均在训练/验证样本上执行三种参考对照。实验只输出 JSON，不覆盖正式训练检查点。

```powershell
& D:/Anaconda_envs/envs/pytorch_env/python.exe probe_dmrs.py --qm 2 --num-rbs 10 --min-snr 20 --train-samples 16 --validation-samples 16 --steps 200 --batch-size 4 --output results/dmrs_probe_trial.json
```

输出包含样本索引、配置、训练条件和各组指标。默认实验规模很小，只用于检查可学习性和参考依赖。
互不重叠的样本索引并不保证不同信道轨迹；独立信道泛化仍需正式划分。

## 本次验证结果

- 自动测试覆盖导频选择性读取、MATLAB/Python 轴转换、变长补零、错误参考拒绝、两层参考梯度、
  导频扰动影响、旧检查点拒绝，以及训练/续训/验证/推理/星座导出流程；8 项测试全部通过。
- 全部 450 个分片的必需字段与形状检查通过。30 个均匀抽取的真实样本结构审计为 PASS；
  每层累计 23,328 个导频，平均参考功率约 1，未发现全局秩不足。报告在 `results/dmrs_audit.json`。
- 在真实数据上用默认网络、CUDA AMP 执行 200 步小样本对照：16 个训练样本、16 个留出样本，
  QPSK、10 RB、SNR ≥ 20 dB。报告在 `results/dmrs_probe_initial.json`。

| 训练条件 | 评估集合 | 评估参考 | EVM | coded BER |
| --- | --- | --- | ---: | ---: |
| 使用参考 | 训练 | 提供的参考 | 87.00% | 0.23121 |
| 使用参考 | 训练 | 清零 | 88.23% | 0.24969 |
| 使用参考 | 留出 | 提供的参考 | 111.66% | 0.48311 |
| 使用参考 | 留出 | 清零 | 112.24% | 0.48338 |
| 使用参考 | 留出 | 旋转 90° | 111.67% | 0.48504 |
| 清零参考 | 训练 | 清零 | 87.76% | 0.24368 |
| 清零参考 | 留出 | 清零 | 111.50% | 0.48370 |

这些结果确认训练链路可运行，但**尚未证明增加参考已经打破泛化退化**：
该小样本短程训练在留出集合上仍接近随机，两种参考条件差距很小。
不能据此断言参考有误，也不能据此声称模型已经学会正确利用 DM-RS。
需要进一步核对回填参考的 slot/端口/幅度对齐，并在更有代表性的训练集上观察验证 BER/EVM 与消融差距。
