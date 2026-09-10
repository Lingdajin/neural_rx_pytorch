# PyTorch 版 2x2 MIMO PDSCH 神经接收机

本目录是独立于原 TensorFlow/Sionna PUSCH 运行环境的新实现。模型保留了
CGNN 的主要结构：状态初始化、层间消息聚合、多轮卷积状态更新和读出头，
并针对 MATLAB 采集的 PDSCH 数据改为监督训练。原 TensorFlow 权重不能直接
加载，需要使用 PDSCH 数据重新训练。

模型以 `Qm`、目标码率和 RB 数作为条件输入，卷积权重不依赖固定带宽；不同
RB 的网格会在 batch 内补零并用 `data_mask` 屏蔽，因此一个检查点可以覆盖
多种 MCS 和 RB 配置。

## 1. 安装

建议使用 Python 3.10，并安装与本机 CUDA 对应的 PyTorch：

```bash
cd neural_rx_pytorch
python -m venv .venv
pip install -r requirements.txt
```

## 2. MATLAB 数据

使用 `matlab/writeNeuralRxH5.m` 写 HDF5 文件。大数据集可拆成多个 `.h5`
分片，训练时传入目录即可流式读取，不会一次载入全部数据。

必需的训练数据如下：

| 字段 | MATLAB 逻辑形状 | 说明 |
| --- | --- | --- |
| `rx_grid` | `[N,S,K,2,2]` | 两根接收天线的频域网格，末维为实部/虚部 |
| `channel_est` | `[N,S,K,2,2,2]` | 部署接收机给出的有效信道 `H(rx,layer)` |
| `data_mask` | `[N,S,K,2]` | PDSCH 数据 RE 为 1，导频/PTRS/补零为 0 |
| `tx_symbols` | `[N,S,K,2,2]` | 层映射后的发送 QAM 符号 |
| `coded_bits` | `[N,S,K,2,Qmax]` | 速率匹配和加扰后的 mapper 输入 bit |
| `traditional_eq` | `[N,S,K,2,2]` | 可选，MATLAB 传统接收机均衡符号 |

每个样本还需提供 `mcs_index`、`qm`、`code_rate`、`num_rbs` 和
`noise_var`；建议额外提供 `snr_db`。完整定义与 bit 顺序要求见
[DATA_FORMAT.md](DATA_FORMAT.md)；样本数量与分布规划见
[DATA_COLLECTION_GUIDE_zh.md](DATA_COLLECTION_GUIDE_zh.md)。
特别注意：`coded_bits` 不是传输块原始 bit，
必须与对应 `tx_symbols` 的 QAM mapper 输入逐 RE 对齐。

## 3. 训练与续训

训练前建议先审计全部数据：

```bash
python inspect_dataset.py --data dataset_h5 --output dataset_report.json
```

只有报告状态为 `PASS` 或已确认 `WARN` 原因后再开始训练。

```bash
python train.py --data data --output-dir runs/pdsch_2x2 --device cuda
```

训练脚本按 `(MCS,RB)` 分层生成固定的 80%/10%/10% 训练、验证和测试集。如果
`snr_db` 是有足够重复样本的离散扫描值，还会自动按 SNR 分层，确保每个 SNR
均进入三个子集；连续随机 SNR 则自动退回 `(MCS,RB)` 分层。
比例和网络参数可修改 `config/default_2x2.json`。主要输出为：

- `best.pt`：验证 loss 最优检查点；
- `last.pt`：最近一轮完整训练状态；
- `loss.png` 和 `history.csv`：训练曲线与所有指标；
- `splits.npz`：固定数据划分及数据集指纹；
- `config.json`：本次训练配置副本。

续训命令：

```bash
python train.py --data data --output-dir runs/pdsch_2x2 \
  --config runs/pdsch_2x2/config.json \
  --resume runs/pdsch_2x2/last.pt --device cuda
```

## 4. 验证和测试

```bash
python validate.py --data data \
  --checkpoint runs/pdsch_2x2/best.pt \
  --splits runs/pdsch_2x2/splits.npz \
  --config runs/pdsch_2x2/config.json \
  --split validation --output-dir results/validation
```

将 `--split validation` 改成 `--split test` 可运行最终测试集。输出包含 coded
BER、神经接收机 EVM、传统接收机 EVM，以及按 MCS/RB/SNR 的 CSV 明细。

这里的 BER 是 LDPC 前的 coded BER。正式性能结论应将网络输出 LLR 接回已有
MATLAB PDSCH 解扰、速率恢复、LDPC 和 CRC 流程，绘制各 MCS/RB/SNR 下的 BLER。

## 5. 导出 EVM/星座数据

```bash
python export_constellation.py --data data \
  --checkpoint runs/pdsch_2x2/best.pt \
  --splits runs/pdsch_2x2/splits.npz \
  --config runs/pdsch_2x2/config.json \
  --split test --output results/constellation.mat
```

输出 `.mat` 和 `.npz`，包含 `reference`、`neural`、`traditional` 三组复数星座
点以及两种 EVM。数据中存在 `traditional_eq` 时优先使用真实 MATLAB 传统
均衡结果；否则使用同一 `channel_est/noise_var` 计算的内部 LMMSE 基线。

## 6. MATLAB 调用

运行 `matlab/exampleCallNeuralRx.m`。示例将 MATLAB 的频域网格、信道估计和
调度信息写入临时 HDF5，调用 `infer.py`，再读取：

- `equalized_symbols`：神经接收机均衡星座；
- `traditional_lmmse_symbols`：LMMSE 对照星座；
- `bit_llr_positive_one`：软 bit，正值表示 bit 1；
- `data_mask`：有效数据 RE 掩码。

文件桥接便于验证接口，不适合逐 slot 实时运行。完成数值对齐后，应改为常驻
Python 服务或另行导出部署模型，以免重复启动 Python 进程。
