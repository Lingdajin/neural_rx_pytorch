# 基于 PyTorch 的 2×2 MIMO PDSCH 神经接收机算法详细设计报告

## 0. 文档范围与实现基线

本文档针对当前仓库中的 PyTorch 实现编写，描述 `neural_rx_pytorch/model.py`、`data.py`、`losses.py`、`engine.py` 以及 `train.py`、`infer.py` 的实际算法和数据流。系统面向 **2 根接收天线、2 个空间层的 PDSCH**，输入是 OFDM 解调后的频域资源栅格、部署侧信道估计和噪声方差，输出是每个数据资源单元（RE）的均衡 QAM 符号及 coded bit 软信息。

符号约定如下：

- $B$：批大小；$S$：OFDM 符号数；$K$：有效子载波数，通常为 $12N_\mathrm{RB}$；
- $R=2$：接收天线数；$L=2$：空间层数；$Q_\max\leq 10$：模型支持的最大每符号比特数；
- 复数使用末维 `[实部, 虚部]` 表示；网络输出的 bit logit 遵循“**正值表示 bit=1**”的约定。

当前模型是一个带物理先验的条件化迭代网络：先用 LMMSE 得到可解释的初始符号估计，再使用卷积状态和层间消息传递逐轮修正符号并预测 bit。网络权重在 MCS 和 RB 配置之间共享，通过 $Q_m$、码率和 RB 数条件化。

## 1. 设计目标

### 1.1 功能目标

1. **完成 2×2 MIMO PDSCH 的数据 RE 检测。** 对每个有效 RE、每个空间层输出复数均衡符号和最多 10 个 coded bit 的 logit。
2. **保留传统接收机的物理先验。** 信道估计和噪声方差首先进入 LMMSE 求解，神经网络学习残余干扰、信道估计误差和非理想条件下的修正量。
3. **联合服务软解调和星座均衡。** 训练目标同时包含 bit BCE 和符号误差，使输出既可用于后续 PDSCH 软译码，也可用于 EVM/星座分析。
4. **支持可变带宽和可变调制。** 采用全卷积结构，样本在 batch 内按最大 $S,K$ 补零，使用 `data_mask` 屏蔽无效位置；同一个检查点可覆盖多个 $Q_m$、码率和 RB 分配。
5. **提供 MATLAB/PyTorch 桥接接口。** 训练数据、推理输入采用统一 HDF5 合同，推理结果写为 MATLAB 可直接读取的 MAT 文件。

### 1.2 性能与工程目标

- 在网络规模、显存占用和跨场景泛化之间取得平衡。默认配置为隐藏通道 64、条件通道 16、迭代 6 轮，卷积使用深度可分离卷积降低计算量。
- 通过流式 HDF5 读取、按样本裁剪有效子载波和按 batch 动态补零，避免一次加载全部数据，也避免用最大带宽固定计算所有样本。
- 采用确定性的 `(MCS, RB[, SNR])` 分层划分，固定数据指纹和随机种子，保证训练、验证和测试可复现。
- 在训练中使用 AdamW、梯度裁剪、可选 CUDA AMP、验证集 ReduceLROnPlateau 和早停，降低大规模数据训练的工程风险。

### 1.3 适用边界

当前代码明确限制 `num_rx_antennas=2`、`num_layers=2`。它输出的是 LDPC 前的 coded BER；完整 BLER 需要把 bit logit 按部署侧约定接入解扰、速率恢复、LDPC 和 CRC 链路。推理脚本通过启动 Python 进程完成文件桥接，适合接口验证和离线评估；逐 slot 实时部署仍需常驻服务或模型导出。

## 2. 总体处理流程

### 2.1 系统级数据流

```text
MATLAB/仿真数据
  └─ HDF5: rx_grid, channel_est, data_mask, qm, code_rate, num_rbs, noise_var
       └─ MatlabH5Dataset：懒加载、轴转换、按 RB 裁剪
            └─ padded_collate：按 batch 最大 S/K 补零
                 └─ 输入归一化（数据 RE RMS）
                      └─ 复数 LMMSE 初始均衡 x_L
                           └─ 证据特征构造
                                └─ 状态初始化 ConvBlock
                                     └─ 6 轮层间消息聚合 + 门控卷积更新
                                          ├─ 符号头：Δx 实部、Δx 虚部、log 方差
                                          └─ bit 头：Qmax 个 coded-bit logits
                                               └─ data_mask/Qm 掩码
                                                    └─ loss、BER、EVM 或 MAT 输出
```

### 2.2 输入、输出及张量形状

| 张量 | 逻辑形状 | 含义 |
|---|---|---|
| `rx_grid` | `[B,S,K,R,2]` | 接收频域栅格，末维为实部/虚部 |
| `channel_est` | `[B,S,K,R,L,2]` | 有效信道 $H(r,l)$，末维为实部/虚部 |
| `data_mask` | `[B,S,K,L]` | 数据 RE 为 1，DM-RS、PTRS、保护子载波和补零为 0 |
| `noise_var` | `[B]` | 每个样本的复噪声功率 E[abs(n)²] |
| `qm` | `[B]` | 每个 QAM 符号的 bit 数，支持 2/4/6/8/10 |
| `code_rate` | `[B]` | 目标码率，范围通常为 `[0,1]` |
| `num_rbs` | `[B]` | 分配的 RB 数，决定有效 $K=12N_\mathrm{RB}$ |
| `equalized_symbols` | `[B,S,K,L,2]` | 神经均衡复数符号 |
| `bit_logits` | `[B,S,K,L,Qmax]` | coded-bit logits，正值表示 bit=1 |
| `log_symbol_var` | `[B,S,K,L]` | 符号不确定度的对数输出，当前用于诊断/导出 |
| `lmmse_symbols` | `[B,S,K,L,2]` | 未经神经修正的 LMMSE 基线 |

在数据集层面，MATLAB 轴序可以是 `axis_order="matlab"`，Python loader 会自动反转存储轴，统一返回样本优先的逻辑形状。每个样本先裁剪到 `12*num_rbs` 个子载波，`padded_collate` 再把一个 batch 补到该 batch 的最大尺寸；因此所有补零位置必须由 `data_mask=0` 标记。

默认网络的层级尺寸如下。所有卷积 stride=1，3×3 深度卷积 padding=1，保持 S/K 分辨率；两层共享同一套层内算子，同轮参数共享，轮与轮之间参数独立。

| 阶段 | 输入通道 | 输出通道 | 结构 |
|---|---:|---:|---|
| 条件编码 | 3 | 16 | 两层线性层及 SiLU |
| 状态初始化 | 28 | 64 | ConvBlock |
| 单轮消息变换 | 64 | 64 | 1×1 Conv + SiLU |
| 单轮状态更新 | 64+64+28=156 | 64 | ConvBlock |
| 单轮门控 | 64 | 64 | 1×1 Conv + Sigmoid |
| 符号头 | 64 | 3 | ConvBlock(64,64) + 1×1 Conv |
| bit 头 | 64 | 10 | ConvBlock(64,64) + 1×1 Conv |

ConvBlock 内的两次深度可分离卷积均为“3×3 逐通道卷积 + 1×1 通道混合”，每次之后接 GroupNorm 和 SiLU。默认 64 输出通道对应 8 个归一化组。

### 2.3 运行模式

- **训练：** `train.py` 创建数据划分、加载 batch、前向计算、反向传播和检查点；输出 `best.pt`、`last.pt`、训练曲线和历史指标。
- **验证/测试：** `validate.py` 复用保存的划分，在总体及 `(MCS,RB,SNR)` 分组上计算 loss、coded BER 和 EVM。
- **离线推理：** `infer.py` 在无标签 HDF5 上输出 `equalized_symbols`、`bit_llr_positive_one`、LMMSE 对照和 mask 到 MAT 文件。
- **星座导出：** `export_constellation.py` 选取有效数据 RE，导出参考、神经、传统三组复数点和 EVM。

## 3. 分模块处理流程

### 3.1 数据组织、校验与流式读取

训练 HDF5 至少包含 `rx_grid`、`channel_est`、`data_mask`、`tx_symbols`、`coded_bits` 及 `mcs_index`、`qm`、`code_rate`、`num_rbs`、`noise_var`。可选 `snr_db` 用于分层和报告，`traditional_eq` 用于真实传统接收机对照。

`MatlabH5Dataset` 在初始化时检查每个 shard 的 rank、天线/层维度、RB 上限、元数据长度和 `Qm` 上限；以全局偏移表把多个文件映射成一个 map-style dataset。每个 worker 自己懒开 HDF5 句柄，避免把全量数组读入内存。读取时依据 `num_rbs` 裁剪子载波，并把小于 `Qmax` 的 `coded_bits` 在末维补零。

这里的 bit 标签必须是逐 RE、逐层、按 QAM mapper 顺序排列的 rate-matched/scrambled bits，而不是传输块原始 bit。该对齐关系决定 bit BCE 与最终软译码是否有效。

### 3.2 Batch 补零与有效位置管理

`padded_collate` 在 $S$ 和 $K$ 方向取 batch 内最大值，生成零填充张量；元数据按 batch 堆叠。模型中所有对数据的监督和有效输出均通过 `data_mask` 控制：

$$
M_{s,k,l}=1 \quad\text{表示该层该 RE 是 PDSCH 数据位置，否则为 0。}
$$

模型内部使用联合 RE mask

$$
M_{s,k}=\bigvee_{l=1}^{L}M_{s,k,l}
$$

来计算归一化和屏蔽 LMMSE 输出；符号修正量和 bit 输出再乘逐层 data_mask，损失只统计有效位置。无效位置不直接贡献监督损失，但卷积和 GroupNorm 仍可能通过这些位置影响有效 RE 的特征与梯度。GroupNorm 会统计空间维，同一样本放入不同补零尺寸的 batch 时不保证输出完全相同。若两层 mask 不一致，被屏蔽层的均衡符号仍可能保留联合 RE mask 下的 LMMSE 值，调用端须按逐层 mask 提取数据。

### 3.3 接收信号归一化

对每个样本，根据有效 RE 计算接收信号尺度。代码先求

$$
p_{s,k}=\frac{1}{R}\sum_{r=1}^{R}\left[\left(\Re y_{s,k,r}\right)^2+\left(\Im y_{s,k,r}\right)^2\right],
$$

再在 $M_{s,k}=1$ 的位置取平均并开方：

$$
\rho=\sqrt{\max\left(10^{-8},\frac{\sum_{s,k}M_{s,k}p_{s,k}}
 {\max(1,\sum_{s,k}M_{s,k})}\right)}.
$$

接收栅格和信道估计除以尺度 ρ，噪声功率除以 ρ²：

$$
\tilde y=y/\rho,\qquad \tilde H=H/\rho,\qquad
\tilde N_0=N_0/\rho^2.
$$

这一步让网络看到相对稳定的动态范围，同时保持信号与噪声的物理比例不变。噪声在进入对数特征前被限制到不小于 $10^{-8}$，对数值再裁剪到 `[-20,10]`。

### 3.4 复数 LMMSE 初始均衡

对每个 $(b,s,k)$，将接收向量和信道矩阵写为

$$
\mathbf y\in\mathbb C^{R},\qquad
\mathbf H\in\mathbb C^{R\times L},\qquad
\mathbf y=\mathbf H\mathbf x+\mathbf n .
$$

假设各层发送符号具有单位平均功率，噪声空间协方差为标量方差乘单位阵。下式中 y、H 表示归一化后的接收向量和信道矩阵；当前代码用归一化后的噪声方差求解

$$
\hat{\mathbf x}_{\mathrm LMMSE}=
(\mathbf H^H\mathbf H+\tilde N_0\mathbf I_L)^{-1}
\mathbf H^H\mathbf y .
$$

实现将 `[实部,虚部]` 转为 PyTorch complex，在展平后的 RE 批次上计算 Gram 矩阵和右端项，再用 `torch.linalg.solve` 求解，避免显式求逆。求解使用 float32 复数，即使外层启用 AMP 也不会使用不兼容的复数半精度线性代数。无效 RE 的结果乘以联合 mask 后置零，随后再转回实数二维表示 `[实部,虚部]`。

LMMSE 是两个作用的共同输入：一方面作为神经输出的物理基线，另一方面作为每层证据特征，使网络主要学习 LMMSE 未建模的残差。

### 3.5 条件向量编码

对每个样本构造三维条件向量：

$$
\mathbf c_0=\left[\frac{Q_m}{Q_{\max}},\; r,\;
\frac{N_\mathrm{RB}}{N_{\mathrm{RB,max}}}\right],
$$

其中 $r$ 为 `code_rate`，默认 $Q_\max=10$、$N_\mathrm{RB,max}=275$。三维向量经过 `Linear(3,condition_channels)`、SiLU、`Linear(condition_channels,condition_channels)`、SiLU 得到条件嵌入 $\mathbf c$。该向量复制到所有层、所有 RE，使共享卷积核可以根据调制阶数、码率和带宽改变判决策略。

### 3.6 证据特征拼接与状态初始化

每个层节点 $l$ 的证据张量为

$$
\mathbf e_l=\operatorname{concat}
 (\mathbf y_{RI},\;\mathbf h_{l,RI},\;\hat{\mathbf x}_{L,l,RI},\;
 M_l,\;\log\tilde N_0,\;\mathbf c).
$$

默认 2×2 配置下通道数为

$$
2R+2R+2+1+1+16=28.
$$

具体排列与代码一致：

- `rx_features`：2 根接收天线的实部和虚部，共 4 通道，并复制到每个层节点；
- `h_features`：当前层对应的信道估计实部/虚部，共 4 通道；
- `x_features`：该层 LMMSE 符号的实部/虚部，共 2 通道；
- `mask_features`：该层数据 RE mask，1 通道；
- `log_n0`：归一化噪声功率对数，1 通道；
- `conditions`：MCS/RB/码率条件嵌入，16 通道。

证据从 `[B,L,C_e,S,K]` 展平为 `[B·L,C_e,S,K]`，输入两层“深度可分离卷积 + GroupNorm + SiLU”的 `state_init`，得到隐藏状态

$$
\mathbf z^{(0)}\in\mathbb R^{B\times L\times C_h\times S\times K},
$$

默认 $C_h=64$。深度卷积在每个通道提取局部时频邻域，逐点卷积完成通道混合；GroupNorm 对不同 batch 尺寸和动态范围更稳定。

### 3.7 迭代层间图消息传递

网络把两个空间层视为图节点，把同一 RE 上另一层的隐藏状态视为干扰消除消息。第 $t$ 轮首先对每层状态使用共享的 `1×1 Conv + SiLU`：

$$
\mathbf m_l'=\phi(W_m*\mathbf z_l^{(t)}).
$$

然后聚合除自身之外的节点：

$$
\mathbf m_l^{(t)}=
\frac{1}{L-1}\sum_{j\neq l}\mathbf m_j'.
$$

对于当前固定的 $L=2$，这等价于直接使用另一层的消息；代码保留了对任意 $L>1$ 的平均形式。若 $L=1$，消息置零。

将自身状态、层间消息和原始证据沿通道拼接：

$$
\mathbf u_l^{(t)}=\operatorname{concat}
(\mathbf z_l^{(t)},\mathbf m_l^{(t)},\mathbf e_l).
$$

`update` 使用两次深度可分离卷积、GroupNorm、SiLU 产生增量 $\Delta_l^{(t)}$，随后用 `1×1 Conv + Sigmoid` 生成逐通道门控 $g_l^{(t)}$：

$$
\mathbf z_l^{(t+1)}=\mathbf z_l^{(t)}+
g_l^{(t)}\odot\Delta_l^{(t)}.
$$

默认执行 6 轮。残差更新让每轮只需学习增量，门控以 0 到 1 的学习系数调节状态增量；所有迭代共享输入证据但使用各自独立的 `ReceiverIteration` 参数。

### 3.8 符号修正头

最终状态展平为 `[B·L,C_h,S,K]`，经过一个 ConvBlock 和 `1×1 Conv` 输出 3 个通道：

1. 符号实部修正 $\Delta x_\mathrm{re}$；
2. 符号虚部修正 $\Delta x_\mathrm{im}$；
3. 对数符号方差 $\log\hat\sigma_x^2$。

神经均衡符号采用残差形式：

$$
\hat x_l=\hat x_{L,l}+M_l\cdot
(\Delta x_{\mathrm{re},l}+j\Delta x_{\mathrm{im},l}).
$$

log_symbol_var 被裁剪到 [-10,6] 后输出，但当前 ReceiverLoss 不使用该输出。该通道没有方差监督，不能作为已校准的置信度或噪声估计；它是预留接口，实际使用前需要增加训练目标和校准验证。

### 3.9 Coded-bit 软输出头

bit 头同样由 ConvBlock 和 `1×1 Conv` 组成，输出 $Q_{max}$ 个通道并转置为 `[B,S,K,L,Qmax]`：

$$
\ell_{s,k,l,q}=f_q(\mathbf z_l^{(T)}).
$$

输出先乘逐层 `data_mask`。损失计算时再用

$$
M^\mathrm{bit}_{s,k,l,q}=M_{s,k,l}\land(q<Q_m)
$$

屏蔽超过当前调制阶数的 bit 平面和所有非数据 RE。因此 QPSK 只训练前 2 个平面，1024QAM 训练前 10 个平面。logit 经 sigmoid 后可视为 bit=1 的概率；下游若约定正 LLR 表示 bit=0，应在接口处取负。

### 3.10 联合损失

bit 分支使用逐元素 BCE with logits：

$$
\mathcal L_\mathrm{bit}=\frac{1}{\sum M^\mathrm{bit}}
\sum M^\mathrm{bit}\,
\operatorname{BCEWithLogits}(\ell,b).
$$

符号分支使用有效 RE 上的归一化复数误差：

$$
\mathcal L_\mathrm{sym}=
\frac{\sum_{s,k,l}M_{s,k,l}
\left\|\hat x_{s,k,l}-x_{s,k,l}\right\|_2^2}
{\max\left(\epsilon,\sum_{s,k,l}M_{s,k,l}
\left\|x_{s,k,l}\right\|_2^2\right)}.
$$

总损失为

$$
\mathcal L=\lambda_b\mathcal L_\mathrm{bit}
+\lambda_s\mathcal L_\mathrm{sym},
$$

默认 `bit_weight=1.0`、`symbol_weight=0.2`。这种设计使 bit 软信息作为主任务，星座精度作为辅助约束，同时适配不同 QAM 阶数和不同带宽。

### 3.11 训练流程与检查点

1. 读取配置并设置 Python、NumPy、PyTorch 随机种子；
2. 扫描 HDF5 文件，建立数据集和确定性分层划分；
3. 训练集使用 shuffle，验证集不打乱；CUDA 时启用 pin memory 和可选 AMP；
4. 前向计算联合损失；AMP 时先缩放损失再反向传播，随后还原梯度、裁剪梯度范数，最后执行 AdamW 更新和 scaler 更新；
5. 每个 epoch 后用验证损失驱动 `ReduceLROnPlateau`；验证损失改善时保存 `best.pt`，每轮保存 `last.pt`；
6. 连续达到 `early_stop_patience` 个 epoch 未改善则早停，同时保存 `history.csv` 和 `loss.png`。

默认配置为 50 个 epoch、batch size 16、学习率 0.0005、权重衰减 0.0001、梯度裁剪范数 1.0、早停耐心 10。

### 3.12 分层划分与评估

基本分组键为 `(mcs_index, num_rbs)`。当 `snr_db` 是离散扫描值且每个 SNR 子组足以填充所有非空 split 时，进一步按 SNR 分组，从而让每个 SNR 都进入训练、验证和测试；连续随机 SNR 或样本过少时退回基本分组。默认比例是 80%/10%/10%，划分文件保存数据集长度和 shard 文件指纹，数据发生变化时拒绝复用旧 split。

BER/EVM 累计原始错误数或功率后计算比值。loss 的日志统计有所不同：训练循环按 batch 样本数加权平均 batch loss，独立验证脚本逐样本求 loss 再平均；混合 RB/Qm 时两者不一定相同。

- **coded BER：** `logit >= 0` 判为 1，与 `coded_bits` 比较，只统计有效 RE 和前 $Q_m$ 个 bit；
- **神经 EVM：** $\sqrt{\sum|\hat x-x|^2/\sum|x|^2}$；
- **传统 EVM：** 若 HDF5 有 `traditional_eq`，使用真实 MATLAB 传统接收机结果；否则使用相同 `channel_est/noise_var` 计算的 LMMSE；
- 输出还可以按 MCS、RB、SNR 分组，定位某个调制、带宽或信噪比区间的退化。

### 3.13 MATLAB 推理接口

MATLAB 侧完成 OFDM 解调、DM-RS 信道估计和噪声方差估计后，按 HDF5 合同写入 `rx_grid`、`channel_est`、`data_mask` 和调度元数据。`infer.py` 加载检查点并输出：

- `equalized_symbols`：神经均衡复数符号；
- `traditional_lmmse_symbols`：LMMSE 基线；
- `bit_llr_positive_one`：正值表示 bit=1 的 logits；
- `log_symbol_variance`：对数符号方差；
- `data_mask`、`mcs_index`、`qm`、`num_rbs`：用于后处理筛选和对齐。

MATLAB 后处理只应选取 `data_mask=1` 且 bit 索引 `1:qm` 的位置，再按下游译码器要求转换 LLR 符号。若用于正式 BLER，应继续执行 PDSCH 解扰、速率恢复、LDPC 和 CRC。

## 4. 复杂度、内存和数值稳定性设计

### 4.1 主要计算量

- LMMSE 对每个 RE 求解 $L\times L$ 线性系统；当前 $L=2$，矩阵求解代价很小，主要随 $B\cdot S\cdot K$ 线性增长。
- 卷积部分是全卷积，主要代价随 $B\cdot L\cdot S\cdot K\cdot C_h$ 线性增长；深度可分离卷积比普通 $3\times3$ 卷积减少通道间空间卷积开销。
- 6 轮迭代需要保留中间状态用于反向传播；增大 RB、batch 或迭代数时显存近似线性增加。

### 4.2 稳定性措施

- LMMSE 噪声方差 `clamp_min(1e-8)`，避免正则项退化；
- 噪声对数特征裁剪到 `[-20,10]`，符号方差头裁剪到 `[-10,6]`；
- 复数线性求解强制使用 float32，规避 CUDA 复数半精度实现差异；
- loss 的分母使用最小值保护；BER/EVM 的计数和功率使用 float64 累加，避免 AMP 下大规模 bit 计数溢出；
- `data_mask` 贯穿归一化、LMMSE、输出和 loss，降低 padding/导频标签污染风险。

## 5. 部署前验证和验收标准

1. **数据对齐验收：** 验证 `tx_symbols` 与相同 QAM 映射、归一化和加扰设置下的 `coded_bits` 逐 RE 一致；确认 `channel_est` 来自部署同款估计器，而不是完美 CSI。
2. **掩码验收：** DM-RS、PTRS、保护载波、未分配子载波和 batch padding 的 `data_mask` 必须为 0；有效 PDSCH RE 必须为 1。
3. **数值验收：** 检查 `noise_var` 与 `rx_grid` 处于同一 AGC 标度，检查不同 RB 样本在 batch 补零后输出形状和有效位置正确。
4. **基线验收：** 在相同数据、相同有效 RE 上比较神经 EVM 与传统 `traditional_eq` 或 LMMSE EVM；同时按 SNR、MCS、RB 检查 coded BER 曲线。
5. **链路验收：** 将输出 logits 接入既有 MATLAB PDSCH 解扰、速率恢复、LDPC、CRC，最终以 BLER/吞吐率而不是 coded BER 单独判定部署效果。
6. **覆盖验收：** 已保存的 dataset_report.json 记录了 80 个样本、单一 MCS/RB，并给出 WARN；这是历史审计结果，不能代表新目录 dataset_h5_new 的覆盖范围。正式结论须对实际使用的数据重新审计，覆盖目标 MCS、RB、SNR、信道和多普勒范围。现有划分不会自动识别同一信道轨迹或重复采集，数据生成时须保证独立性。

## 6. 代码模块对应关系

| 算法模块 | 主要实现文件 |
|---|---|
| HDF5 读取、轴转换、裁剪、补零 | `neural_rx_pytorch/data.py` |
| LMMSE、条件编码、迭代图卷积、输出头 | `neural_rx_pytorch/model.py` |
| 联合 bit/符号损失 | `neural_rx_pytorch/losses.py` |
| 训练/验证 epoch、设备迁移、AMP | `neural_rx_pytorch/engine.py` |
| BER/EVM 累计 | `neural_rx_pytorch/metrics.py` |
| 分层划分和数据指纹 | `neural_rx_pytorch/splits.py` |
| 训练和 checkpoint | `train.py`、`neural_rx_pytorch/checkpoint.py` |
| 验证与分组报告 | `validate.py` |
| 离线推理和 MATLAB 输出 | `infer.py`、`matlab/writeNeuralRxH5.m` |
| 星座点及 EVM 导出 | `export_constellation.py` |
