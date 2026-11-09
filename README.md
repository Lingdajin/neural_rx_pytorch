# PyTorch 2x2 MIMO PDSCH Neural Receiver

[中文使用说明](README_zh.md)

This folder is independent of the original TensorFlow/Sionna PUSCH runtime. It
contains a fully convolutional PyTorch receiver trained from MATLAB link-level
captures. One checkpoint supports variable MCS and RB allocations by using
`Qm`, code rate, and RB count as conditioning inputs and masking padded REs.

The model consumes the raw received grid plus the known layer-specific DM-RS
grid/mask. A dilated pilot encoder spreads sparse pilot evidence, then iterative
convolutional graph blocks exchange messages between the two spatial layers.
No traditional channel estimate or LMMSE result is used in the forward path.
It predicts:

- neural equalized constellation symbols;
- coded-bit logits for up to 1024-QAM (`Qm <= 10`);
- an internal effective-channel estimate for auxiliary supervision/diagnostics.

This is a structural CGNN migration (state initialization, inter-layer graph
aggregation, iterative convolutional updates, and readout heads), adapted to
supervised PDSCH capture data. TensorFlow checkpoints from the original PUSCH
model are not weight-compatible; the PyTorch model must be trained anew.

It deliberately does not reproduce MATLAB's PDSCH channel estimation, rate
recovery, LDPC decoder, or CRC. Those standards-sensitive stages remain in the
existing MATLAB receiver.

## Installation

Python 3.10 is recommended. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. Install the PyTorch wheel
matching the installed CUDA version when GPU training is required.

## Prepare MATLAB data

Read [DATA_FORMAT.md](DATA_FORMAT.md), then call:

```matlab
writeNeuralRxH5('capture_0001.h5', data);
```

The directory may contain many `.h5` shards. The complete capture does not
need to fit in RAM.

Required inference inputs are `rx_grid`, `dmrs_grid`, `dmrs_mask`, `data_mask`,
`noise_var`, `qm`, `code_rate`, and `num_rbs`. `channel_est` and
`traditional_eq` are optional training/evaluation-only fields.

## Train

Run commands from `neural_rx_pytorch/`:

```bash
python train.py --data data/*.h5 --output-dir runs/pdsch_2x2 --device cuda
```

Override common settings without editing JSON:

```bash
python train.py --data data --output-dir runs/pdsch_2x2 \
  --epochs 80 --batch-size 24 --num-workers 8
```

Outputs include `best.pt`, `last.pt`, `history.csv`, `loss.png`, `config.json`,
`splits.npz`, and the auditable per-scenario `split_report.json`. Resume with:

```bash
python train.py --data data --output-dir runs/pdsch_2x2 \
  --resume runs/pdsch_2x2/last.pt
```

Do not change or reorder the dataset shards when resuming or evaluating: the
saved split file contains a dataset fingerprint and rejects a mismatch. New
runs stratify by MCS, Qm, RB, and discrete SNR. Legacy assisted-model split and
checkpoint files are rejected explicitly.

## Validate and test

```bash
python validate.py --data data \
  --checkpoint runs/pdsch_2x2/best.pt \
  --splits runs/pdsch_2x2/splits.npz \
  --config runs/pdsch_2x2/config.json \
  --split validation --output-dir results/validation

python validate.py --data data \
  --checkpoint runs/pdsch_2x2/best.pt \
  --splits runs/pdsch_2x2/splits.npz \
  --config runs/pdsch_2x2/config.json \
  --split test --output-dir results/test
```

`summary.json` contains aggregate loss, coded-bit BER, exact bit error/count,
zero-error 95% BER upper bound, neural EVM, and traditional EVM.
`metrics_by_scenario.csv` gives MCS/Qm/RB/SNR breakdowns when the optional
`snr_db` metadata is available.
Post-LDPC BLER should be measured by passing the logits through the MATLAB
decoder; coded BER alone is not a substitute for BLER.

## Export constellation points

```bash
python export_constellation.py --data data \
  --checkpoint runs/pdsch_2x2/best.pt \
  --splits runs/pdsch_2x2/splits.npz \
  --config runs/pdsch_2x2/config.json \
  --split test --output results/constellation.mat
```

The `.mat` and `.npz` outputs contain reference, neural, and traditional
complex points plus EVM values and MCS/RB/layer metadata. If the source file
has `traditional_eq`, it is used; otherwise a diagnostic LMMSE result can be
computed only when optional `channel_est` is present. Neither baseline enters
the neural forward path.

## Call from MATLAB

See [matlab/exampleCallNeuralRx.m](matlab/exampleCallNeuralRx.m). It writes one
inference HDF5 batch, invokes `infer.py`, and loads equalized symbols and bit
logits back into MATLAB. For high-throughput integration, keep a Python service
alive or export the model after validating this file-based reference path; a
new Python process per slot is intended only as a clear integration example.

## Practical acceptance criteria

Before deployment, evaluate held-out channels for every supported MCS and RB:

1. coded-bit BCE/BER and pre-decoder EVM;
2. post-LDPC BER/BLER and CRC in MATLAB;
3. traditional and neural receivers evaluated on the same received grids and
   transmitted references (only the traditional branch uses its estimator);
4. unseen SNR, Doppler, delay profiles, and RF impairments;
5. runtime and memory at the maximum RB allocation.
