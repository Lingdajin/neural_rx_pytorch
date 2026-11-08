# PyTorch 2x2 MIMO PDSCH Neural Receiver

[中文使用说明](README_zh.md)

This folder is independent of the original TensorFlow/Sionna PUSCH runtime. It
contains a fully convolutional PyTorch receiver trained from MATLAB link-level
captures. One checkpoint supports variable MCS and RB allocations by using
`Qm`, code rate, and RB count as conditioning inputs and masking padded REs.

The model uses received IQ, noise variance, data masks and scheduling conditions
without channel estimates or LMMSE. Fixed one-hot layer identities distinguish
the two streams in the shared iterative graph blocks. It directly predicts:

- neural equalized constellation symbols;
- coded-bit logits for up to 1024-QAM (`Qm <= 10`).

Legacy checkpoints are incompatible: train from scratch in a new directory.
Legacy HDF5 files remain usable; training never reads `channel_est` or
`traditional_eq`. See [the architecture and training analysis](NO_CSI_ANALYSIS_zh.md).
Keep received pilot REs in `rx_grid`; the data mask limits supervision and outputs.

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

## Train

Run commands from `neural_rx_pytorch/`:

```bash
python train.py --data dataset_h5 --output-dir runs/no_csi --device cuda
```

Override common settings without editing JSON:

```bash
python train.py --data dataset_h5 --output-dir runs/no_csi \
  --epochs 80 --batch-size 24 --num-workers 8
```

Outputs include `best.pt`, `last.pt`, `history.csv`, `loss.png`, `config.json`,
and `splits.npz`. Without CLI overrides, training reads `dataset_h5` and writes
`runs/no_csi`. Only checkpoints from the current architecture can resume. Resume with:

```bash
python train.py --data dataset_h5 --output-dir runs/no_csi \
  --resume runs/no_csi/last.pt
```

Do not change or reorder the dataset shards when resuming or evaluating: the
saved split file contains a dataset fingerprint and rejects a mismatch.

## Validate and test

```bash
python validate.py --data dataset_h5 \
  --checkpoint runs/no_csi/best.pt \
  --splits runs/no_csi/splits.npz \
  --config runs/no_csi/config.json \
  --split validation --output-dir results/validation

python validate.py --data dataset_h5 \
  --checkpoint runs/no_csi/best.pt \
  --splits runs/no_csi/splits.npz \
  --config runs/no_csi/config.json \
  --split test --output-dir results/test
```

`summary.json` contains aggregate loss, coded-bit BER, neural EVM, and
traditional EVM when stored `traditional_eq` is available. `metrics_by_mcs_rb.csv` gives MCS/RB/SNR breakdowns when the
optional `snr_db` metadata is available.
Post-LDPC BLER should be measured by passing the logits through the MATLAB
decoder; coded BER alone is not a substitute for BLER.

## Export constellation points

```bash
python export_constellation.py --data dataset_h5 \
  --checkpoint runs/no_csi/best.pt \
  --splits runs/no_csi/splits.npz \
  --config runs/no_csi/config.json \
  --split test --output results/constellation.mat
```

The `.mat` and `.npz` outputs contain reference, neural, and traditional
complex points plus EVM values and MCS/RB/layer metadata. If the source file
has `traditional_eq`, it is used only as an offline comparison; otherwise the
traditional points and EVM are omitted. No internal LMMSE is computed.

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
3. traditional and neural receivers using identical received samples (only
   the traditional receiver consumes channel estimates);
4. unseen SNR, Doppler, delay profiles, and RF impairments;
5. runtime and memory at the maximum RB allocation.
