# MATLAB Dataset Format

Use HDF5 (`.h5` or MATLAB `-v7.3`) so training can read samples without loading
the complete dataset into memory. `matlab/writeNeuralRxH5.m` writes the format
below. Large captures should be split into multiple shard files; all shards are
treated as one dataset.

## Tensor convention

All shapes below are **logical MATLAB shapes**. The writer adds the file
attribute `axis_order="matlab"`; the Python loader reverses the HDF5 storage
axes automatically.

| Dataset | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `rx_grid` | `[N,S,K,2,2]` | `single` | Received grid; last axes are RX antenna and `[real,imag]` |
| `channel_est` | `[N,S,K,2,2,2]` | `single` | Effective channel `H(rx,layer)` and `[real,imag]` |
| `data_mask` | `[N,S,K,2]` | `uint8` | One on PDSCH data REs for each layer; zero on DM-RS/PTRS/padding |
| `tx_symbols` | `[N,S,K,2,2]` | `single` | Transmitted layer-mapped QAM symbols and `[real,imag]` |
| `coded_bits` | `[N,S,K,2,Qmax]` | `single` | Rate-matched/scrambled bits in mapper order, padded with zeros |
| `traditional_eq` | `[N,S,K,2,2]` | `single` | Optional traditional receiver symbols and `[real,imag]` |
| `mcs_index` | `[N]` | integer | 3GPP MCS index (metadata/metric grouping) |
| `qm` | `[N]` | integer | Bits per QAM symbol: 2, 4, 6, 8, or 10 |
| `code_rate` | `[N]` | `single` | Target code rate in `[0,1]`, not the integer scaled by 1024 |
| `num_rbs` | `[N]` | integer | Allocated RB count; active `K = 12*num_rbs` |
| `noise_var` | `[N]` | `single` | Complex noise variance `E[abs(n)^2]` in the same scale as `rx_grid` |
| `snr_db` | `[N]` | `single` | Optional SNR label used for stratification and validation reports |

Here `N` is the sample count, `S` is normally 14, `K` is the maximum allocated
subcarrier count in that shard, and `Qmax <= 10`. A shard may mix MCS and RB
sizes. Shorter allocations are zero-padded along `K`; `data_mask` must be zero
on the padding. The loader crops each sample to `12*num_rbs`, then the collate
function pads only to the largest item in that minibatch.

## Label alignment

`coded_bits(n,s,k,layer,1:qm)` must be exactly the bit vector passed to the QAM
mapper for `tx_symbols(n,s,k,layer)`, in the same order. Do not use transport
block bits as labels. Keep scrambling enabled in both labels and the deployed
PDSCH receive chain. The model output uses the convention **positive LLR/logit
means bit 1**; negate it if the downstream MATLAB decoder expects positive for
bit 0.

`channel_est` should come from the same DM-RS estimator and interpolation used
at deployment. Training on perfect channel values and deploying with LS/MMSE
estimates creates a substantial distribution mismatch. `traditional_eq`, when
present, should contain the actual conventional receiver output on the same
data RE ordering. Otherwise the scripts compare against an internal LMMSE
equalizer based on `channel_est` and `noise_var`.

## Collection recommendations

- Generate independent channel/noise realizations before splitting data.
- Cover every intended `(MCS, RB, SNR, channel profile, Doppler)` combination.
- Store roughly balanced sample counts for MCS/RB combinations, or use separate
  shards so the training corpus can be rebalanced easily.
- Do not include DM-RS, PTRS, guard carriers, or unused REs in `data_mask`.
- Verify `tx_symbols == qammod(coded_bits, ...)` under your exact MATLAB mapping
  and normalization before a long training run.
- Use approximately 80/10/10 train/validation/test data. `train.py` creates and
saves a deterministic split stratified by `(mcs_index,num_rbs)`. When every
discrete SNR subgroup has enough repeated samples, `snr_db` is added to the
stratification key so every SNR occurs in every split. Continuously sampled SNR
falls back to `(mcs_index,num_rbs)` automatically.

The current loss is supervised at the equalized-symbol and coded-bit levels.
CRC/BLER requires feeding exported logits into the existing MATLAB PDSCH
descrambling, rate recovery, LDPC decoding, and CRC chain.
