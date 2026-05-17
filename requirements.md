# Requirements

This repository accompanies the *Energies* paper
*"Multi-Level Short-Term Load Forecasting for Residential Energy Communities:
A Topology-Aware Transformer Benchmark."* It targets the exact environment
used for the experiments reported in §4.5 of the paper.

## Mandatory

| Item | Version used in the paper | Notes |
|---|---|---|
| MATLAB | **R2024a** | Earlier R2023b should also work; not tested below. |
| Deep Learning Toolbox | shipped with R2024a | Provides `multiHeadAttentionLayer`. |
| Statistics and Machine Learning Toolbox | shipped with R2024a | Provides `kmeans`, `arima`, `lillietest`. |
| Parallel Computing Toolbox | shipped with R2024a | Enables `gpuArray`. |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB VRAM | Any CUDA-capable GPU with ≥ 6 GB should work; CPU fallback exists but is ≈ 30× slower. |
| Disk | ≈ 1 GB for raw data + ≈ 500 MB for cached `.mat` results |

## Optional

| Item | Purpose |
|---|---|
| MATPOWER ≥ 7.1 | Provides the `case33bw.m` test feeder. The case file is loaded by `src/load_topology.m`. |
| Git LFS | Only if you wish to clone the result-tensor archive in-place; otherwise download the GitHub Release zip. |

## Quick sanity check

```matlab
>> addpath('src'); check_gpu
```

This prints the detected GPU and toolbox versions and exits cleanly only when
every mandatory dependency is satisfied.
