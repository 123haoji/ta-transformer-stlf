# TA-Transformer for Multi-Level Short-Term Load Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MATLAB R2024a](https://img.shields.io/badge/MATLAB-R2024a-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![DOI](https://img.shields.io/badge/DOI-pending-blue.svg)](#citation)

Reference implementation for the *Energies* article
**"Multi-Level Short-Term Load Forecasting for Residential Energy Communities:
A Topology-Aware Transformer Benchmark"** (Lu et al., 2026).

We benchmark five forecasting models (ARIMA, LSTM, CNN-LSTM, vanilla
Transformer, and the proposed **Topology-Aware Transformer**) plus a
temporal-free **PureGAT** ablation on 114 apartment-level meters from the
publicly available UMass Smart\* dataset, deterministically mapped onto the
IEEE `case33bw` radial feeder. The model achieves cluster-level
**WAPE 32.60 ± 0.07 %** and aggregate-level **WAPE 11.23 ± 0.08 %**, a
**2.9× reduction** through bottom-up aggregation.

---

## Repository contents

```
.
├── main.m                          Quick debug entry (≈ 5 min smoke test)
├── run_paper.m                     Production entry (3-seed, 3–6 h)
├── src/                            Model + data loaders + algorithms
│   ├── preprocess_aggregate.m      UMass-to-33-bus protocol (paper §4.2)
│   ├── synthesize_ebike.m          E-bike charging synthesis (paper §4.3)
│   ├── topology_aware_attention.m  TA-Attention core operator
│   ├── train_ta_transformer.m
│   ├── train_transformer.m
│   ├── train_lstm.m
│   ├── train_arima.m
│   ├── build_features.m
│   ├── evaluate_metrics.m
│   ├── load_gefcom.m
│   ├── load_umass_apartment.m
│   ├── load_topology.m
│   ├── plot_results.m
│   ├── predict_external.m
│   ├── auto_gpu_config.m
│   └── check_gpu.m
├── experiments/                    Stage 1 – 4 runners
│   ├── exp1_gefcom_baseline.m              §5.1 GEFCom2014 single-zone
│   ├── exp2_umass_main.m                   §5.2 UMass-33 cluster
│   ├── exp2_aggregation_sensitivity.m      §5.6 K-sweep
│   ├── exp2b_aggregate.m                   §5.3 system-aggregate
│   ├── exp2b_horizons.m                    §5.3.1 multi-horizon
│   ├── exp3_ebike_sensitivity.m            §5.4 e-bike penetration
│   └── exp4_zeroshot_443.m                 (exploratory; not in paper)
├── analysis/                       Post-processing
│   ├── stats_significance.m                Wilcoxon + Lilliefors
│   ├── holiday_sensitivity.m               §5.5 Thanksgiving robustness
│   ├── complexity_table.m                  parameters + latency
│   ├── reeval_exp2_wape.m
│   ├── replot_horizons.m
│   ├── replot_predictions.m
│   ├── make_aux_figures.m
│   ├── diagnose_ta_h1.m
│   └── diagnose_ta_h1_v2.m
├── scripts/                        Single-purpose helpers
│   ├── exp2_k33_only.m
│   ├── exp2_k_partial.m
│   ├── sanity_check_tier1.m
│   └── regenerate_all_tables_figures.m     One-click reproduction
├── data/                           Where to put external datasets (see data/README.md)
├── docs/
│   ├── reproducibility.md
│   ├── aggregation_protocol.md
│   ├── ebike_synthesis.md
│   └── experiments_data.md
├── results/
│   ├── figures/                    Eleven publication PDFs
│   ├── tables/                     Stage-level CSV outputs
│   └── tables_paper/               Seed-aggregated CSV outputs
├── README.md
├── LICENSE                         MIT
├── CITATION.cff
├── requirements.md
├── .gitignore
└── .gitattributes
```

---

## Requirements

| | Version used in the paper |
|---|---|
| MATLAB | R2024a |
| Deep Learning Toolbox | yes |
| Statistics and Machine Learning Toolbox | yes |
| Parallel Computing Toolbox | yes |
| GPU | NVIDIA GeForce RTX 4070 Laptop, 8 GB VRAM |
| MATPOWER (optional) | provides `case33bw.m` |

See [`requirements.md`](requirements.md) for full details.

---

## Data

Three external datasets are referenced; **none are redistributed in this
repository**. Place them under `data/` as described in
[`data/README.md`](data/README.md).

| Dataset | Source | License |
|---|---|---|
| UMass Smart\* Apartment 2016 | <https://traces.cs.umass.edu/docs/traces/smartstar/> | CC BY 4.0 |
| GEFCom2014 (load track) | Hong et al. (2016) | competition terms |
| Open-Meteo ERA5-Land reanalysis | <https://open-meteo.com/> | Public domain |
| IEEE `case33bw` | MATPOWER package | 3-clause BSD |

---

## Quick start (5-minute smoke test)

```matlab
>> main('Stage', 2, 'Quick', true)
```

This runs Stage 2 (UMass-33 cluster forecast) at a small scale to verify the
environment. Output CSV / figures land under `results/`.

---

## Reproducing the paper (3 – 6 hours)

```matlab
>> run_paper                                  % all four stages, 3 seeds
>> run scripts/regenerate_all_tables_figures  % aggregate → CSV + PDF
```

This regenerates every numeric table and every figure that appears in the
paper. The seed protocol (paper §4.5) is fixed at `{42, 1, 2}` for all
UMass-33, UMass-1, and e-bike-sensitivity experiments; single-seed
configurations (GEFCom2014, holiday-week, K-sweep) use seed `42`.

For a detailed step-by-step walkthrough, see
[`docs/reproducibility.md`](docs/reproducibility.md).

---

## Key results

| Aggregation level | Best non-graph baseline | TA-Transformer | Reduction vs. cluster |
|---|---|---|---|
| Cluster (33 buses)   | CNN-LSTM WAPE 31.70 ± 0.05 % | WAPE 32.60 ± 0.07 % | — |
| System (1 aggregate) | CNN-LSTM WAPE 10.09 ± 0.05 % | WAPE 11.23 ± 0.08 % | **2.9×** |

Three-way ablation (cluster MAPE penalty when removed):

| Component | MAPE cost | Wilcoxon $p$ |
|---|---|---|
| Temporal Transformer backbone (PureGAT) | **6.10 pp** | $p < 10^{-47}$ |
| Topology-aware attention (vanilla Transformer) | 1.05 pp | $p = 1.98\times10^{-32}$ |
| E-bike covariate | < 1.0 pp | — |

The temporal backbone is the load-bearing component; the topology-aware
attention is a statistically detectable but operationally negligible
refinement on a $k$-means-derived adjacency.

---

## Citation

If you use this code or build on the benchmark, please cite both the paper
and the software (`CITATION.cff`):

```bibtex
@article{lu2026taformer,
  title   = {Multi-Level Short-Term Load Forecasting for Residential Energy
             Communities: A Topology-Aware Transformer Benchmark},
  author  = {Lu, Shan min},
  journal = {Energies},
  year    = {2026},
  volume  = {1},
  number  = {1},
  doi     = {10.3390/enXXXXXXXX}
}
```

---

## Acknowledgements

We thank the UMass Smart\* project for releasing the apartment-level
metering data, the GEFCom2014 organisers for the load-track benchmark,
Open-Meteo for the ERA5-Land reanalysis service, and the MATPOWER
maintainers for the IEEE test feeder library.

## License

Code: [MIT](LICENSE). Third-party content retains its original licence.
