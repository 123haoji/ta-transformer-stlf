# Reproducibility guide

This document walks through reproducing every numeric result and every
figure that appears in the paper. Each step references the corresponding
section of the manuscript so you can verify what is being recomputed.

## 0. Environment

Confirm the environment matches `requirements.md`:

```matlab
>> addpath('src'); check_gpu
```

You should see MATLAB R2024a, the three required toolboxes, and a
CUDA-capable GPU. If GPU acceleration is unavailable, training will fall
back to CPU and run ≈ 30× slower; the numeric results are unchanged.

## 1. Data

Populate `data/raw/` following `data/README.md`. Then run the deterministic
aggregation:

```matlab
>> addpath('src'); preprocess_aggregate
```

This emits `data/processed/umass_33bus_15min.mat`. The protocol is
deterministic given a fixed seed (default 42), so the SHA-256 of the
output should match the hash committed in `docs/aggregation_protocol.md`.

## 2. Seeds

The paper uses two seed protocols (§4.5):

* **Multi-seed** `{42, 1, 2}` for **UMass-33, UMass-1, and the e-bike
  sensitivity** experiments. Reported as mean ± standard deviation.
* **Single seed** `42` for the **GEFCom2014 single-zone sanity check, the
  K-sweep, the multi-horizon table, and the holiday-week table** (clearly
  flagged as single-seed in each table caption).

`run_paper` honours these defaults automatically.

## 3. Stage-by-stage reproduction

### Stage 1 — GEFCom2014 sanity check (Table 2 in the paper)

```matlab
>> run_paper('Stage', 1)
```

Output: `results/tables/exp1_metrics.csv`.

### Stage 2 — UMass-33 cluster forecast (Table 4 + Table 5 + Figure 3)

```matlab
>> run_paper('Stage', 2)
>> stats_significance('cluster')          % paired Wilcoxon, Lilliefors
```

Outputs:

* `results/tables_paper/exp2_metrics_aggregated.csv` (Table 4)
* `results/tables_paper/significance_cluster.csv`    (Table 5)
* `results/figures/fig_cluster_errorbar.pdf`
* `results/figures/fig_wilcoxon_heatmap.pdf`

### Stage 2b — System-aggregate forecast (Table 6 + Figure 4)

```matlab
>> run_paper('Stage', 5)                  % stage 2b enumerated as 5
```

Outputs:

* `results/tables_paper/exp2b_aggregate_metrics_aggregated.csv` (Table 6)
* `results/figures/fig_aggregate_predictions.pdf` (Figure 4)

### Stage 2b — Multi-horizon (Table 7 + Figure 5)

```matlab
>> run experiments/exp2b_horizons
>> run analysis/replot_horizons
```

### Stage 2c — Aggregation-granularity sweep (Table 9)

```matlab
>> run experiments/exp2_aggregation_sensitivity
```

Output: `results/tables/exp2_aggregation_sensitivity.csv`.

### Stage 3 — E-bike penetration (Table 8)

```matlab
>> run_paper('Stage', 3)
```

Output: `results/tables_paper/exp3_ebike_sensitivity_aggregated.csv`.

### Stage 4 — Holiday-week robustness (Table 9 in the paper, Thanksgiving)

```matlab
>> run analysis/holiday_sensitivity
```

Outputs:

* `results/tables_paper/holiday_sensitivity_cluster.csv`
* `results/tables_paper/holiday_sensitivity_aggregate.csv`

## 4. One-click figure & table regeneration

After all stages have completed:

```matlab
>> run scripts/regenerate_all_tables_figures
```

This script walks the `results/tables/*.mat` cache and re-emits every CSV
and PDF that appears in the paper, in less than five minutes.

## 5. Wall-clock budget

On the NVIDIA GeForce RTX 4070 Laptop GPU used for the paper:

| Stage | Wall-clock per seed | Total (3 seeds) |
|---|---|---|
| Stage 1 (GEFCom2014, 1 seed) | ≈ 4 min | 4 min |
| Stage 2 (UMass-33, 3 seeds)  | ≈ 6 min | ≈ 18 min |
| Stage 2b (aggregate, 3 seeds)| ≈ 5 min | ≈ 15 min |
| Stage 2b horizons (4 horizons × 4 models, 1 seed) | ≈ 25 min | ≈ 25 min |
| Stage 3 (e-bike, 4 × 2 configs × 3 seeds) | ≈ 60 min | ≈ 60 min |
| Holiday + complexity post-processing | ≈ 10 min | ≈ 10 min |
| **Total** | | **≈ 2 – 3 h** |

The "3 – 6 h" budget quoted in `README.md` adds a safety margin for
slower GPUs and for the K-sweep ablation.

## 6. Deterministic notes

* Per-seed standard deviations reflect mainly GPU-kernel and mini-batch
  non-determinism, not statistical sampling variability. See paper
  §6.4 (Methodology limitations).
* The borderline single-seed standard deviation at horizon $H=1$ for the
  TA-Transformer (0.86 pp MAPE) is documented in §5.3 of the paper and is
  attributed to the mean-pool decoder head.
