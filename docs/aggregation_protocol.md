# UMass-to-33-Bus aggregation protocol

This document is the operational companion to paper §4.2 and Fig. 2. The
goal of the protocol is to deterministically map the 114 apartment-level
meters of the UMass Smart\* Apartment 2016 release onto the 32 load buses
of the IEEE `case33bw` radial distribution test feeder so that the
resulting multi-bus dataset is bit-exact reproducible.

## Inputs

* `{y_j(t)}_{j=1}^{114}` — apartment-level active power, 1-minute
  resolution, 2016-01-01 to 2016-12-15, from `data/raw/umass_smartstar/2016/`.
* `case33bw` — IEEE 33-bus radial feeder, via MATPOWER.

## Five-step protocol

### Step 1 — temporal resampling

`y_j(t) → ỹ_j(t)` by mean aggregation, 1-min → 15-min. Total `33,578`
time steps per apartment.

### Step 2 — per-apartment feature vector `φ_j ∈ ℝ⁵`

| Coord. | Symbol | Definition |
|---|---|---|
| (a) | `ȳ_j`                        | annual mean load |
| (b) | `max_t ỹ_j(t) / ȳ_j`         | peak-to-mean ratio |
| (c) | `Σ_{t∈T_eve} ỹ_j(t) / Σ_t ỹ_j(t)` | evening share (T_eve = 18:00 – 23:00 local) |
| (d) | `r_we/wd`                    | weekend-to-weekday mean ratio |
| (e) | `Var(Δ ỹ_j)`                 | ramp-rate variance |

### Step 3 — standardisation and k-means

* Standardise each coordinate to mean 0, variance 1 (z-score).
* Apply `k-means` with `k = 32` working clusters plus a slack cluster that
  collects apartments with `> 10 %` missing values.
* The `rng(42)` setting and the MATLAB `kmeans` default `Replicates = 1`
  combine to make the partition deterministic.

### Step 4 — rank-match cluster ↔ bus

* Rank the 32 working clusters by annual mean load.
* Rank the 32 load buses of `case33bw` by tabulated `Pd`.
* Match by rank (1:1 isotonic mapping). Bus 1 (slack) is excluded; it does
  not receive aggregated load.

### Step 5 — scaled bus-level aggregation

For each load bus `i` and each time step `t`,

```
P_{i,t} = γ_i · Σ_{j ∈ c(i)} ỹ_j(t),
```

where `c(i)` is the apartment set assigned to bus `i`, and `γ_i` is the
scaling factor that aligns the cluster's annual mean with the bus's
tabulated `P_d`.

## Implementation

A single file implements the entire protocol:
[`src/preprocess_aggregate.m`](../src/preprocess_aggregate.m). It writes
`data/processed/umass_33bus_15min.mat` and reports a SHA-256 hash so the
output can be verified against a published reference.

```matlab
>> addpath('src'); preprocess_aggregate
```

## Determinism guarantee

Given:

* the exact UMass Smart\* Apartment 2016 release,
* MATLAB R2024a with Statistics and Machine Learning Toolbox,
* `rng(42)` seeded before the call,

the output `umass_33bus_15min.mat` is bit-exact reproducible by any
reader. We strongly recommend verifying the SHA-256 before drawing any
quantitative comparisons against the numbers in the paper.

## Reference cross-checks

* The 32 cluster sizes, the bus-rank assignment, and the per-cluster `γ_i`
  values are written to `results/tables/aggregation_summary.csv` for
  inspection.
* The cluster-level aggregate annual load should be within ± 1 % of the
  `case33bw` tabulated total `Σ Pd`.

## Notes on the protocol's role in the paper

The k-means-derived adjacency used by the TA-Transformer arises from the
*clustering output of Step 3*, not from any electrical measurement; this
is why paper §6.3 attributes the modest topology-aware advantage to the
lack of genuine electrical information in the adjacency matrix. Future
work (paper §6.4) will replace this predefined adjacency with a learnable
one.
