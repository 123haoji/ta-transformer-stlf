# E-bike charging synthesis

This document is the operational companion to paper §4.3 and Equation (12).
It describes how the per-bus aggregate e-bike charging power
`p^{chg}_{i,t}` is generated for **Dataset C** (the complementary
semi-synthetic 33-bus residential community used in the e-bike sensitivity
analysis of §5.4).

> **The synthesis model is *not* a contribution of the paper.** It is a
> domain-realistic exogenous covariate calibrated against published
> residential electric-mobility statistics; the contribution lies in
> empirically reporting that the covariate adds < 1 percentage point of
> MAPE across penetration scenarios at the cluster aggregation level.

## Generative model

```
p^{chg}_{i,t} = n^{eb}_i · P̄^{single} · π(hour(t)) · ξ_{i,t} · μ^{SF}(t)
```

| Symbol | Meaning | Source |
|---|---|---|
| `n^{eb}_i`        | registered e-bike count assigned to bus `i`        | sampled per-bus from a Poisson( λ = µ × per-bus capita ) |
| `P̄^{single}`      | mean per-bike charging power = **0.25 kW**         | residential electric-mobility literature, §4.3 |
| `π(·)`            | empirical hour-of-day charging probability profile  | two-peak profile (morning departure + evening return) |
| `ξ_{i,t}`         | multiplicative noise, `logN(0, 0.2)`                | reflects per-event variability |
| `μ^{SF}(t)`       | calendar coefficient ∈ [0.45, 1.0]                  | dampens demand during the major migration holiday window |

The penetration multiplier `µ ∈ {0.5, 1.0, 2.0, 4.0}` is the sweep
parameter for §5.4; it scales `n^{eb}_i` proportionally so that the
expected per-bus bike count grows by 2× / 4× / 8× across the sweep.

## Implementation

Single entry point: [`src/synthesize_ebike.m`](../src/synthesize_ebike.m).

```matlab
>> addpath('src');
>> opts.mu = 1.0;           % penetration multiplier
>> opts.seed = 42;
>> p_chg = synthesize_ebike(opts);   % returns N×T matrix in kW
```

The function honours `rng(opts.seed)` so that every penetration sweep is
deterministic.

## Caveats and intended use

* The synthesis is **calibrated against published statistical summaries**
  (mean per-bike kW, hour-of-day charging propensity) rather than against
  direct measurements; absolute magnitudes should therefore be read as
  indicative.
* Dataset C is used **only** for the e-bike sensitivity analysis; it is
  not used as a primary measure of forecasting accuracy (paper §4.1).
* The negative finding in §5.4 — that the e-bike covariate provides
  < 1 pp MAPE benefit at the 33-bus aggregation — does not preclude
  benefit at finer aggregation levels (individual households or building
  meters) or under more extreme penetration scenarios.

## Reference cross-checks

* The realised peak-hour probability `π(hour)` should integrate to 1.0
  over a 24-h cycle.
* For `µ = 1.0`, the community-wide mean charging power should fall
  within ± 5 % of the value reported in the paper supplementary table.
