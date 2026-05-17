# `data/` — where external datasets go

This repository **does not redistribute** raw data. Please obtain the three
external datasets directly from their authoritative sources and place them
under this directory before running `run_paper` or `main`.

## Expected layout

```
data/
├── raw/
│   ├── umass_smartstar/         <-- UMass Smart* Apartment 2016 release
│   │   ├── 2016/
│   │   │   ├── Apt1.csv
│   │   │   ├── Apt2.csv
│   │   │   ├── ...
│   │   │   └── Apt114.csv
│   │   └── weather/             <-- co-located hourly weather
│   ├── gefcom2014/              <-- competition data archive
│   │   └── Load/
│   │       ├── L1-train.csv
│   │       └── L1-test.csv
│   └── era5_land/               <-- Open-Meteo download
│       └── era5_22.547N_113.976E_hourly.csv
├── processed/                   <-- auto-populated by preprocess_aggregate.m
│   ├── umass_33bus_15min.mat
│   └── synth_33bus_15min.mat
└── case33bw.m                   <-- copy from MATPOWER (or load_topology fetches it)
```

The `raw/`, `processed/`, and `cache/` sub-directories are listed in
`.gitignore` so that you can keep the data locally without it ever being
committed.

## How to obtain each dataset

### 1. UMass Smart\* Apartment 2016

* URL: <https://traces.cs.umass.edu/docs/traces/smartstar/>
* Citation: Barker, S.; Mishra, A.; Irwin, D.; Cecchet, E.; Shenoy, P.;
  Albrecht, J. *Smart\*: An Open Data Set and Tools for Enabling Research in
  Sustainable Homes.* SustKDD 2012.
* Licence: CC BY 4.0
* Download the **Apartment** release for the year **2016** and unpack the
  per-apartment CSV files under `data/raw/umass_smartstar/2016/`.

### 2. GEFCom2014 (load track)

* Citation: Hong, T.; Pinson, P.; Fan, S.; Zareipour, H.; Troccoli, A.;
  Hyndman, R.J. *Probabilistic energy forecasting: Global Energy
  Forecasting Competition 2014 and beyond.* *International Journal of
  Forecasting* 2016, 32, 896–913. DOI: 10.1016/j.ijforecast.2016.02.001
* Obtain the L1 (load) track from the competition organisers or from
  publicly indexed academic mirrors; we cannot redistribute it.

### 3. Open-Meteo ERA5-Land reanalysis

* URL: <https://open-meteo.com/>
* Licence: public domain (CC0).
* For the urban centroid used in Dataset C (22.547° N, 113.976° E), call
  the historical-weather endpoint with `models=era5_land` at hourly
  resolution covering 2016-01-01 to 2016-12-15.

### 4. IEEE case33bw (MATPOWER)

* Install MATPOWER (≥ 7.1) and add its directory to the MATLAB path; the
  loader `src/load_topology.m` calls `case33bw` directly.
* Alternatively, copy the single file `case33bw.m` from MATPOWER's `data/`
  directory into this `data/` folder.

## Reproducibility check

After populating the raw files, run:

```matlab
>> addpath('src'); preprocess_aggregate
```

This produces `data/processed/umass_33bus_15min.mat`. Given a fixed random
seed (default 42), the resulting tensor is **bit-exact reproducible** —
the SHA-256 reported by `preprocess_aggregate` should match the value
quoted in `docs/aggregation_protocol.md`.
