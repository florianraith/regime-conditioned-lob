# Regime-Conditioned Generative Modelling of Limit Order Books

Code for the KIT research project "Regime-Conditioned Generative Modelling of Limit Order Books".

## Layout

* `src/lobgen/common/`: shared pipeline used by both models (config, snapshot grid, flows, transforms, metrics).
* `src/lobgen/lobgan/`: LOB GAN (Cont et al. 2023), Florian.
* `src/lobgen/difflob/`: DiffLOB (Wang & Ventre, IJCAI 2026), Arda.
* `configs/`: YAML hyperparameters per training run.
* `tests/`: checks that the code matches the implementation guides.

## Setup

Requires [uv](https://docs.astral.sh/uv/). 

```sh
uv sync                  # installs Python 3.12 and the locked packages into .venv
uv run pytest
uv run python -m lobgen.<module>
```

The raw data is not in git. Copy `.env.example` to `.env` and point it at your local copy. Relative paths are relative to the repo root.

```sh
LOBGEN_DATA_DIR=data/sample_6months_2025-01_to_06   # contains fesx/ and vstoxx/ with YYYYMMDD.parquet
LOBGEN_OUT_DIR=data/out                             # snapshots, trade tables, runs
```

Add a dependency with `uv add <package>` (updates `pyproject.toml` and `uv.lock`). For Colab or a cluster without uv: `uv export --no-hashes > requirements.txt`, then `pip install -r requirements.txt`.

## Data rules

* Whole days in date order, never shuffle across time.
* Every transform, cap and z-score is fitted on the training days of its split only.
* Context features use the past only. Regimes describe the future block and are conditions, never inputs from the target.
* Splits are defined once in `src/lobgen/common/config_sample6m.py`: A (fixed, headline numbers), B (monthly expanding walk forward), C (DiffLOB paper replication).

## Snapshot grid

```sh
uv run python -m lobgen.common.grid --workers 6     # all 121 days, about 4 min
uv run python -m lobgen.common.grid --days 20250407 # selected days
```

Writes to `LOBGEN_OUT_DIR`:

* `snapshots/fesx_{0p5s,1s,2s}/YYYYMMDD.parquet`: grid time `t_ns`, 10 level book in ticks, per step `n_upd`, `n_trd`, `trd_vol`, `signed_vol`, `ofi_1`, `ofi_10`, and `book_valid` (0 where the book was empty and the last valid one is forward filled). Grid runs 07:00:00.5 to 15:35:00 UTC; select the sample window 07:30 to 15:00 yourself.
* `snapshots/vstoxx_ctx_{0p5s,1s,2s}/YYYYMMDD.parquet`: `vx_mid`, `vx_age_s`, `vx_valid`.
* `trades/YYYYMMDD.parquet`: every FESX trade row with sign, volume and the book just before it.
* `reports/clean_report.csv`: invalid rows, gaps and signed trade share per day.

`lobgen.common.flows.flow_vector` turns the flow columns into the GAN's 5 number flow vector.

## Shared config

`config_sample6m.py` holds paths, tick sizes, excluded days, time windows and the splits. Both models import it, so they see the same days.

```python
import pandas as pd
from lobgen.common import config_sample6m as cfg

days = cfg.available_days()          # 121 usable trading days, sorted
split = cfg.split_a_days(days)       # {"train": [...], "val": [...], "test": [...], "stress": [...], ...}

for day in split["train"]:
    book = pd.read_parquet(cfg.raw_path("fesx", day))
    ...

# Walk forward folds (split B)
for fold in cfg.SPLIT_B:
    train, val, test = (fold[k].days(days) for k in ("train", "val", "test"))
    ...

cfg.TICK["fesx"]                     # 1.0
cfg.VSTOXX_EXPIRY_DAYS               # VSTOXX feed stops at 12:00 CET on these days
```
