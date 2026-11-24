# Regime-Conditioned Generative Modelling of Limit Order Books

KIT BWL Praktikum WS26. Generative models for the Eurex FESX order book, conditioned on market regimes, trained on a 6 month sample (2025-01-02 to 2025-06-30, FESX with VSTOXX as context).

Both models use the same 0.5 s / 1 s / 2 s grid, 64 s blocks and splits, so results go into one table.

## Layout

* `src/lobgen/common/`: shared pipeline used by both models (config, snapshot grid, flows, transforms, metrics).
* `src/lobgen/lobgan/`: LOB GAN (Cont et al. 2023), Florian.
* `src/lobgen/difflob/`: DiffLOB (Wang & Ventre, IJCAI 2026), Arda.
* `configs/`: YAML hyperparameters per training run.
* `tests/`: checks that the code matches the implementation guides.

## Setup

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The raw data is not in git. Point the code at your local copy:

```sh
export LOBGEN_DATA_DIR=/path/to/sample_6months_2025-01_to_06   # contains fesx/ and vstoxx/ with YYYYMMDD.parquet
export LOBGEN_OUT_DIR=/path/to/outputs                         # snapshots, trade tables, runs
```

## Data rules

* Whole days in date order, never shuffle across time.
* Every transform, cap and z-score is fitted on the training days of its split only.
* Context features use the past only. Regimes describe the future block and are conditions, never inputs from the target.
* Splits are defined once in `src/lobgen/common/config_sample6m.py`: A (fixed, headline numbers), B (monthly expanding walk forward), C (DiffLOB paper replication).

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
