"""Paths, excluded days, splits and tick sizes for the 6 month sample (2025-01-02 to 2025-06-30).

Source: implementation guides 04 (DiffLOB) and 05 (LOB GAN), sections "Load and clean" and "Splits".
All splits use whole days in date order and are shared by DiffLOB and the GAN.
"""

import os
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE lines from the repo's .env. Comments (whole line or after " #") are skipped."""
    if not path.is_file():
        return {}
    lines = (line.split(" #", 1)[0].strip() for line in path.read_text().splitlines())
    pairs = (line.split("=", 1) for line in lines if "=" in line and not line.startswith("#"))
    return {k.strip(): v.strip().strip("\"'") for k, v in pairs}


_ENV_FILE = _read_env_file(REPO_ROOT / ".env")


def _path_var(name: str) -> Path:
    """Path from the shell environment, else from .env. Relative paths are relative to the repo root."""
    value = os.environ.get(name) or _ENV_FILE.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and set it.")
    return (REPO_ROOT / Path(value).expanduser()).resolve()


# Raw input: {fesx,vstoxx}/YYYYMMDD.parquet.
DATA_DIR = _path_var("LOBGEN_DATA_DIR")
# Derived snapshots, trade tables and runs.
OUT_DIR = _path_var("LOBGEN_OUT_DIR")

INSTRUMENTS = ("fesx", "vstoxx")
TICK = {"fesx": 1.0, "vstoxx": 0.05}
LEVELS = 10

# FESX roll and expiry days, dropped entirely (125 -> 121 usable days).
EXCLUDED_DAYS = frozenset({date(2025, 3, 20), date(2025, 3, 21), date(2025, 6, 19), date(2025, 6, 20)})

# VSTOXX monthly expiry: feed stops at 12:00 CET. Forward fill the last mid and set vx_valid = 0 afterwards.
VSTOXX_EXPIRY_DAYS = frozenset({
    date(2025, 1, 22), date(2025, 2, 19), date(2025, 3, 18),
    date(2025, 4, 16), date(2025, 5, 21), date(2025, 6, 18),
})

# Recording window is fixed in UTC. Keep it all for look back; samples only in the sample window.
RECORD_START_UTC = time(7, 0)
RECORD_END_UTC = time(15, 35)
SAMPLE_START_UTC = time(7, 30)
SAMPLE_END_UTC = time(15, 0)
DST_START = date(2025, 3, 31)  # dst flag = 1 from this day

# Grid: 0.5 s base, 1 s and 2 s derived by subsampling (never resampled separately).
BASE_STEP_S = 0.5
RESOLUTIONS = {"0p5s": 1, "1s": 2, "2s": 4}  # name -> number of base steps
BLOCK_S = 64  # history and future length in seconds
STRIDE_S = 8


def raw_path(instrument: str, day: date) -> Path:
    return DATA_DIR / instrument / f"{day:%Y%m%d}.parquet"


def available_days(instrument: str = "fesx") -> list[date]:
    """Trading days with a raw file, sorted, excluded days removed."""
    days = (date(int(p.stem[:4]), int(p.stem[4:6]), int(p.stem[6:])) for p in (DATA_DIR / instrument).glob("*.parquet"))
    return sorted(d for d in days if d not in EXCLUDED_DAYS)


@dataclass(frozen=True)
class Span:
    """Inclusive date range."""

    start: date
    end: date

    def days(self, universe: list[date]) -> list[date]:
        return [d for d in universe if self.start <= d <= self.end]


def _d(md: str) -> date:
    m, d = md.split("-")
    return date(2025, int(m), int(d))


def _span(a: str, b: str) -> Span:
    return Span(_d(a), _d(b))


# Split A: fixed split for all headline numbers. Train = Train A + Train B (79 days).
SPLIT_A = {
    "train_a": _span("01-02", "03-28"),
    "stress": _span("03-31", "04-17"),  # tariff shock, never trained on
    "embargo": _span("04-22", "04-25"),  # unused
    "train_b": _span("04-28", "05-23"),
    "val": _span("05-26", "06-06"),
    "test": _span("06-09", "06-30"),
}

# Split B: monthly expanding walk forward. Validation = last 5 days before the test month and is not trained on.
# The first trading day of each test month is embargoed, so test starts on the second one.
SPLIT_B = [
    {"train": _span("01-02", "02-21"), "val": _span("02-24", "02-28"), "test": _span("03-04", "03-31")},
    {"train": _span("01-02", "03-24"), "val": _span("03-25", "03-31"), "test": _span("04-02", "04-30")},
    {"train": _span("01-02", "04-23"), "val": _span("04-24", "04-30"), "test": _span("05-05", "05-30")},
    {"train": _span("01-02", "05-23"), "val": _span("05-26", "05-30"), "test": _span("06-03", "06-30")},
]

# Split C: DiffLOB paper replication (1 s, history 32 / future 32), the paper's 16 / 1 / 2 (+1) layout.
SPLIT_C = {
    "train": _span("01-02", "01-23"),
    "val": _span("01-24", "01-24"),
    "test": _span("01-27", "01-28"),
    "usefulness": _span("01-29", "01-29"),
}


def split_a_days(universe: list[date]) -> dict[str, list[date]]:
    out = {k: s.days(universe) for k, s in SPLIT_A.items()}
    out["train"] = out["train_a"] + out["train_b"]
    return out
