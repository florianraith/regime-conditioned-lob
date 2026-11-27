"""Load one raw day and clean it.

A raw file has one row per book change: timestamp_ns, {bid,ask}_{price,vol}_{1..10}, trade_count, trade_volume.
Timestamps are UTC and already in order. Some rows carry an empty or partial book (all prices NaN and volumes 0,
e.g. FESX 2025-04-07 14:11 to 14:13 UTC). Those rows are kept for counting updates and trades but are never used
as a book state: the grid forward fills the last valid book instead and flags the step.
"""

from dataclasses import dataclass
from datetime import date, datetime, time

import numpy as np
import pandas as pd

from lobgen.common import config_sample6m as cfg

NS = 1_000_000_000


@dataclass
class RawDay:
    """One instrument day in tick units. Book arrays have shape (rows, LEVELS), level 1 first."""

    instrument: str
    day: date
    ts: np.ndarray  # int64 ns UTC, non decreasing
    bid_px: np.ndarray  # int64 ticks, 0 where missing
    ask_px: np.ndarray
    bid_vol: np.ndarray  # int64
    ask_vol: np.ndarray
    trade_count: np.ndarray  # int64
    trade_volume: np.ndarray  # int64
    valid: np.ndarray  # bool, full valid book on `levels` levels
    levels: int
    n_dropped_ts: int  # rows dropped for timestamp_ns <= 0

    def __len__(self) -> int:
        return len(self.ts)


def utc_ns(day: date, t: time) -> int:
    """Nanoseconds since epoch of `day` at UTC wall time `t`."""
    return int(pd.Timestamp(datetime.combine(day, t), tz="UTC").value)


def vstoxx_cutoff_ns(day: date) -> int | None:
    """End of the VSTOXX feed on its monthly expiry days (12:00 CET or CEST), else None."""
    if day not in cfg.VSTOXX_EXPIRY_DAYS:
        return None
    hour_utc = 10 if day >= cfg.DST_START else 11
    return utc_ns(day, time(hour_utc))


def _book(df: pd.DataFrame, side: str, field: str) -> np.ndarray:
    return df[[f"{side}_{field}_{k}" for k in range(1, cfg.LEVELS + 1)]].to_numpy(dtype=np.float64)


def book_valid(bid_px, ask_px, bid_vol, ask_vol, levels: int) -> np.ndarray:
    """Rows with prices on `levels` levels, volumes >= 1, ask_1 > bid_1, asks ascending and bids descending."""
    bp, ap = bid_px[:, :levels], ask_px[:, :levels]
    ok = ~(np.isnan(bp).any(1) | np.isnan(ap).any(1))
    ok &= (bid_vol[:, :levels] >= 1).all(1) & (ask_vol[:, :levels] >= 1).all(1)
    with np.errstate(invalid="ignore"):
        ok &= ap[:, 0] > bp[:, 0]
        ok &= (np.diff(ap, axis=1) > 0).all(1) & (np.diff(bp, axis=1) < 0).all(1)
    return ok


def load_day(instrument: str, day: date, levels: int | None = None) -> RawDay:
    """Read, clean and convert one raw day.

    Drops rows with timestamp_ns <= 0 and rows at or after the end of the recording window (15:35 UTC). Rows before
    07:00 UTC stay so the first grid snapshot has a state. On VSTOXX expiry days rows after the feed cutoff are
    dropped. `levels` sets how many levels must be present for a row to count as valid: 10 for FESX, 1 for
    VSTOXX (only the mid is used).
    """
    if levels is None:
        levels = cfg.LEVELS if instrument == "fesx" else 1
    df = pd.read_parquet(cfg.raw_path(instrument, day))
    ts = df["timestamp_ns"].to_numpy(dtype=np.int64)
    end = utc_ns(day, cfg.RECORD_END_UTC)
    cutoff = vstoxx_cutoff_ns(day) if instrument == "vstoxx" else None
    if cutoff is not None:
        end = min(end, cutoff)
    keep = (ts > 0) & (ts < end)
    n_dropped_ts = int((ts <= 0).sum())
    df, ts = df[keep], ts[keep]
    if np.any(np.diff(ts) < 0):
        raise ValueError(f"{instrument} {day}: timestamps out of order")

    bid_px, ask_px = _book(df, "bid", "price"), _book(df, "ask", "price")
    bid_vol, ask_vol = _book(df, "bid", "vol"), _book(df, "ask", "vol")
    valid = book_valid(bid_px, ask_px, bid_vol, ask_vol, levels)

    tick = cfg.TICK[instrument]

    def to_ticks(px):
        t = np.nan_to_num(px / tick)
        r = np.rint(t)
        off = np.abs(t - r)[valid, :levels]
        if off.size and off.max() > 1e-6:
            raise ValueError(f"{instrument} {day}: price off the {tick} tick grid")
        return r.astype(np.int64)

    return RawDay(
        instrument=instrument,
        day=day,
        ts=ts,
        bid_px=to_ticks(bid_px),
        ask_px=to_ticks(ask_px),
        bid_vol=np.nan_to_num(bid_vol).astype(np.int64),
        ask_vol=np.nan_to_num(ask_vol).astype(np.int64),
        trade_count=df["trade_count"].to_numpy(dtype=np.int64),
        trade_volume=df["trade_volume"].to_numpy(dtype=np.int64),
        valid=valid,
        levels=levels,
        n_dropped_ts=n_dropped_ts,
    )


def last_valid_index(valid: np.ndarray) -> np.ndarray:
    """For each row the index of the latest valid row at or before it, -1 if none."""
    idx = np.where(valid, np.arange(len(valid)), -1)
    return np.maximum.accumulate(idx) if len(idx) else idx


def invalid_spans(raw: RawDay) -> tuple[int, float]:
    """Number of invalid runs and their total duration in seconds (until the next valid row)."""
    v = raw.valid
    if v.all() or not len(v):
        return 0, 0.0
    starts = np.flatnonzero(~v & np.r_[True, v[:-1]])
    ends = np.flatnonzero(~v & np.r_[v[1:], True])
    next_ts = np.r_[raw.ts[1:], raw.ts[-1]]
    return len(starts), float((next_ts[ends] - raw.ts[starts]).sum() / NS)


def sample_window_ns(day: date) -> tuple[int, int]:
    return utc_ns(day, cfg.SAMPLE_START_UTC), utc_ns(day, cfg.SAMPLE_END_UTC)


def grid_ns(day: date, step_s: float = cfg.BASE_STEP_S) -> np.ndarray:
    """Grid times g_1..g_N, g_k = 07:00:00 UTC + k * step, up to and including 15:35:00 UTC."""
    start = utc_ns(day, cfg.RECORD_START_UTC)
    end = utc_ns(day, cfg.RECORD_END_UTC)
    step = int(round(step_s * NS))
    return np.arange(start + step, end + 1, step, dtype=np.int64)

