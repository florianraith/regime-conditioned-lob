from datetime import date

import numpy as np
import pandas as pd
import pytest

from lobgen.common import config_sample6m as cfg
from lobgen.common.clean import RawDay, grid_ns, load_day
from lobgen.common.flows import ofi_levels, trade_sign
from lobgen.common.grid import FLOW_COLS, fesx_grid, snapshot_index, step_of_row, subsample

L = cfg.LEVELS
DAY = date(2025, 1, 15)


def make_raw(ts, bid1, bid_vol1, ask1, ask_vol1, trade_vol=None, valid=None) -> RawDay:
    """Synthetic book: level k price = level 1 -/+ (k - 1), volume 10 on deeper levels."""
    n = len(ts)
    steps = np.arange(L)
    bid_px = np.asarray(bid1)[:, None] - steps
    ask_px = np.asarray(ask1)[:, None] + steps
    bid_vol = np.full((n, L), 10)
    ask_vol = np.full((n, L), 10)
    bid_vol[:, 0], ask_vol[:, 0] = bid_vol1, ask_vol1
    tv = np.zeros(n, dtype=np.int64) if trade_vol is None else np.asarray(trade_vol)
    return RawDay(
        instrument="fesx", day=DAY, ts=np.asarray(ts, dtype=np.int64),
        bid_px=bid_px, ask_px=ask_px, bid_vol=bid_vol, ask_vol=ask_vol,
        trade_count=(tv > 0).astype(np.int64), trade_volume=tv,
        valid=np.ones(n, bool) if valid is None else np.asarray(valid), levels=L, n_dropped_ts=0,
    )


def test_snapshot_strictly_before_and_skips_invalid():
    ts = np.array([10, 20, 30, 40])
    valid = np.array([True, True, False, True])
    grid = np.array([5, 10, 11, 20, 35, 41])
    assert snapshot_index(ts, valid, grid).tolist() == [-1, -1, 0, 0, 1, 3]


def test_step_of_row():
    grid = np.array([10, 20, 30])
    ts = np.array([-1, 0, 9, 10, 19, 29, 30])
    assert step_of_row(ts, grid).tolist() == [-1, 0, 0, 1, 1, 2, 3]


def test_ofi_level1():
    # bid up with volume 5: +5. Ask unchanged, volume 20 -> 25: -25 + 20 = -5.
    raw = make_raw([0, 1], bid1=[100, 101], bid_vol1=[7, 5], ask1=[103, 103], ask_vol1=[20, 25])
    e = ofi_levels(raw)
    assert e[0].tolist() == [0] * L
    assert e[1, 0] == 5 - 5


def test_ofi_skips_invalid_rows():
    raw = make_raw([0, 1, 2], bid1=[100, 0, 100], bid_vol1=[7, 0, 9], ask1=[103, 0, 103], ask_vol1=[20, 0, 20],
                   valid=[True, False, True])
    e = ofi_levels(raw)
    assert e[1].tolist() == [0] * L
    assert e[2, 0] == 9 - 7


def test_trade_sign():
    raw = make_raw(
        [0, 1, 2, 3, 4],
        bid1=[100, 100, 100, 99, 99],
        bid_vol1=[10, 10, 10, 10, 10],
        ask1=[101, 101, 102, 102, 102],
        ask_vol1=[10, 6, 10, 10, 10],
        trade_vol=[0, 4, 6, 3, 2],
    )
    # row 1 ask volume fell: buy; row 2 ask rose: buy; row 3 bid fell: sell; row 4 nothing consumed: unsigned
    assert trade_sign(raw).tolist() == [0, 1, 1, -1, 0]


def test_subsample_sums_and_last():
    df = pd.DataFrame({"t_ns": np.arange(8), "x": np.arange(8), "n_upd": np.ones(8, dtype=np.int32)})
    s = subsample(df, 4, ["n_upd"], ["t_ns", "x"])
    assert s["t_ns"].tolist() == [3, 7]
    assert s["x"].tolist() == [3, 7]
    assert s["n_upd"].tolist() == [4, 4]


def test_grid_shape():
    g = grid_ns(DAY)
    assert len(g) == 61_800
    assert len(g) % 4 == 0
    assert (g[1::2] % 1_000_000_000 == 0).all()


days = cfg.available_days()


@pytest.mark.skipif(DAY not in days, reason="raw data not found")
def test_real_day_grid():
    raw = load_day("fesx", DAY)
    snap = fesx_grid(raw)
    assert len(snap) == 61_800
    assert snap["book_valid"].all()
    s0, s1 = (pd.Timestamp(f"{DAY} {t}", tz="UTC").value for t in (cfg.RECORD_START_UTC, cfg.RECORD_END_UTC))
    in_window = (raw.ts >= s0) & (raw.ts < s1)
    assert snap["n_upd"].sum() == in_window.sum()
    assert snap["trd_vol"].sum() == raw.trade_volume[in_window].sum()
    two = subsample(snap, 4, FLOW_COLS, [c for c in snap.columns if c not in FLOW_COLS])
    assert two["n_upd"].sum() == snap["n_upd"].sum()
    pd.testing.assert_frame_equal(
        two.drop(columns=FLOW_COLS), snap.iloc[3::4].drop(columns=FLOW_COLS).reset_index(drop=True)
    )
