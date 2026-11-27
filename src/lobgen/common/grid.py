"""One 0.5 s grid per day, 1 s and 2 s derived from it.

Grid g_k = 07:00:00 UTC + k * 0.5 s, k = 1..61,800 (last point 15:35:00). The snapshot at g_k is the last valid
book strictly before g_k (searchsorted(ts, g_k, "left") - 1, then back to the last valid row). Flows of step k sum
the raw rows in [g_k-1, g_k). 1 s keeps every 2nd and 2 s every 4th snapshot, flows summed over the merged steps,
so the three grids agree exactly.

Outputs under cfg.OUT_DIR:
  snapshots/fesx_{0p5s,1s,2s}/YYYYMMDD.parquet        book in ticks (int32) + per step flows + book_valid
  snapshots/vstoxx_ctx_{0p5s,1s,2s}/YYYYMMDD.parquet  vx_mid, vx_age_s, vx_valid
  trades/YYYYMMDD.parquet                             FESX trade rows with the book just before
  reports/clean_report.csv                            missing data and gaps per day

Run: uv run python -m lobgen.common.grid [--days 20250115 ...] [--workers 4]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

from lobgen.common import config_sample6m as cfg
from lobgen.common.clean import (
    NS, RawDay, grid_ns, invalid_spans, last_valid_index, load_day, sample_window_ns, utc_ns, vstoxx_cutoff_ns,
)
from lobgen.common.flows import ofi_levels, trade_sign

L = cfg.LEVELS
BOOK_COLS = [f"{s}_{f}_{k}" for k in range(1, L + 1) for s in ("ask", "bid") for f in ("price", "vol")]
FLOW_COLS = ["n_upd", "n_trd", "trd_vol", "signed_vol", "ofi_1", "ofi_10"]


def snapshot_index(ts: np.ndarray, valid: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Row of the last valid book strictly before each grid time, -1 if there is none."""
    idx = np.searchsorted(ts, grid, "left") - 1
    lv = last_valid_index(valid)
    return np.where(idx >= 0, lv[np.maximum(idx, 0)], -1)


def step_of_row(ts: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Position in `grid` of the step [g_k-1, g_k) holding each row, -1 before g_0 and len(grid) from g_N on."""
    edges = np.r_[2 * grid[0] - grid[1], grid]
    return np.searchsorted(edges, ts, "right") - 1


def sum_per_step(values: np.ndarray, step: np.ndarray, n: int) -> np.ndarray:
    inside = (step >= 0) & (step < n)
    return np.bincount(step[inside], weights=values[inside], minlength=n)[:n].round().astype(np.int64)


def check_book(df: pd.DataFrame, where: str) -> None:
    """Assert ask_1 > bid_1, ask prices increasing, bid prices decreasing, volumes >= 1."""
    ap = df[[f"ask_price_{k}" for k in range(1, L + 1)]].to_numpy()
    bp = df[[f"bid_price_{k}" for k in range(1, L + 1)]].to_numpy()
    vol = df[[c for c in BOOK_COLS if "_vol_" in c]].to_numpy()
    assert (ap[:, 0] > bp[:, 0]).all(), f"{where}: ask_1 <= bid_1"
    assert (np.diff(ap, axis=1) > 0).all(), f"{where}: ask prices not increasing"
    assert (np.diff(bp, axis=1) < 0).all(), f"{where}: bid prices not decreasing"
    assert (vol >= 1).all(), f"{where}: volume < 1"


def fesx_grid(raw: RawDay) -> pd.DataFrame:
    """0.5 s snapshots with per step flows for one FESX day."""
    grid = grid_ns(raw.day)
    idx = snapshot_index(raw.ts, raw.valid, grid)
    if (idx < 0).any():
        raise ValueError(f"fesx {raw.day}: no valid book before {pd.Timestamp(grid[np.argmax(idx < 0)])}")
    step = step_of_row(raw.ts, grid)
    ofi = ofi_levels(raw)
    sign = trade_sign(raw)
    n = len(grid)

    out = {"t_ns": grid}
    for k in range(L):
        out[f"ask_price_{k + 1}"] = raw.ask_px[idx, k]
        out[f"ask_vol_{k + 1}"] = raw.ask_vol[idx, k]
        out[f"bid_price_{k + 1}"] = raw.bid_px[idx, k]
        out[f"bid_vol_{k + 1}"] = raw.bid_vol[idx, k]
    out["n_upd"] = sum_per_step(np.ones(len(raw)), step, n)
    out["n_trd"] = sum_per_step(raw.trade_count, step, n)
    out["trd_vol"] = sum_per_step(raw.trade_volume, step, n)
    out["signed_vol"] = sum_per_step(sign * raw.trade_volume, step, n)
    out["ofi_1"] = sum_per_step(ofi[:, 0], step, n)
    out["ofi_10"] = sum_per_step(ofi.sum(1), step, n)
    # 0 when the latest raw row before g_k is an empty or broken book and the snapshot is forward filled.
    last_row = np.searchsorted(raw.ts, grid, "left") - 1
    out["book_valid"] = (idx == last_row).astype(np.int8)

    df = pd.DataFrame(out)
    df[BOOK_COLS + FLOW_COLS] = df[BOOK_COLS + FLOW_COLS].astype(np.int32)
    check_book(df, f"fesx {raw.day} 0p5s")
    return df


def subsample(df: pd.DataFrame, factor: int, sum_cols: list[str], last_cols: list[str]) -> pd.DataFrame:
    """Every `factor`-th row of a 0.5 s frame; `sum_cols` summed over the merged steps, `last_cols` taken at the
    kept row. The first kept row is position factor - 1, so 1 s and 2 s points fall on whole seconds."""
    n = len(df) // factor * factor
    out = df[last_cols].iloc[factor - 1:n:factor].reset_index(drop=True)
    for c in sum_cols:
        out[c] = df[c].to_numpy()[:n].reshape(-1, factor).sum(1).astype(df[c].dtype)
    return out[[c for c in df.columns if c in out.columns]]


def vstoxx_ctx(raw: RawDay) -> pd.DataFrame:
    """VSTOXX mid (index points), age of the last valid quote and validity on the 0.5 s grid.

    The mid is forward filled through empty books and after the feed cutoff on expiry days, with vx_valid = 0 in
    both cases. vx_valid is also 0 before the first valid quote of the day (vx_mid NaN there).
    """
    grid = grid_ns(raw.day)
    idx = snapshot_index(raw.ts, raw.valid, grid)
    have = idx >= 0
    i = np.maximum(idx, 0)
    mid = (raw.ask_px[i, 0] + raw.bid_px[i, 0]) * (cfg.TICK["vstoxx"] / 2)
    last_row = np.searchsorted(raw.ts, grid, "left") - 1
    valid = have & (idx == last_row)
    cutoff = vstoxx_cutoff_ns(raw.day)
    if cutoff is not None:
        valid &= grid <= cutoff
    return pd.DataFrame({
        "t_ns": grid,
        "vx_mid": np.where(have, mid, np.nan),
        "vx_age_s": np.where(have, (grid - raw.ts[i]) / NS, np.nan).astype(np.float32),
        "vx_valid": valid.astype(np.int8),
    })


def trades_table(raw: RawDay) -> pd.DataFrame:
    """Every FESX row with trade_volume > 0 inside the recording window, with sign and the book just before it.

    "Just before" is the previous valid row. Prices are in ticks. Used by the GAN market impact study.
    """
    sign = trade_sign(raw)
    lv = last_valid_index(raw.valid)
    prev = np.r_[-1, lv[:-1]]
    start = utc_ns(raw.day, cfg.RECORD_START_UTC)
    rows = np.flatnonzero((raw.trade_volume > 0) & (raw.ts >= start) & (prev >= 0))
    p = prev[rows]
    out = {
        "t_ns": raw.ts[rows],
        "sign": sign[rows].astype(np.int8),
        "trade_count": raw.trade_count[rows].astype(np.int32),
        "trade_volume": raw.trade_volume[rows].astype(np.int32),
        "row_valid": raw.valid[rows].astype(np.int8),
        "bid_price_1": raw.bid_px[p, 0].astype(np.int32),
        "ask_price_1": raw.ask_px[p, 0].astype(np.int32),
        "spread": (raw.ask_px[p, 0] - raw.bid_px[p, 0]).astype(np.int32),
    }
    for k in range(5):
        out[f"ask_vol_{k + 1}"] = raw.ask_vol[p, k].astype(np.int32)
        out[f"bid_vol_{k + 1}"] = raw.bid_vol[p, k].astype(np.int32)
    return pd.DataFrame(out)


def day_report(raw: RawDay, vx: RawDay, snap: pd.DataFrame, ctx: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """Missing data and gaps for one day, measured on the sample window 07:30 to 15:00 UTC."""
    s0, s1 = sample_window_ns(raw.day)
    in_s = (snap["t_ns"] > s0) & (snap["t_ns"] <= s1)
    ts = raw.ts[(raw.ts >= s0) & (raw.ts < s1)]
    gaps = np.diff(np.r_[s0, ts, s1]) / NS
    n_inv, inv_s = invalid_spans(raw)
    vts = vx.ts[(vx.ts >= s0) & (vx.ts < s1)]
    traded = trades["trade_volume"]
    return {
        "day": raw.day.isoformat(),
        "fesx_rows": len(raw),
        "fesx_rows_dropped_ts": raw.n_dropped_ts,
        "fesx_invalid_rows": int((~raw.valid).sum()),
        "fesx_invalid_runs": n_inv,
        "fesx_invalid_s": round(inv_s, 3),
        "fesx_max_gap_s": round(float(gaps.max()), 3),
        "fesx_gaps_over_5s": int((gaps > 5).sum()),
        "steps_no_update_0p5s": round(float((snap.loc[in_s, "n_upd"] == 0).mean()), 4),
        "steps_book_invalid_0p5s": int((snap.loc[in_s, "book_valid"] == 0).sum()),
        "trade_rows": len(trades),
        "trade_vol_signed_share": round(float(traded[trades["sign"] != 0].sum() / max(traded.sum(), 1)), 4),
        "vx_rows": len(vx),
        "vx_invalid_rows": int((~vx.valid).sum()),
        "vx_max_gap_s": round(float(np.diff(np.r_[s0, vts, s1]).max() / NS), 1),
        "vx_invalid_steps_0p5s": int((ctx.loc[in_s, "vx_valid"] == 0).sum()),
    }


def build_day(day: date) -> dict:
    raw = load_day("fesx", day)
    vx = load_day("vstoxx", day)
    snap = fesx_grid(raw)
    ctx = vstoxx_ctx(vx)
    trades = trades_table(raw)

    name = f"{day:%Y%m%d}.parquet"
    book_last = ["t_ns", *BOOK_COLS, "book_valid"]
    for res, factor in cfg.RESOLUTIONS.items():
        s = snap if factor == 1 else subsample(snap, factor, FLOW_COLS, book_last)
        c = ctx if factor == 1 else subsample(ctx, factor, [], list(ctx.columns))
        if factor > 1:
            check_book(s, f"fesx {day} {res}")
        for kind, frame in (("fesx", s), ("vstoxx_ctx", c)):
            path = cfg.OUT_DIR / "snapshots" / f"{kind}_{res}" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
    path = cfg.OUT_DIR / "trades" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(path, index=False)
    return day_report(raw, vx, snap, ctx, trades)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", nargs="*", help="YYYYMMDD, default all usable days")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    days = cfg.available_days()
    if args.days:
        wanted = {date(int(d[:4]), int(d[4:6]), int(d[6:])) for d in args.days}
        days = [d for d in days if d in wanted]

    with ProcessPoolExecutor(args.workers) as pool:
        rows = []
        for r in pool.map(build_day, days):
            print(r["day"], f"rows {r['fesx_rows']:,}", f"invalid {r['fesx_invalid_rows']}",
                  f"max gap {r['fesx_max_gap_s']} s", f"signed {r['trade_vol_signed_share']:.1%}", flush=True)
            rows.append(r)
    report = pd.DataFrame(rows)
    path = cfg.OUT_DIR / "reports" / "clean_report.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.days and path.exists():
        old = pd.read_csv(path)
        report = pd.concat([old[~old["day"].isin(report["day"])], report]).sort_values("day")
    report.to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
