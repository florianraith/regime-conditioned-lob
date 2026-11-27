"""Per raw row order flow: OFI and trade sign.

Both compare a valid row with the previous valid row, so invalid (empty book) rows are skipped. The first valid
row of a day and every invalid row get OFI 0 and trade sign 0.
"""

import numpy as np

from lobgen.common.clean import RawDay


def ofi_levels(raw: RawDay) -> np.ndarray:
    """Cont, Kukanov, Stoikov OFI per row and level, shape (rows, LEVELS).

    e(k) = 1[b_n >= b_n-1] vb_n - 1[b_n <= b_n-1] vb_n-1 - 1[a_n <= a_n-1] va_n + 1[a_n >= a_n-1] va_n-1
    """
    out = np.zeros(raw.bid_px.shape, dtype=np.int64)
    rows = np.flatnonzero(raw.valid)
    if len(rows) < 2:
        return out
    cur, prev = rows[1:], rows[:-1]
    b, b0 = raw.bid_px[cur], raw.bid_px[prev]
    a, a0 = raw.ask_px[cur], raw.ask_px[prev]
    vb, vb0 = raw.bid_vol[cur], raw.bid_vol[prev]
    va, va0 = raw.ask_vol[cur], raw.ask_vol[prev]
    out[cur] = (b >= b0) * vb - (b <= b0) * vb0 - (a <= a0) * va + (a >= a0) * va0
    return out


def trade_sign(raw: RawDay) -> np.ndarray:
    """+1 buy, -1 sell, 0 unsigned, per row. Rows without trade_volume get 0.

    Buy if the ask side was consumed (ask_1 rose, or unchanged with less volume). Sell if the bid side was consumed
    (bid_1 fell, or unchanged with less volume). Both or neither: unsigned.
    """
    sign = np.zeros(len(raw), dtype=np.int64)
    rows = np.flatnonzero(raw.valid)
    if len(rows) < 2:
        return sign
    cur, prev = rows[1:], rows[:-1]
    a, a0 = raw.ask_px[cur, 0], raw.ask_px[prev, 0]
    b, b0 = raw.bid_px[cur, 0], raw.bid_px[prev, 0]
    buy = (a > a0) | ((a == a0) & (raw.ask_vol[cur, 0] < raw.ask_vol[prev, 0]))
    sell = (b < b0) | ((b == b0) & (raw.bid_vol[cur, 0] < raw.bid_vol[prev, 0]))
    sign[cur] = buy.astype(np.int64) - sell.astype(np.int64)
    sign[raw.trade_volume <= 0] = 0
    return sign


def flow_vector(n_upd, n_trd, signed_vol, ofi_1, ofi_10) -> np.ndarray:
    """F_t (5 numbers per step): log1p(n_upd), log1p(n_trd), asinh(signed_vol), asinh(ofi_1), asinh(ofi_10).

    Inputs are per step sums from a snapshot file at any resolution. Returns float32 of shape (steps, 5).
    """
    cols = [
        np.log1p(n_upd), np.log1p(n_trd),
        np.arcsinh(signed_vol), np.arcsinh(ofi_1), np.arcsinh(ofi_10),
    ]
    return np.stack([np.asarray(c, dtype=np.float64) for c in cols], axis=1).astype(np.float32)
