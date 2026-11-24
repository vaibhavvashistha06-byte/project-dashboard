"""Pure pandas/numpy stand-ins for the TA-Lib pieces we use, so the pipeline also runs
where TA-Lib is not installed. Numbers differ slightly from TA-Lib (warm-up handling),
and only 12 core candlestick patterns are implemented instead of all 61."""
import numpy as np
import pandas as pd


def sma(x, n): return pd.Series(x).rolling(n).mean().to_numpy()
def _wilder(s, n): return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def atr(h, l, c, n=14):
    h, l, c = map(pd.Series, (h, l, c))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return _wilder(tr, n).to_numpy()


def rsi(c, n=14):
    c = pd.Series(c); d = c.diff()
    up, dn = _wilder(d.clip(lower=0), n), _wilder((-d).clip(lower=0), n)
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).to_numpy()


def bbands(c, n=20, k=2):
    c = pd.Series(c); m, sd = c.rolling(n).mean(), c.rolling(n).std(ddof=0)
    return (m + k * sd).to_numpy(), m.to_numpy(), (m - k * sd).to_numpy()


def macd_hist(c, fast=12, slow=26, sig=9):
    c = pd.Series(c); m = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    return (m - m.ewm(span=sig, adjust=False).mean()).to_numpy()


def adx(h, l, c, n=14):
    h, l, c = map(pd.Series, (h, l, c))
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = pd.Series(atr(h, l, c, n))
    pdi = 100 * _wilder(pd.Series(pdm), n) / a; mdi = 100 * _wilder(pd.Series(mdm), n) / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return _wilder(dx, n).to_numpy()


def patterns(o, h, l, c):
    """12 core candlestick patterns, +1 bullish / -1 bearish / 0 none."""
    o, h, l, c = (pd.Series(x, dtype="float64") for x in (o, h, l, c))
    body, rng = c - o, (h - l).replace(0, np.nan)
    ab, up_w, lo_w = body.abs(), h - pd.concat([o, c], axis=1).max(axis=1), pd.concat([o, c], axis=1).min(axis=1) - l
    po, pc, ph = o.shift(), c.shift(), h.shift()
    pbody = body.shift(); pab = pbody.abs(); prng = rng.shift()
    red, green, pred, pgreen = body < 0, body > 0, pbody < 0, pbody > 0
    prior_dn, prior_up = c.shift() < c.shift(6), c.shift() > c.shift(6)
    out = {}
    out["CDLDOJI"] = (ab <= 0.1 * rng).astype(int)
    out["CDLHAMMER"] = ((lo_w >= 2 * ab) & (up_w <= 0.1 * rng) & (ab > 0) & prior_dn).astype(int)
    out["CDLSHOOTINGSTAR"] = -((up_w >= 2 * ab) & (lo_w <= 0.1 * rng) & (ab > 0) & prior_up).astype(int)
    out["CDLENGULFING"] = ((pred & green & (o <= pc) & (c >= po)).astype(int)
                           - (pgreen & red & (o >= pc) & (c <= po)).astype(int))
    inside = (np.maximum(o, c) <= np.maximum(po, pc)) & (np.minimum(o, c) >= np.minimum(po, pc)) & (pab > 0.6 * prng)
    out["CDLHARAMI"] = (pred & green & inside).astype(int) - (pgreen & red & inside).astype(int)
    c2red, c2green = (body.shift(2) < 0) & (ab.shift(2) > 0.6 * rng.shift(2)), (body.shift(2) > 0) & (ab.shift(2) > 0.6 * rng.shift(2))
    small1 = ab.shift() < 0.3 * rng.shift()
    mid2 = (o.shift(2) + c.shift(2)) / 2
    out["CDLMORNINGSTAR"] = (c2red & small1 & green & (c > mid2)).astype(int)
    out["CDLEVENINGSTAR"] = -(c2green & small1 & red & (c < mid2)).astype(int)
    out["CDLMARUBOZU"] = np.sign(body) * (ab >= 0.95 * rng)
    mid_p = (po + pc) / 2
    out["CDLPIERCING"] = (pred & green & (o < pc) & (c > mid_p) & (c < po)).astype(int)
    out["CDLDARKCLOUDCOVER"] = -(pgreen & red & (o > pc) & (c < mid_p) & (c > po)).astype(int)
    g3 = green & green.shift() & green.shift(2) & (c > pc) & (pc > c.shift(2))
    r3 = red & red.shift() & red.shift(2) & (c < pc) & (pc < c.shift(2))
    out["CDL3WHITESOLDIERS"] = g3.astype(int); out["CDL3BLACKCROWS"] = -r3.astype(int)
    return {k: np.sign(v.fillna(0)).astype("int8").to_numpy() for k, v in out.items()}
