#turn each stock's prices into one row of features per day.

import numpy as np
import pandas as pd
try:
    import talib
except ImportError:          # runs without TA-Lib, with 12 core patterns instead of all 61
    talib = None
    import ta_fallback as tf

from config import COMPANIES_FILE, FEATURES_FILE, PROCESSED_DIR


def safe_name(symbol):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in symbol)


def load_prices(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str[:10])  # drop any timezone suffix
    return df.set_index("Date").sort_index()[["Open", "High", "Low", "Close", "Volume"]].astype("float64")


def build_features(df):
    o, h, l, c, v = (df[x].to_numpy() for x in ["Open", "High", "Low", "Close", "Volume"])
    f = pd.DataFrame(index=df.index)

    #1. candlestick patterns: +1 bullish, -1 bearish, 0 none ---
    if talib is not None:
        for name in talib.get_function_groups()["Pattern Recognition"]:
            f[name] = np.sign(getattr(talib, name)(o, h, l, c)).astype("int8")
        SMA, ATR, BB, MACD, RSI, ADX = (talib.SMA, talib.ATR, lambda x: talib.BBANDS(x, 20, 2, 2),
                                        lambda x: talib.MACD(x, 12, 26, 9)[2], talib.RSI, talib.ADX)
    else:
        for name, vals in tf.patterns(o, h, l, c).items():
            f[name] = vals
        SMA, ATR, BB, MACD, RSI, ADX = (tf.sma, tf.atr, lambda x: tf.bbands(x, 20, 2), tf.macd_hist, tf.rsi, tf.adx)

    #2. context: trend, momentum, volatility, volume ---
    sma20v = SMA(v, 20)
    sma20, sma50, sma200 = SMA(c, 20), SMA(c, 50), SMA(c, 200)
    atr = ATR(h, l, c, 14)
    upper, _, lower = BB(c)
    macd_hist = MACD(c)

    # trend regime from SMA20/SMA50 + RSI (team suggestion): describes the stock NOW; the label is still the forward return
    rsi = RSI(c, 14)
    f["dist_sma20"] = c / sma20 - 1
    f["sma20_vs_50"] = sma20 / sma50 - 1
    f["trend_state"] = np.where((sma20 > sma50) & (c > sma50), 1, np.where((sma20 < sma50) & (c < sma50), -1, 0)).astype("int8")
    f["rsi_zone"] = np.where(rsi > 70, 1, np.where(rsi < 30, -1, 0)).astype("int8")

    f["dist_sma50"] = c / sma50 - 1
    f["dist_sma200"] = c / sma200 - 1
    f["sma50_vs_200"] = sma50 / sma200 - 1
    f["rsi14"] = rsi
    f["macd_hist_pct"] = macd_hist / c
    f["adx14"] = ADX(h, l, c, 14)
    f["atr_pct"] = atr / c
    f["bb_pos"] = (c - lower) / (upper - lower)
    f["vol_ratio"] = v / sma20v
    f["dist_high20"] = c / pd.Series(h, index=df.index).rolling(20).max() - 1
    f["dist_low20"] = c / pd.Series(l, index=df.index).rolling(20).min() - 1
    for n in (1, 5, 10, 20):
        f[f"ret_{n}"] = df["Close"].pct_change(n)

    #2b. extra momentum / volatility / pattern-count features
    r1 = df["Close"].pct_change()
    f["ret_60"] = df["Close"].pct_change(60)
    f["vol_20"] = r1.rolling(20).std()
    f["vol_ratio_5_20"] = r1.rolling(5).std() / f["vol_20"]
    f["dist_high120"] = c / pd.Series(h, index=df.index).rolling(120).max() - 1
    f["up_days_10"] = (r1 > 0).rolling(10).sum()
    pc = [x for x in f.columns if x.startswith("CDL")]
    f["n_bull_pat"], f["n_bear_pat"] = (f[pc] > 0).sum(axis=1), (f[pc] < 0).sum(axis=1)
    f["dow"] = df.index.dayofweek

    #3. candle shape (relative, so cheap and expensive stocks look alike) ---
    rng = pd.Series(h - l, index=df.index).replace(0, np.nan)
    f["body"] = (df["Close"] - df["Open"]) / rng              # signed: + green, - red
    f["upper_wick"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / rng
    f["lower_wick"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / rng
    f["gap"] = df["Open"] / df["Close"].shift() - 1
    f["range_atr"] = rng / atr

    # price columns kept only for labelling / simulation, NOT used as model inputs
    f["open"], f["close"] = df["Open"], df["Close"]
    return f.dropna(subset=["dist_sma200", "atr_pct"])      # drop the 200-day warm-up period


def add_market_features(data):
    """Market proxy = equal-weight average of the universe itself (no extra download needed).
    Everything uses only information up to each day's close."""
    g = data.groupby("date")
    for n in (1, 5, 20):
        data[f"mkt_ret_{n}"] = g[f"ret_{n}"].transform("mean")
    data["mkt_breadth"] = g["trend_state"].transform(lambda x: (x > 0).mean())      # share of stocks in uptrend
    mkt = (1 + g["ret_1"].mean().fillna(0)).cumprod()
    mk = pd.DataFrame({"mkt_dist_sma50": mkt / mkt.rolling(50).mean() - 1,
                       "mkt_vol_20": mkt.pct_change().rolling(20).std()})
    data = data.merge(mk, left_on="date", right_index=True, how="left")
    data["rel_ret_5"] = data["ret_5"] - data["mkt_ret_5"]
    data["rel_ret_20"] = data["ret_20"] - data["mkt_ret_20"]
    data["sector_rel_ret_5"] = data["ret_5"] - data.groupby(["date", "sector_code"])["ret_5"].transform("mean")
    data["rank_ret_5"] = g["ret_5"].rank(pct=True)
    data["rank_ret_20"] = g["ret_20"].rank(pct=True)
    return data


def main():
    companies = pd.read_csv(COMPANIES_FILE)
    sectors = {s: i for i, s in enumerate(sorted(companies["sector"].dropna().unique()))}
    parts = []
    for _, row in companies.iterrows():
        path = PROCESSED_DIR / f"{safe_name(row['symbol'])}.csv"
        if not path.exists():
            print("missing:", row["symbol"])
            continue
        f = build_features(load_prices(path))
        f["symbol"] = row["symbol"]
        f["cluster"] = int(row["cluster"])
        f["sector_code"] = sectors.get(row["sector"], -1)
        parts.append(f.rename_axis("date").reset_index())

    data = pd.concat(parts, ignore_index=True)
    data = add_market_features(data)
    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_pickle(FEATURES_FILE)
    pat = [c for c in data.columns if c.startswith("CDL")]
    print(f"Saved {len(data):,} rows, {data['symbol'].nunique()} stocks, {len(data.columns)} columns -> {FEATURES_FILE}")
    fired = (data[pat] != 0).mean().sort_values()
    print(f"\nPatterns that never fire: {int((fired == 0).sum())} of {len(pat)}")
    print("Most frequent patterns (share of days):")
    print((fired.tail(8) * 100).round(1).to_string())


if __name__ == "__main__":
    main()
