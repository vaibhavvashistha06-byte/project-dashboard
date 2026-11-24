from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"



MIN_ROWS = 1000
MAX_MISSING_PCT = 1.0
MAX_ZERO_VOL_PCT = 2.0
MAX_GAP_DAYS = 10
MIN_MEDIAN_TURNOVER_CR = 5
MAX_BAD_CANDLE_PCT = 0.5


def safe_name(symbol):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in symbol)


def assess(path):
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    cols = ["Open", "High", "Low", "Close", "Volume"]
    n = len(df)
    nan_rows = df[cols].isna().any(axis=1)
    df = df[~nan_rows]
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    bad = (df["High"] < df[["Open", "Close", "Low"]].max(axis=1)) | (df["Low"] > df[["Open", "Close", "High"]].min(axis=1))
    gaps = df.index.to_series().diff().dt.days
    turnover_cr = (df["Close"] * df["Volume"]) / 1e7
    m = {
        "rows": n,
        "start": df.index.min().date() if len(df) else None,
        "end": df.index.max().date() if len(df) else None,
        "missing_pct": round(100 * nan_rows.mean(), 2) if n else 100,
        "zero_vol_pct": round(100 * (df["Volume"] == 0).mean(), 2) if len(df) else 100,
        "max_gap_days": int(gaps.max()) if len(df) > 1 else 999,
        "median_turnover_cr": round(turnover_cr.median(), 2) if len(df) else 0,
        "bad_candle_pct": round(100 * bad.mean(), 2) if len(df) else 100,
    }
    reasons = []
    if m["rows"] < MIN_ROWS: reasons.append(f"rows<{MIN_ROWS}")
    if m["missing_pct"] > MAX_MISSING_PCT: reasons.append("too many NaN")
    if m["zero_vol_pct"] > MAX_ZERO_VOL_PCT: reasons.append("zero volume")
    if m["max_gap_days"] > MAX_GAP_DAYS: reasons.append("big gap")
    if m["median_turnover_cr"] < MIN_MEDIAN_TURNOVER_CR: reasons.append("illiquid")
    if m["bad_candle_pct"] > MAX_BAD_CANDLE_PCT: reasons.append("bad candles")
    m["passed"] = not reasons
    m["reason"] = "; ".join(reasons)
    return m, df[~bad.reindex(df.index, fill_value=False)]


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    companies = pd.read_csv(ROOT / "data" / "clustered_companies.csv")
    rows = []
    for sym in companies["symbol"]:
        f = RAW / f"{safe_name(sym)}.csv"
        if not f.exists():
            rows.append({"symbol": sym, "passed": False, "reason": "file missing"})
            continue
        m, clean = assess(f)
        rows.append({"symbol": sym, **m})
        if m["passed"]:
            clean.to_csv(PROCESSED / f.name)
    report = pd.DataFrame(rows).merge(companies[["symbol", "sector", "cluster"]], on="symbol", how="left")
    report.to_csv(ROOT / "data" / "quality_report.csv", index=False)
    companies[companies["symbol"].isin(report.loc[report["passed"], "symbol"])].to_csv(
        ROOT / "data" / "filtered_companies.csv", index=False)

    print(f"Passed {report['passed'].sum()} / {len(report)}")
    print("\nPassed per cluster:\n", report.groupby("cluster")["passed"].agg(["sum", "count"]))
    print("\nRejected:\n", report.loc[~report["passed"], ["symbol", "rows", "start", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()