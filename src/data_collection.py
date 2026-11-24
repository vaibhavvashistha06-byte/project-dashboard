#Step 1: download 5 years of daily OHLCV for every stock in clustered_companies.csv

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
TICKER_FILE = ROOT / "data" / "clustered_companies.csv"
RAW_DIR = ROOT / "data" / "raw"
LOG_DIR = ROOT / "logs"
FAILED_FILE = LOG_DIR / "failed_tickers.csv"

PERIOD = "5y"
RETRIES = 3
PAUSE = 1.0  # seconds between tickers, to be polite to Yahoo


def to_yahoo(symbol: str) -> str:
    """NSE symbol -> Yahoo ticker, e.g. RELIANCE -> RELIANCE.NS"""
    return f"{symbol.strip()}.NS"


def safe_name(symbol: str) -> str:
    """M&M -> M_M so it is a safe file name."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in symbol)


def download_one(ticker: str) -> pd.DataFrame:
    """Download one stock, retrying on errors or empty results."""
    last_err = "empty result"
    for attempt in range(1, RETRIES + 1):
        try:
            df = yf.download(
                ticker,
                period=PERIOD,
                interval="1d",
                auto_adjust=True,   
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):  
                    df.columns = df.columns.get_level_values(0)
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                df.index.name = "Date"
                return df
        except Exception as e:  
            last_err = str(e)
        logging.warning("%s attempt %d/%d failed (%s)", ticker, attempt, RETRIES, last_err)
        time.sleep(2 * attempt)
    raise RuntimeError(last_err)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--limit", type=int, help="only the first N tickers (for testing)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_DIR / "download.log"), logging.StreamHandler()],
    )

    symbols = pd.read_csv(TICKER_FILE)["symbol"].dropna().astype(str).tolist()
    if args.limit:
        symbols = symbols[: args.limit]

    failed, done, skipped = [], 0, 0
    for i, sym in enumerate(symbols, 1):
        out = RAW_DIR / f"{safe_name(sym)}.csv"
        if out.exists() and not args.force:
            skipped += 1
            continue
        ticker = to_yahoo(sym)
        try:
            df = download_one(ticker)
            df.to_csv(out)
            done += 1
            logging.info("[%d/%d] %s saved: %d rows (%s to %s)", i, len(symbols), ticker,
                         len(df), df.index.min().date(), df.index.max().date())
        except Exception as e:
            failed.append({"symbol": sym, "ticker": ticker, "error": str(e)})
            logging.error("[%d/%d] %s FAILED: %s", i, len(symbols), ticker, e)
        time.sleep(PAUSE)

    if failed:
        pd.DataFrame(failed).to_csv(FAILED_FILE, index=False)
    logging.info("Done. downloaded=%d skipped=%d failed=%d", done, skipped, len(failed))
    if failed:
        logging.info("Failed tickers written to %s", FAILED_FILE)


if __name__ == "__main__":
    main()