"""Stage 4: live pattern detection. For each stock's latest bar: which candlestick patterns fire,
the SMA20/50 + RSI trend state, the model's probability, and how that pattern has historically done
in the stock's cluster (from the scorecard built by evaluate.py)."""
import json

import joblib
import numpy as np
import pandas as pd

from config import DATASET_FILE, FEATURES_FILE, MODEL_DIR, REPORT_DIR, TOP_QUANTILE

TREND = {1: "uptrend", 0: "sideways", -1: "downtrend"}


def main():
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    feats, model = meta["features"], joblib.load(MODEL_DIR / "gbm.pkl")
    ds = pd.read_pickle(DATASET_FILE)
    thr = np.quantile(model.predict_proba(ds.loc[ds["split"] == "val", feats])[:, 1], TOP_QUANTILE)

    allf = pd.read_pickle(FEATURES_FILE).sort_values(["symbol", "date"])
    last = allf.groupby("symbol").tail(1).copy()
    last["prob_up"] = model.predict_proba(last[feats])[:, 1]
    last["model_signal"] = last["prob_up"] >= thr
    last["trend"] = last["trend_state"].map(TREND)

    sc = pd.read_csv(REPORT_DIR / "pattern_scorecard.csv")
    pats = [c for c in feats if c.startswith("CDL")]
    rows = []
    for _, r in last.iterrows():
        fired = [(p, int(r[p])) for p in pats if r[p] != 0]
        if not fired:
            rows.append({**_base(r), "pattern": "", "direction": 0, "confirmed": False})
            continue
        for p, d in fired:
            m = sc[(sc["cluster"].astype(str) == str(int(r["cluster"]))) & (sc["pattern"] == p) & (sc["direction"] == d)]
            is_conf = bool(m["confirmed"].iloc[0]) if len(m) else False
            
            # Determine Trade Decision and Status
            if is_conf and d == 1:
                t_dec, stat = "Trade", "Confirmed Bullish"
            elif is_conf and d == -1:
                t_dec, stat = "Trade", "Confirmed Bearish"
            elif bool(r["model_signal"]) and d == 1:
                t_dec, stat = "Trade", "Top Decile Signal"
            elif not is_conf and d != 0:
                t_dec, stat = "Don't Trade", "Unconfirmed Pattern"
            else:
                t_dec, stat = "Don't Trade", "No Setup"

            rows.append({**_base(r), "pattern": p, "direction": d,
                         "confirmed": is_conf,
                         "trade_decision": t_dec, "status": stat,
                         "hist_n": int(m["n_disc"].iloc[0]) if len(m) else 0,
                         "hist_excess_ret": float(m["excess_ret_disc"].iloc[0]) if len(m) else np.nan})
    out = pd.DataFrame(rows).sort_values(["confirmed", "prob_up"], ascending=False)
    out.to_csv(REPORT_DIR / "latest_signals.csv", index=False)
    hits = out[out["pattern"] != ""]
    print(f"{len(last)} stocks scanned as of {last['date'].max().date()} | {len(hits)} pattern hits | "
          f"{int(last['model_signal'].sum())} model top-decile signals | {int(out['confirmed'].sum())} backed by a confirmed pattern")
    print(hits.head(15)[["symbol", "pattern", "direction", "trend", "prob_up", "trade_decision", "status"]].round(3).to_string(index=False))


def _base(r):
    return {"symbol": r["symbol"], "date": r["date"].date(), "cluster": int(r["cluster"]), "trend": r["trend"],
            "rsi14": round(float(r["rsi14"]), 1), "prob_up": float(r["prob_up"]), "model_signal": bool(r["model_signal"])}


if __name__ == "__main__":
    main()
