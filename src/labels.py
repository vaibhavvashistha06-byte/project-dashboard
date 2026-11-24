"""Stage 2: add the target (what we want to predict) and the train/val/test split.

Trade logic: signal at day t's close -> buy at day t+1's OPEN -> sell at day t+HORIZON's close.
label = 1 if that return beats ATR_MULT * daily-ATR% * sqrt(HORIZON)  (volatility-adjusted target).
"""

import numpy as np
import pandas as pd

from config import (ATR_MULT, LABEL_MODE, DATASET_FILE, EMBARGO_DAYS, FEATURES_FILE, HORIZON,
                    TRAIN_END, VAL_END)


def add_labels(g):
    entry = g["open"].shift(-1)
    exit_ = g["close"].shift(-HORIZON)
    g["fwd_ret"] = exit_ / entry - 1
    return g


def main():
    data = pd.read_pickle(FEATURES_FILE).sort_values(["symbol", "date"])
    data = data.groupby("symbol", group_keys=False).apply(add_labels)
    data = data.dropna(subset=["fwd_ret"])                       #last HORIZON days have no outcome yet

    threshold = ATR_MULT * data["atr_pct"] * np.sqrt(HORIZON)
    if LABEL_MODE == "relative":        # remove the market-wide move: compare to the average stock that day
        data["fwd_excess"] = data["fwd_ret"] - data.groupby("date")["fwd_ret"].transform("mean")
        data["label"] = (data["fwd_excess"] > threshold).astype("int8")
    else:
        data["label"] = (data["fwd_ret"] > threshold).astype("int8")

    #time split with an embargo gap so a training label never overlaps validation/test days
    t_end, v_end = pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END)
    gap = pd.Timedelta(days=EMBARGO_DAYS)
    d = data["date"]
    data["split"] = "drop"
    data.loc[d <= t_end - gap, "split"] = "train"
    data.loc[(d > t_end) & (d <= v_end - gap), "split"] = "val"
    data.loc[d > v_end, "split"] = "test"
    data = data[data["split"] != "drop"]

    DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_pickle(DATASET_FILE)

    print(f"Saved {len(data):,} rows -> {DATASET_FILE}\n")
    summary = data.groupby("split").agg(rows=("label", "size"), share_up=("label", "mean"),
                                        avg_fwd_ret=("fwd_ret", "mean"),
                                        first=("date", "min"), last=("date", "max"))
    num = ["share_up", "avg_fwd_ret"]
    summary[num] = summary[num].round(4)
    print(summary.to_string())
    if (data["split"] == "test").sum() == 0:
        print("\nWARNING: no test rows. Your data may end before the test period; adjust dates in config.py")


if __name__ == "__main__":
    main()