#final check on the TEST period, simulated trades, feature importance, and the per-cluster pattern scorecard.

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from config import (COST, DATASET_FILE, HORIZON, MIN_SIGNALS, MODEL_DIR, RANDOM_STATE,
                    REPORT_DIR, TOP_QUANTILE)


def trade_stats(df, mask, name):
    s = df[mask]
    if len(s) == 0:
        return {"group": name, "trades": 0}
    net = s["fwd_ret"] - COST
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()
    # portfolio curve: each day's new trades get 1/HORIZON of capital (HORIZON holds overlap); idle days earn 0
    daily = (net.groupby(s["date"]).mean() / HORIZON).reindex(sorted(df["date"].unique()), fill_value=0.0)
    equity = (1 + daily).cumprod()
    dd = (equity / equity.cummax() - 1).min()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan
    return {"group": name, "trades": len(s), "avg_net_ret_%": round(100 * net.mean(), 3),
            "win_rate_%": round(100 * (net > 0).mean(), 1),
            "profit_factor": round(wins / losses, 2) if losses > 0 else np.nan,
            "sharpe": round(sharpe, 2), "max_drawdown_%": round(100 * dd, 1)}


def scorecard(df, pat_cols):
    rows = []
    for cl, g in [("ALL", df)] + [(int(k), v) for k, v in df.groupby("cluster")]:
        base, base_up = g["fwd_ret"].mean(), (g["fwd_ret"] > 0).mean()
        for p in pat_cols:
            for d in (1, -1):
                s = g[g[p] == d]
                if len(s):
                    rows.append({"cluster": cl, "pattern": p, "direction": d, "n": len(s),
                                 "excess_ret": s["fwd_ret"].mean() - base,
                                 "excess_hit": (s["fwd_ret"] > 0).mean() - base_up})
    return pd.DataFrame(rows)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_pickle(DATASET_FILE)
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    features = meta["features"]
    model = joblib.load(MODEL_DIR / "gbm.pkl")
    va, te = data[data["split"] == "val"].copy(), data[data["split"] == "test"].copy()
    if te.empty:
        raise SystemExit("No test rows - check the dates in config.py")
    va["p"], te["p"] = model.predict_proba(va[features])[:, 1], model.predict_proba(te[features])[:, 1]

    # model quality
    print("=== MODEL QUALITY ===")
    print(f"AUC  val {roc_auc_score(va['label'], va['p']):.3f} | test {roc_auc_score(te['label'], te['p']):.3f}  (0.5 = coin flip)")
    thr = va["p"].quantile(TOP_QUANTILE)          # threshold chosen on VALIDATION, applied to test
    top = te["p"] >= thr
    print(f"Share of 'up' labels on test: all days {te['label'].mean():.3f} | top-confidence days {te.loc[top, 'label'].mean():.3f}")

    y, pred = te["label"], top.astype(int)
    f1m = {"threshold": float(thr), "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
           "base_rate": float(y.mean()), "auc_val": float(roc_auc_score(va["label"], va["p"])), "auc_test": float(roc_auc_score(y, te["p"]))}
    print(f"At the top-{int((1 - TOP_QUANTILE) * 100)}% cutoff: precision {f1m['precision']:.3f} | recall {f1m['recall']:.3f} | F1 {f1m['f1']:.3f} "
          f"(a random pick would score precision {f1m['base_rate']:.3f})")

    # simulated trades
    print(f"\n=== SIMULATED TRADES on test (cost {COST:.1%} per trade; holds overlap, so treat as rough) ===")
    out = [trade_stats(te, te["p"].notna(), "buy every day (baseline)"),
           trade_stats(te, top, f"model top {int((1 - TOP_QUANTILE) * 100)}%")]
    for cl in sorted(te["cluster"].unique()):
        out.append(trade_stats(te, top & (te["cluster"] == cl), f"  model top, cluster {cl}"))
    tbl = pd.DataFrame(out)
    print(tbl.to_string(index=False))
    tbl.to_csv(REPORT_DIR / "trade_metrics.csv", index=False)
    (REPORT_DIR / "model_metrics.json").write_text(json.dumps(f1m, indent=2))
    print("win_rate = share of profitable trades | profit_factor = gross wins / gross losses (>1 profitable) | "
          "sharpe = annualised, capital split across overlapping holds | max_drawdown = worst peak-to-trough")

    # which features matter
    print("\n=== FEATURE IMPORTANCE (drop in validation AUC when shuffled) ===")
    sample = va.sample(min(20000, len(va)), random_state=RANDOM_STATE)
    imp = permutation_importance(model, sample[features], sample["label"], scoring="roc_auc",
                                 n_repeats=3, random_state=RANDOM_STATE, n_jobs=1)
    fi = pd.DataFrame({"feature": features, "importance": imp.importances_mean}).sort_values("importance", ascending=False)
    fi.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    print(fi.head(15).to_string(index=False))

    # pattern scorecard per cluster
    pats = [c for c in features if c.startswith("CDL")]
    disc = scorecard(data[data["split"].isin(["train", "val"])], pats)    # discovery period
    conf = scorecard(te, pats)                                              # confirmation period
    sc = disc.merge(conf, on=["cluster", "pattern", "direction"], how="left", suffixes=("_disc", "_test"))
    sc = sc[sc["n_disc"] >= MIN_SIGNALS].copy()
    sc["confirmed"] = ((sc["excess_ret_disc"] * sc["direction"] > 0) &
                       (sc["excess_ret_test"] * sc["direction"] > 0) & (sc["n_test"] >= 10))
    sc = sc.sort_values(["confirmed", "excess_ret_disc"], ascending=[False, False])
    sc.to_csv(REPORT_DIR / "pattern_scorecard.csv", index=False)
    print(f"\n=== PATTERN SCORECARD === ({len(sc)} pattern/cluster combos with >= {MIN_SIGNALS} signals)")
    print(f"'confirmed' = pattern beat the average in BOTH discovery and test periods (in its expected direction): {int(sc['confirmed'].sum())}")
    show = sc[sc["confirmed"]].head(15)[["cluster", "pattern", "direction", "n_disc", "excess_ret_disc", "n_test", "excess_ret_test"]]
    print(show.round(4).to_string(index=False) if len(show) else "none confirmed")
    print(f"\nFull tables saved in {REPORT_DIR}")
    print("With ~120 patterns x 4 groups, a few will 'confirm' by pure luck. Trust patterns with large n and consistent results.")


if __name__ == "__main__":
    main()