#training the models and compare them on VALIDATION data.

import json

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import DATASET_FILE, MODEL_DIR, RANDOM_STATE

NON_FEATURES = ["symbol", "date", "open", "close", "fwd_ret", "label", "split", "fwd_excess"]


def get_gbm():
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=200,
            subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
            random_state=RANDOM_STATE, verbose=-1)
        return model, "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=4, max_iter=300, min_samples_leaf=200,
            l2_regularization=5.0, early_stopping=True, validation_fraction=0.15,
            random_state=RANDOM_STATE)
        return model, "sklearn-histgb"


def main():
    data = pd.read_pickle(DATASET_FILE)
    features = [c for c in data.columns if c not in NON_FEATURES]
    tr, va = data[data["split"] == "train"], data[data["split"] == "val"]
    Xtr, ytr, Xva, yva = tr[features], tr["label"], va[features], va["label"]
    print(f"{len(features)} features | train {len(tr):,} rows | val {len(va):,} rows")
    print(f"share of 'up' labels: train {ytr.mean():.3f}, val {yva.mean():.3f}\n")

    results = {}
    # 0. baseline: same score for everyone -> AUC 0.5 by definition
    print("baseline (no skill)        AUC 0.500")

    # 1. logistic regression
    logit = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=0.1, max_iter=500))
    logit.fit(Xtr, ytr)
    results["logistic"] = roc_auc_score(yva, logit.predict_proba(Xva)[:, 1])
    print(f"logistic regression        AUC {results['logistic']:.3f}")

    # 2. gradient boosting
    gbm, kind = get_gbm()
    if kind == "lightgbm":
        import lightgbm as lgb
        gbm.fit(Xtr, ytr, eval_X=Xva, eval_y=yva, eval_metric="auc",
                callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        gbm.fit(Xtr, ytr)
    results["gbm"] = roc_auc_score(yva, gbm.predict_proba(Xva)[:, 1])
    print(f"gradient boosting ({kind})  AUC {results['gbm']:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(gbm, MODEL_DIR / "gbm.pkl")
    joblib.dump(logit, MODEL_DIR / "logistic.pkl")
    (MODEL_DIR / "meta.json").write_text(json.dumps({"features": features, "gbm_kind": kind,
                                                      "val_auc": results}, indent=2))
    print(f"\nSaved models to {MODEL_DIR}")
    print("AUC 0.50 = coin flip. 0.52-0.56 would already be interesting. Above ~0.65: suspect a leak/bug.")


if __name__ == "__main__":
    main()
