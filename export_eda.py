"""
export_eda.py — regenerates frontend/data/*.js from the Bitcoin Pulse dataset.

Reproduces the analysis in IDA_Lab.ipynb (feature engineering, EDA, regime
labels) and, when xgboost is installed, retrains the market-regime classifier
to export its feature importance. Model comparison numbers are taken verbatim
from the notebook outputs so the dashboard matches the report.

Usage:  python export_eda.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "Bitcoin Pulse  Hourly Dataset from Markets Trends and Fear.csv"
OUT = ROOT / "frontend" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def write_js(filename, var_name, obj):
    path = OUT / filename
    path.write_text(f"window.{var_name} = {json.dumps(obj, separators=(',', ':'))};\n")
    print(f"  wrote frontend/data/{filename} ({path.stat().st_size / 1e6:.2f} MB)")


# =====================================================================
# 1. Load
# =====================================================================
df = pd.read_csv(CSV, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")

# =====================================================================
# 2. Feature engineering (IDA_Lab.ipynb cells 24-61)
# =====================================================================
data = df.copy()

data["btc_return"] = data["BTC_USDT_1h_close"].pct_change() * 100
data["btc_price_range"] = data["BTC_USDT_1h_high"] - data["BTC_USDT_1h_low"]
data["btc_range_pct"] = (data["btc_price_range"] / data["BTC_USDT_1h_close"]) * 100
data["btc_volume_change_pct"] = data["BTC_USDT_1h_volume"].pct_change() * 100
data["btc_volatility_6h"] = data["btc_return"].rolling(window=6).std()
data["btc_volatility_24h"] = data["btc_return"].rolling(window=24).std()
data["btc_ma_6h"] = data["BTC_USDT_1h_close"].rolling(window=6).mean()
data["btc_ma_24h"] = data["BTC_USDT_1h_close"].rolling(window=24).mean()
data["btc_ma_72h"] = data["BTC_USDT_1h_close"].rolling(window=72).mean()
data["btc_ma_24h_diff_pct"] = (
    (data["BTC_USDT_1h_close"] - data["btc_ma_24h"]) / data["btc_ma_24h"]
) * 100
data["btc_price_lag_1h"] = data["BTC_USDT_1h_close"].shift(1)
data["btc_price_lag_3h"] = data["BTC_USDT_1h_close"].shift(3)
data["btc_price_lag_6h"] = data["BTC_USDT_1h_close"].shift(6)
data["btc_return_lag_1h"] = data["btc_return"].shift(1)
data["btc_return_lag_3h"] = data["btc_return"].shift(3)
data["eth_return"] = data["ETH_USDT_1h_close"].pct_change() * 100
data["bnb_return"] = data["BNB_USDT_1h_close"].pct_change() * 100
data["sol_return"] = data["SOL_USDT_1h_close"].pct_change() * 100
data["xrp_return"] = data["XRP_USDT_1h_close"].pct_change() * 100
data["doge_return"] = data["DOGE_USDT_1h_close"].pct_change() * 100
data["nasdaq_return"] = data["Close_nasdaq"].pct_change() * 100
data["sp500_return"] = data["Close_sp500"].pct_change() * 100
data["dow_return"] = data["Close_dow_jones"].pct_change() * 100
data["russell_return"] = data["Close_russell_2000"].pct_change() * 100
data["vix_change_pct"] = data["Close_vix"].pct_change() * 100
data["btc_dominance_change"] = data["btc_dominance"].pct_change() * 100
data["altcoin_market_cap_change"] = data["altcoin_market_cap"].pct_change() * 100
data["fear_greed_change"] = data["fear_greed_index"].diff()
data["bitcoin_trend_change"] = data["google_trends_bitcoin"].diff()
data["buy_crypto_trend_change"] = data["google_trends_buy_crypto"].diff()
data["open_interest_change_pct"] = data["open_interest"].pct_change() * 100
data["hour"] = data["timestamp"].dt.hour
data["day_of_week"] = data["timestamp"].dt.dayofweek
data["day"] = data["timestamp"].dt.day
data["month"] = data["timestamp"].dt.month
data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
data["next_hour_return"] = data["btc_return"].shift(-1)
data["btc_direction"] = (data["next_hour_return"] > 0).astype(int)

engineered_features = [
    "btc_return", "btc_price_range", "btc_range_pct", "btc_volume_change_pct",
    "btc_volatility_6h", "btc_volatility_24h", "btc_ma_6h", "btc_ma_24h",
    "btc_ma_72h", "btc_ma_24h_diff_pct", "btc_price_lag_1h", "btc_price_lag_3h",
    "btc_price_lag_6h", "btc_return_lag_1h", "btc_return_lag_3h",
    "eth_return", "bnb_return", "sol_return", "xrp_return", "doge_return",
    "nasdaq_return", "sp500_return", "dow_return", "russell_return",
    "vix_change_pct", "btc_dominance_change", "altcoin_market_cap_change",
    "fear_greed_change", "bitcoin_trend_change", "buy_crypto_trend_change",
    "open_interest_change_pct", "hour", "day_of_week", "is_weekend",
]

data_ml = data.dropna(
    subset=engineered_features + ["next_hour_return", "btc_direction"]
).copy()
data_ml = data_ml.reset_index(drop=True)
print(f"Feature-engineered rows: {len(data_ml)} (notebook: 17444)")
assert len(data_ml) == 17444, f"expected 17444 rows, got {len(data_ml)}"

# =====================================================================
# 3. Regime label (IDA_Lab.ipynb cell 163)
# =====================================================================
vol_threshold = data_ml["btc_volatility_24h"].median()
data_ml["volatility_regime"] = (data_ml["btc_volatility_24h"] >= vol_threshold).astype(int)


def sentiment_category(value):
    if value <= 25:
        return "Extreme Fear"
    if value <= 45:
        return "Fear"
    if value <= 55:
        return "Neutral"
    if value <= 75:
        return "Greed"
    return "Extreme Greed"


data_ml["sentiment_category"] = data_ml["fear_greed_index"].apply(sentiment_category)

# =====================================================================
# 4. overview.js
# =====================================================================
print("\nBuilding overview...")
missing = df.isnull().sum()
missing_pct = missing / len(df) * 100
top_missing = (
    pd.DataFrame({"count": missing, "pct": missing_pct})
    .sort_values("count", ascending=False)
    .head(20)
)
diff = df["timestamp"].diff()
real_gaps = int((diff.notna() & (diff != pd.Timedelta(hours=1))).sum())

overview = {
    "rows": int(df.shape[0]),
    "cols": int(df.shape[1]),
    "start": df["timestamp"].min().strftime("%Y-%m-%d"),
    "end": df["timestamp"].max().strftime("%Y-%m-%d"),
    "duplicateRows": int(df.duplicated().sum()),
    "duplicateTimestamps": int(df["timestamp"].duplicated().sum()),
    "gaps": real_gaps,
    "btcMin": round(float(df["BTC_USDT_1h_close"].min()), 2),
    "btcMax": round(float(df["BTC_USDT_1h_close"].max()), 2),
    "missing": [
        {"name": str(i), "count": int(r["count"]), "pct": round(float(r["pct"]), 2)}
        for i, r in top_missing.iterrows()
    ],
    "missingTotalCells": int(missing.sum()),
}

# =====================================================================
# 5. series.js  (full-resolution raw series; MA/returns/vol derived in JS)
# =====================================================================
print("Building series...")
ts = (
    (df["timestamp"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
).astype("int64").tolist()  # epoch ms (unit-proof across pandas datetime precisions)


def pts(values, ndigits=2):
    return [{"x": t, "y": round(float(v), ndigits)} for t, v in zip(ts, values) if pd.notna(v)]


series = {
    "price": pts(df["BTC_USDT_1h_close"], 2),
    "volume": pts(df["BTC_USDT_1h_volume"], 2),
    "fearGreed": pts(df["fear_greed_index"], 1),
}

# =====================================================================
# 6. returns.js  (stats + histogram on data_ml, matching notebook cell 69/70)
# =====================================================================
print("Building returns...")
r = data_ml["btc_return"]
hist_counts, hist_edges = np.histogram(r, bins=100)
returns = {
    "stats": {
        "count": int(r.count()),
        "mean": round(float(r.mean()), 4),
        "std": round(float(r.std()), 4),
        "min": round(float(r.min()), 4),
        "q25": round(float(r.quantile(0.25)), 4),
        "median": round(float(r.median()), 4),
        "q75": round(float(r.quantile(0.75)), 4),
        "max": round(float(r.max()), 4),
        "skew": round(float(r.skew()), 3),
        "kurtosis": round(float(r.kurtosis()), 3),
    },
    "histEdges": [round(float(e), 4) for e in hist_edges],
    "histCounts": [int(c) for c in hist_counts],
}

# =====================================================================
# 7. drivers.js  (regime-conditioned findings)
# =====================================================================
print("Building drivers...")

zone_order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
zone_stats = data_ml.groupby("sentiment_category")["volatility_regime"].agg(["count", "mean"])
sentiment_zones = [
    {
        "name": z,
        "n": int(zone_stats.loc[z, "count"]),
        "pctHighVol": round(float(zone_stats.loc[z, "mean"]) * 100, 1),
    }
    for z in zone_order
]

weekend_stats = data_ml.groupby("is_weekend")["volatility_regime"].agg(["count", "mean"])
high_vol = data_ml[data_ml["volatility_regime"] == 1]
weekend = {
    "weekday": {
        "n": int(weekend_stats.loc[0, "count"]),
        "pctHighVol": round(float(weekend_stats.loc[0, "mean"]) * 100, 1),
    },
    "weekend": {
        "n": int(weekend_stats.loc[1, "count"]),
        "pctHighVol": round(float(weekend_stats.loc[1, "mean"]) * 100, 1),
    },
    "highVolWeekendShare": round(float(high_vol["is_weekend"].mean()) * 100, 1),
}

month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
month_stats = data_ml.groupby("month")["volatility_regime"].agg(["count", "mean"])
monthly = [
    {
        "name": month_names[m - 1],
        "n": int(month_stats.loc[m, "count"]),
        "pctHighVol": round(float(month_stats.loc[m, "mean"]) * 100, 1),
    }
    for m in range(1, 13)
]

level_vs_return = [
    {
        "name": "S&P 500",
        "levelCorr": round(float(data_ml["Close_sp500"].corr(data_ml["BTC_USDT_1h_close"])), 3),
        "returnCorr": round(float(data_ml["sp500_return"].corr(data_ml["btc_return"])), 3),
    },
    {
        "name": "NASDAQ",
        "levelCorr": round(float(data_ml["Close_nasdaq"].corr(data_ml["BTC_USDT_1h_close"])), 3),
        "returnCorr": round(float(data_ml["nasdaq_return"].corr(data_ml["btc_return"])), 3),
    },
    {
        "name": "Dow Jones",
        "levelCorr": round(float(data_ml["Close_dow_jones"].corr(data_ml["BTC_USDT_1h_close"])), 3),
        "returnCorr": round(float(data_ml["dow_return"].corr(data_ml["btc_return"])), 3),
    },
]

factor_features = [
    "btc_volatility_6h", "btc_volatility_24h", "btc_volume_change_pct",
    "eth_return", "bnb_return", "sol_return", "xrp_return", "doge_return",
    "nasdaq_return", "sp500_return", "dow_return", "russell_return",
    "vix_change_pct", "fear_greed_change",
    "btc_dominance_change", "altcoin_market_cap_change", "open_interest_change_pct",
]
factor_corr = data_ml[factor_features + ["btc_return"]].corr()["btc_return"].drop("btc_return")
factor_rank = [
    {"name": name, "corr": round(float(val), 3)}
    for name, val in factor_corr.reindex(factor_corr.abs().sort_values(ascending=False).index).items()
]

crypto_labels = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE"]
crypto_cols = ["btc_return", "eth_return", "bnb_return", "sol_return", "xrp_return", "doge_return"]
crypto_matrix = data_ml[crypto_cols].corr().round(2)

drivers = {
    "sentimentZones": sentiment_zones,
    "weekend": weekend,
    "monthly": monthly,
    "levelVsReturn": level_vs_return,
    "factorRank": factor_rank,
    "cryptoMatrix": {
        "labels": crypto_labels,
        "values": [[None if pd.isna(v) else float(v) for v in row]
                   for row in crypto_matrix.to_numpy()],
    },
    "regimeMedianVol": round(float(vol_threshold), 4),
}

# =====================================================================
# 8. models.js — notebook numbers (static) + regime retrain (importance)
# =====================================================================
print("Building models...")

# Verbatim from IDA_Lab.ipynb cells 158 + 160 (6-hour direction, 6 models)
six_model = [
    {"model": "Logistic Regression", "accuracy": 0.5107, "precision": 0.5137,
     "recall": 0.7163, "f1": 0.5983, "auc": 0.5270},
    {"model": "Hist Gradient Boosting", "accuracy": 0.5099, "precision": 0.5149,
     "recall": 0.6326, "f1": 0.5677, "auc": 0.5137},
    {"model": "Random Forest", "accuracy": 0.5076, "precision": 0.5150,
     "recall": 0.5500, "f1": 0.5319, "auc": 0.5154},
    {"model": "Extra Trees", "accuracy": 0.5076, "precision": 0.5157,
     "recall": 0.5264, "f1": 0.5210, "auc": 0.5087},
    {"model": "Gradient Boosting", "accuracy": 0.5024, "precision": 0.5087,
     "recall": 0.6438, "f1": 0.5683, "auc": 0.5112},
    {"model": "XGBoost", "accuracy": 0.5047, "precision": 0.5109,
     "recall": 0.6197, "f1": 0.5600, "auc": 0.5103},
]

# Verbatim from IDA_Lab.ipynb cell 164 (market regime classifier)
regime_notebook = {
    "accuracy": 0.7873, "precision": 0.7852, "recall": 0.9145,
    "f1": 0.8449, "auc": 0.8701,
    "cm": [[726, 553], [189, 2021]],
}

regime = {
    "metrics": {k: regime_notebook[k]
                for k in ("accuracy", "precision", "recall", "f1", "auc")},
    "cm": regime_notebook["cm"],
    "cmLabels": ["LOW VOLATILITY", "HIGH VOLATILITY"],
    "metricsSource": "notebook",
    "importance": [],
    "importanceSource": None,
    "retrainAccuracy": None,
}

try:
    from xgboost import XGBClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    )

    regime_features = [f for f in engineered_features if f != "btc_volatility_24h"]
    X_r = data_ml[regime_features]
    y_r = data_ml["volatility_regime"]
    split = int(len(X_r) * 0.80)

    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=3,
        subsample=0.85, colsample_bytree=0.85, gamma=0.1, reg_alpha=0.1,
        reg_lambda=1, objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=4,
    )
    xgb.fit(X_r.iloc[:split], y_r.iloc[:split])
    pred = xgb.predict(X_r.iloc[split:])
    prob = xgb.predict_proba(X_r.iloc[split:])[:, 1]

    retrain_acc = float(accuracy_score(y_r.iloc[split:], pred))
    regime["retrainAccuracy"] = round(retrain_acc, 4)

    # Importance always from the retrain (notebook never exported it)
    imp = pd.DataFrame({
        "feature": regime_features,
        "importance": [float(v) for v in xgb.feature_importances_],
    }).sort_values("importance", ascending=False).head(15)
    regime["importance"] = [
        {"feature": r["feature"], "importance": round(r["importance"], 5)}
        for _, r in imp.iterrows()
    ]
    regime["importanceSource"] = "retrained"

    # Use retrained metrics only if they reproduce the notebook's headline
    if abs(retrain_acc - regime_notebook["accuracy"]) <= 0.005:
        regime["metrics"] = {
            "accuracy": round(retrain_acc, 4),
            "precision": round(float(precision_score(y_r.iloc[split:], pred)), 4),
            "recall": round(float(recall_score(y_r.iloc[split:], pred)), 4),
            "f1": round(float(f1_score(y_r.iloc[split:], pred)), 4),
            "auc": round(float(roc_auc_score(y_r.iloc[split:], prob)), 4),
        }
        from sklearn.metrics import confusion_matrix
        regime["cm"] = confusion_matrix(y_r.iloc[split:], pred).tolist()
        regime["metricsSource"] = "retrained (matches notebook 78.73%)"
    else:
        regime["metricsSource"] = (
            f"notebook (retrain gave {retrain_acc:.2%}, outside ±0.5% tolerance)"
        )
    print(f"  regime retrain accuracy: {retrain_acc:.4f} "
          f"(notebook: {regime_notebook['accuracy']})")
except Exception as exc:  # xgboost missing or training failed
    regime["metricsSource"] = "notebook (xgboost unavailable: %s)" % exc
    print(f"  WARNING: regime retrain skipped: {exc}")

models = {
    "regime": regime,
    "sixModel": six_model,
    "sixModelTarget": "next 6-hour BTC direction",
}

# =====================================================================
# 9. Write everything
# =====================================================================
print("\nWriting frontend/data ...")
write_js("overview.js", "OVERVIEW", overview)
write_js("series.js", "SERIES", series)
write_js("returns.js", "RETURNS", returns)
write_js("drivers.js", "DRIVERS", drivers)
write_js("models.js", "MODELS", models)

# =====================================================================
# 10. Verification summary
# =====================================================================
print("\n--- verification vs notebook / report ---")
print(f"rows: {overview['rows']} (expect 17516), cols: {overview['cols']} (expect 99)")
print(f"gap hours: {overview['gaps']}, duplicate rows: {overview['duplicateRows']}")
print(f"kurtosis: {returns['stats']['kurtosis']} (notebook: 12.441)")
print("sentiment zone %High-Vol:",
      {z['name']: z['pctHighVol'] for z in sentiment_zones}, "(report: EF 96, Neutral 30)")
print(f"weekend %High-Vol: {weekend['weekend']['pctHighVol']}, "
      f"weekday: {weekend['weekday']['pctHighVol']} (report: 26 vs 60)")
print("level vs return corr:",
      {x['name']: (x['levelCorr'], x['returnCorr']) for x in level_vs_return},
      "(report: S&P 0.94 vs 0.09)")
print(f"regime metrics source: {regime['metricsSource']}")
print(f"regime importance source: {regime['importanceSource']}")
print("done.")
