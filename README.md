# Bitcoin Pulse — Market Regime Dashboard

A static, dependency-free dashboard for the **IDA_Lab** Bitcoin analysis: 17,516 hourly
observations of crypto prices, 12 global equity indices and 8 sentiment indicators,
engineered into 34 features and used to train an XGBoost **volatility-regime classifier
(78.7% accuracy, 91.5% High-Vol recall)**.

![Dashboard screenshot](frontend/screenshot.png)

## What's inside the dashboard

| Section | Contents |
|---|---|
| **Hero** | Headline result: regime accuracy, High-Vol recall, ROC-AUC |
| **1 · The data** | Dataset KPIs, column composition, quality checks (0 missing, 0 gaps, 0 duplicates) |
| **2 · Price & sentiment** | BTC close + 24h MA, hourly volume, BTC vs Fear & Greed (dual axis) |
| **3 · Returns & volatility** | Return histogram (kurtosis 12.44), 6h vs 24h rolling volatility |
| **4 · What drives volatility** | Fear & Greed U-shape (96% vs 30% High-Vol), weekend effect (26% vs 60%), monthly share, level-vs-return correlation trap (0.94 → 0.09), factor rankings, crypto correlation heatmap |
| **5 · Model results** | Regime confusion matrix + metrics, XGBoost feature importance, six-model comparison on 6-hour direction |
| **6 · Limitations** | Label-window overlap caveat, median-split caveat, negative direction result, static-export note |

Every chart carries a one-line insight caption.

## Run it locally

No build step, no server required — everything is plain HTML/JS with a vendored copy
of Chart.js, and data lives in `window.*` JS files so `fetch()` is never used:

```bash
# option A: just open it
open frontend/index.html        # macOS
xdg-open frontend/index.html    # Linux

# option B: serve it (needed only if your browser blocks local scripts)
cd frontend && python3 -m http.server 8000
# → http://localhost:8000
```

## Regenerate the data

`frontend/data/*.js` is produced from the CSV by:

```bash
pip install pandas numpy scikit-learn xgboost   # xgboost only needed for feature importance
python export_eda.py
```

The script reproduces the notebook's feature engineering (17,444 modelling rows),
recomputes every EDA figure, retrains the regime classifier, and verifies it against
the notebook's stored metrics. If the retrain lands within ±0.5% of the notebook's
78.73% (it does — it reproduces it exactly), the dashboard shows the retrained
metrics; otherwise it falls back to the notebook's numbers and uses the retrain only
for feature importance. All other model-comparison numbers are taken verbatim from
`IDA_Lab.ipynb` outputs.

## Repository layout

```
├── IDA_Lab.ipynb                 # full analysis notebook (EDA + models)
├── Bitcoin Pulse ...csv          # hourly dataset (Apr 2023 – Apr 2025)
├── export_eda.py                 # regenerates frontend/data/*.js from the CSV
├── README.md
└── frontend/                     # the static site — open index.html directly
    ├── index.html
    ├── screenshot.png
    ├── css/styles.css
    ├── js/chart.umd.min.js       # Chart.js vendored — works offline
    ├── js/app.js                 # renders all 13 charts
    └── data/                     # window.OVERVIEW / SERIES / RETURNS / DRIVERS / MODELS
```
