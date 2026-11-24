# Pattern Recognition Predictor (NSE Top-100)

Quantitative candlestick pattern detection, statistical validation, and machine learning directional forecasting across the top 100 NSE companies.

Run locally (needs internet):
1. `pip install -r requirements.txt`  (TA-Lib optional; without it 12 core patterns are used instead of 61)
2. `python src/data_collection.py`    -> data/raw (5y daily, via yfinance)
3. `python src/run_all.py`            -> filtering, features, labels, train, evaluate, detect
4. `streamlit run app.py`

Split (5th year held out for testing, as agreed in the group chat):
train = years 1-3 (to 2024-10-31), validation = year 4 (to 2025-10-31), test = year 5. A 10-day embargo sits before each boundary.
SMA20/SMA50 + RSI feed in as `trend_state` and `rsi_zone`; the label is still the 5-day forward return vs an ATR band.

Deploy (free): push this folder (including models/ and reports/ after step 3) to a GitHub repo,
then share.streamlit.io -> New app -> pick the repo, main file `app.py`.
Refresh signals daily by re-running `data_collection.py --force`, `features.py`, `detect.py` and pushing reports/.
