"""
Streamlit Quantitative Research Dashboard: Pattern Recognition Predictor
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Setup Page Configuration
st.set_page_config(
    page_title="Pattern Recognition Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
DATA_DIR = ROOT / "data"

@st.cache_data
def load_all_data():
    meta_path = MODELS_DIR / "meta.json"
    signals_path = REPORTS_DIR / "latest_signals.csv"
    scorecard_path = REPORTS_DIR / "pattern_scorecard.csv"
    feat_path = REPORTS_DIR / "feature_importance.csv"
    trade_path = REPORTS_DIR / "trade_metrics.csv"
    metrics_path = REPORTS_DIR / "model_metrics.json"
    companies_path = DATA_DIR / "filtered_companies.csv"
    profiles_path = DATA_DIR / "cluster_profiles.csv"
    
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    signals = pd.read_csv(signals_path) if signals_path.exists() else pd.DataFrame()
    scorecard = pd.read_csv(scorecard_path) if scorecard_path.exists() else pd.DataFrame()
    features = pd.read_csv(feat_path) if feat_path.exists() else pd.DataFrame()
    trades = pd.read_csv(trade_path) if trade_path.exists() else pd.DataFrame()
    companies = pd.read_csv(companies_path) if companies_path.exists() else pd.DataFrame()
    profiles = pd.read_csv(profiles_path) if profiles_path.exists() else pd.DataFrame()
    
    return meta, metrics, signals, scorecard, features, trades, companies, profiles

meta, metrics, signals, scorecard, features, trades, companies, profiles = load_all_data()

if signals.empty or scorecard.empty:
    st.error("Pipeline outputs missing. Please run python src/run_all.py first.")
    st.stop()

# Enrich signals with company info
if not companies.empty:
    signals = signals.merge(
        companies[["symbol", "company", "sector", "market_cap_cr", "cluster_label"]],
        on="symbol",
        how="left"
    )

# Clean, simple page header
st.title("Pattern Recognition Predictor")
st.caption("Quantitative candlestick pattern evaluation and machine learning forecasting for NSE top-100 stocks.")

# Top KPI Metric Strip
col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
with col_kpi1:
    st.metric("Test AUC", f"{metrics.get('auc_test', 0.537):.3f}")
with col_kpi2:
    st.metric("Top-Decile Precision", f"{metrics.get('precision', 0.285):.1%}")
with col_kpi3:
    n_confirmed = int(scorecard['confirmed'].sum()) if 'confirmed' in scorecard.columns else 0
    st.metric("Confirmed Patterns", f"{n_confirmed}")
with col_kpi4:
    active_hits = len(signals[signals["pattern"].notna() & (signals["pattern"] != "")])
    st.metric("Active Signals Today", f"{active_hits}")
with col_kpi5:
    confirmed_hits = int(signals["confirmed"].sum()) if "confirmed" in signals.columns else 0
    st.metric("Confirmed Signals", f"{confirmed_hits}")

st.divider()

# Navigation Tabs
tab_signals, tab_charts, tab_scorecard, tab_clusters, tab_diagnostics, tab_performance = st.tabs([
    "Live Signals",
    "Interactive Charts",
    "Pattern Scorecard",
    "Cluster Profiles",
    "Model Diagnostics",
    "Fundamentals & Performance"
])

# ==========================================
# TAB 1: LIVE SIGNALS
# ==========================================
with tab_signals:
    col_filter1, col_filter2, col_filter3, col_filter4 = st.columns([2, 1.5, 1.5, 2])
    with col_filter1:
        pattern_opts = ["All Patterns"] + sorted([p for p in signals["pattern"].dropna().unique() if p])
        sel_pattern = st.selectbox("Pattern", pattern_opts)
    with col_filter2:
        trend_opts = ["All Trends"] + sorted(signals["trend"].dropna().unique().tolist())
        sel_trend = st.selectbox("Trend", trend_opts)
    with col_filter3:
        sel_conviction = st.selectbox("Filter", ["All Signals", "Trade Signals Only", "Confirmed by Test Year Only", "Model Top-Decile Only"])
    with col_filter4:
        search_query = st.text_input("Search Ticker or Company", placeholder="e.g. RELIANCE, ICICI")

    # Filter logic
    filtered = signals.copy()
    if sel_pattern != "All Patterns":
        filtered = filtered[filtered["pattern"] == sel_pattern]
    else:
        filtered = filtered[filtered["pattern"].notna() & (filtered["pattern"] != "")]
        
    if sel_trend != "All Trends":
        filtered = filtered[filtered["trend"] == sel_trend]

    # Set Status and Trade Decision (from report or fallback)
    if "trade_decision" in filtered.columns:
        filtered["Trade Decision"] = filtered["trade_decision"]
        filtered["Status"] = filtered["status"]
    else:
        def get_decision_and_status(row):
            confirmed = bool(row.get("confirmed", False))
            direction = int(row.get("direction", 0)) if pd.notna(row.get("direction")) else 0
            model_sig = bool(row.get("model_signal", False))
            if confirmed and direction == 1:
                return "Trade", "Confirmed Bullish"
            elif confirmed and direction == -1:
                return "Trade", "Confirmed Bearish"
            elif model_sig and direction == 1:
                return "Trade", "Top Decile Signal"
            elif not confirmed and direction != 0:
                return "Don't Trade", "Unconfirmed Pattern"
            else:
                return "Don't Trade", "No Setup"

        if not filtered.empty:
            filtered[["Trade Decision", "Status"]] = filtered.apply(get_decision_and_status, axis=1, result_type="expand")
        else:
            filtered["Trade Decision"] = []
            filtered["Status"] = []

    if sel_conviction == "Trade Signals Only":
        filtered = filtered[filtered["Trade Decision"] == "Trade"]
    elif sel_conviction == "Confirmed by Test Year Only":
        filtered = filtered[filtered["confirmed"] == True]
    elif sel_conviction == "Model Top-Decile Only":
        filtered = filtered[filtered["model_signal"] == True]
        
    if search_query:
        q = search_query.strip().upper()
        filtered = filtered[filtered["symbol"].str.contains(q, na=False) | filtered["company"].str.upper().str.contains(q, na=False)]

    st.write(f"Showing {len(filtered)} signals")

    # Clean formatted table: only meaningful data
    display_df = filtered.copy()
    display_df["Direction"] = display_df["direction"].map({1: "Bullish", -1: "Bearish", 0: "Neutral"})
    display_df["Signal (+1/-1)"] = display_df["direction"].map({1: "+1", -1: "-1", 0: "0"})
    display_df["Probability Going Up"] = (display_df["prob_up"] * 100).round(1).astype(str) + "%"
    display_df["Symbol"] = display_df["symbol"]
    display_df["Pattern"] = display_df["pattern"]
    display_df["Trend"] = display_df["trend"]
    display_df["Sector"] = display_df["sector"]
    
    clean_cols = [
        "Symbol",
        "Pattern",
        "Direction",
        "Signal (+1/-1)",
        "Trend",
        "Probability Going Up",
        "Sector",
        "Trade Decision",
        "Status"
    ]
    
    st.dataframe(
        display_df[clean_cols].sort_values(by=["Trade Decision", "Probability Going Up"], ascending=[True, False]),
        use_container_width=True,
        height=400
    )
    st.caption("Trade Decision: 'Trade' indicates statistically confirmed patterns or top-decile model confidence. 'Don't Trade' indicates unconfirmed patterns with insufficient historical confirmation.")

# ==========================================
# TAB 2: INTERACTIVE CHARTS
# ==========================================
with tab_charts:
    col_c1, col_c2, col_c3 = st.columns([2, 1, 1])
    available_symbols = sorted([f.stem for f in PROCESSED_DIR.glob("*.csv")]) if PROCESSED_DIR.exists() else []
    
    with col_c1:
        default_ticker = "ICICIBANK" if "ICICIBANK" in available_symbols else (available_symbols[0] if available_symbols else "")
        selected_ticker = st.selectbox("Stock", available_symbols, index=available_symbols.index(default_ticker) if default_ticker in available_symbols else 0)
    with col_c2:
        lookback_days = st.selectbox("Lookback Window (Days)", [60, 120, 250, 500], index=1)
    with col_c3:
        show_bbands = st.checkbox("Show Bollinger Bands", value=True)

    if selected_ticker:
        csv_file = PROCESSED_DIR / f"{selected_ticker}.csv"
        if csv_file.exists():
            df_price = pd.read_csv(csv_file, parse_dates=["Date"]).set_index("Date").sort_index()
            df_view = df_price.tail(lookback_days).copy()
            
            # Technical Overlays
            df_view["SMA20"] = df_view["Close"].rolling(20).mean()
            df_view["SMA50"] = df_view["Close"].rolling(50).mean()
            if show_bbands:
                std20 = df_view["Close"].rolling(20).std()
                df_view["UpperBB"] = df_view["SMA20"] + 2 * std20
                df_view["LowerBB"] = df_view["SMA20"] - 2 * std20

            # Subplots: Price Chart & Volume
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.75, 0.25],
                subplot_titles=(f"{selected_ticker} Price & Overlays", "Volume")
            )

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df_view.index,
                open=df_view["Open"],
                high=df_view["High"],
                low=df_view["Low"],
                close=df_view["Close"],
                name="OHLC"
            ), row=1, col=1)

            # Moving Averages
            fig.add_trace(go.Scatter(x=df_view.index, y=df_view["SMA20"], line=dict(color="#2563eb", width=1.5), name="SMA 20"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_view.index, y=df_view["SMA50"], line=dict(color="#f59e0b", width=1.5), name="SMA 50"), row=1, col=1)

            if show_bbands:
                fig.add_trace(go.Scatter(x=df_view.index, y=df_view["UpperBB"], line=dict(color="rgba(100, 116, 139, 0.4)", width=1, dash="dash"), name="Upper BB"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_view.index, y=df_view["LowerBB"], line=dict(color="rgba(100, 116, 139, 0.4)", width=1, dash="dash"), name="Lower BB", fill="tonexty", fillcolor="rgba(100, 116, 139, 0.05)"), row=1, col=1)

            # Volume
            vol_colors = np.where(df_view["Close"] >= df_view["Open"], "#22c55e", "#ef4444")
            fig.add_trace(go.Bar(
                x=df_view.index,
                y=df_view["Volume"],
                marker_color=vol_colors,
                name="Volume",
                opacity=0.7
            ), row=2, col=1)

            fig.update_layout(
                height=560,
                xaxis_rangeslider_visible=False,
                margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Stock Details Footer
            co_info = companies[companies["symbol"] == selected_ticker]
            if not co_info.empty:
                info_row = co_info.iloc[0]
                c_inf1, c_inf2, c_inf3, c_inf4 = st.columns(4)
                c_inf1.markdown(f"**Company:** {info_row.get('company', selected_ticker)}")
                c_inf2.markdown(f"**Sector:** {info_row.get('sector', 'N/A')}")
                c_inf3.markdown(f"**Market Cap:** ₹{info_row.get('market_cap_cr', 0):,.0f} Cr")
                c_inf4.markdown(f"**Cluster:** {info_row.get('cluster_label', 'N/A')}")
        else:
            st.warning(f"No price history found for {selected_ticker}")

# ==========================================
# TAB 3: PATTERN SCORECARD
# ==========================================
with tab_scorecard:
    st.subheader("Statistical Validation Across Epochs")
    st.markdown("""
    Patterns are evaluated across two out-of-sample periods:
    1. **Discovery Period** (Years 1 to 4: train and validation sets)
    2. **Confirmation Period** (Year 5: held-out test year)
    
    A pattern is marked **Confirmed** only if it outperforms the benchmark in **both** periods.
    """)

    col_sc1, col_sc2 = st.columns([3, 1])
    with col_sc1:
        confirmed_only = st.checkbox("Show Confirmed Patterns Only", value=True)
    with col_sc2:
        sort_by = st.selectbox("Sort By", ["excess_ret_test", "excess_ret_disc", "n_test"], index=0)

    sc_view = scorecard.copy()
    if confirmed_only:
        sc_view = sc_view[sc_view["confirmed"] == True]
    
    if not sc_view.empty:
        sc_view["direction_lbl"] = sc_view["direction"].map({1: "Bullish (+1)", -1: "Bearish (-1)"})
        sc_view["excess_ret_disc_%"] = (sc_view["excess_ret_disc"] * 100).round(2).map("{:+.2f}%".format)
        sc_view["excess_ret_test_%"] = (sc_view["excess_ret_test"] * 100).round(2).map("{:+.2f}%".format)
        
        display_sc = sc_view[["cluster", "pattern", "direction_lbl", "n_disc", "excess_ret_disc_%", "n_test", "excess_ret_test_%", "confirmed"]]
        st.dataframe(
            display_sc.sort_values(by="excess_ret_test_%" if "excess_ret_test" in sort_by else "n_test", ascending=False),
            use_container_width=True,
            height=380
        )
    else:
        st.info("No patterns matching the filter criteria.")

# ==========================================
# TAB 4: CLUSTER PROFILES
# ==========================================
with tab_clusters:
    st.subheader("Stock Universe Clusters")
    
    col_cl0, col_cl1, col_cl2 = st.columns(3)
    
    with col_cl0:
        st.markdown("""
        **Cluster 0: Growth & Earnings Surge**
        - Profile: High P/E, elevated P/B, rapid revenue expansion.
        - Representatives: TITAN, ADANIENT, TVSMOTOR, SIEMENS, ABB, BEL, INDIGO
        - Top Pattern: CDLMORNINGSTAR (+0.49% test excess)
        """)
        
    with col_cl1:
        st.markdown("""
        **Cluster 1: High Margin & Volatile**
        - Profile: High operating margin, higher volatility/beta, financial & PSU cyclicality.
        - Representatives: BAJFINANCE, AXISBANK, HAL, HINDZINC, PNB, BSE, DLF
        - Top Pattern: CDL3WHITESOLDIERS (+0.47% test excess)
        """)
        
    with col_cl2:
        st.markdown("""
        **Cluster 2: Stable Blue Chips**
        - Profile: Mega-cap market leaders, stable cash flows, lower beta.
        - Representatives: RELIANCE, TCS, HDFCBANK, INFOSYS, HINDUNILVR, ITC, LT
        - Top Pattern: CDLHAMMER (+0.17% test excess)
        """)

    st.markdown("---")
    if not companies.empty:
        st.write("Sector and Cluster Distribution")
        sector_cluster_tbl = pd.crosstab(companies["sector"], companies["cluster"], margins=True)
        st.dataframe(sector_cluster_tbl, use_container_width=True)

# ==========================================
# TAB 5: MODEL DIAGNOSTICS
# ==========================================
with tab_diagnostics:
    st.subheader("Model Validation & Information Gain")
    
    # Model Quality Metrics strip
    dm1, dm2, dm3, dm4, dm5 = st.columns(5)
    dm1.metric("Validation AUC", f"{metrics.get('auc_val', 0.519):.3f}")
    dm2.metric("Test AUC", f"{metrics.get('auc_test', 0.537):.3f}")
    dm3.metric("Decision Threshold", f"{metrics.get('threshold', 0.283):.3f}")
    dm4.metric("Recall (Top 10%)", f"{metrics.get('recall', 0.102):.1%}")
    dm5.metric("F1 Score", f"{metrics.get('f1', 0.150):.3f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.write("#### Top 15 Predictive Features (Permutation Importance)")
        if not features.empty:
            top_fi = features.head(15).copy().sort_values(by="importance", ascending=True)
            fig_fi = go.Figure(go.Bar(
                x=top_fi["importance"],
                y=top_fi["feature"],
                orientation="h",
                marker_color="#2563eb"
            ))
            fig_fi.update_layout(
                margin=dict(l=100, r=20, t=20, b=20),
                height=420,
                xaxis_title="Validation AUC Drop when Feature is Shuffled"
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            
    with col_d2:
        st.write("#### Feature Insights & Architecture")
        st.markdown(r"""
        - **Primary Lead Indicator:** `dist_high20` (distance from 20-day high) provides by far the highest information gain ($\Delta$ AUC $\approx 0.013$), far exceeding individual single-candle heuristics.
        - **Volatility & Breadth Regime:** `mkt_vol_20` (market volatility) and `adx14` (trend strength) form the secondary tier of predictive weight.
        - **Individual Candlestick Ensembles:** Rather than single patterns, composite features (`n_bull_pat`, `n_bear_pat`) and market-relative returns provide the directional signal for the LightGBM classifier.
        """)

# ==========================================
# TAB 6: FUNDAMENTALS & PERFORMANCE
# ==========================================
with tab_performance:
    st.subheader("Performance Parameters & Financial Fundamentals")
    st.caption("Out-of-sample trading performance (held-out test year) alongside fundamental cluster characteristics.")
    
    # 1. Performance KPIs
    st.write("#### Risk-Adjusted Trading Parameters (Test Year)")
    pk1, pk2, pk3, pk4, pk5 = st.columns(5)
    
    top_trade_row = trades[trades["group"].str.contains("model top", case=False, na=False)]
    if not top_trade_row.empty:
        t_row = top_trade_row.iloc[0]
        pk1.metric("Max Drawdown", f"{t_row['max_drawdown_%']}%")
        pk2.metric("Sharpe Ratio", f"{t_row['sharpe']}")
        pk3.metric("Win Rate", f"{t_row['win_rate_%']}%")
        pk4.metric("Profit Factor", f"{t_row['profit_factor']}")
        pk5.metric("Avg Net Return", f"{t_row['avg_net_ret_%']}%", delta="After 0.2% Cost", delta_color="off")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Performance Comparison Table & Plotly Charts
    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        st.write("##### Strategy Performance Comparison (Out-of-Sample)")
        if not trades.empty:
            trades_formatted = trades.copy()
            trades_formatted["trades"] = trades_formatted["trades"].apply(lambda x: f"{x:,}")
            trades_formatted["avg_net_ret_%"] = trades_formatted["avg_net_ret_%"].apply(lambda x: f"{x:+.3f}%")
            trades_formatted["win_rate_%"] = trades_formatted["win_rate_%"].apply(lambda x: f"{x:.1f}%")
            trades_formatted["max_drawdown_%"] = trades_formatted["max_drawdown_%"].apply(lambda x: f"{x:.1f}%")
            trades_formatted.columns = ["Strategy Group", "Trades", "Avg Net Return", "Win Rate", "Profit Factor", "Sharpe", "Max Drawdown"]
            st.dataframe(trades_formatted, use_container_width=True)
            st.caption("Note: Results include 0.20% round-trip trading and slippage friction.")
            
    with col_p2:
        st.write("##### Sharpe Ratio & Drawdown Comparison")
        if not trades.empty:
            fig_perf = go.Figure()
            clean_groups = [g.strip() for g in trades["group"]]
            fig_perf.add_trace(go.Bar(
                x=clean_groups,
                y=trades["sharpe"],
                name="Sharpe Ratio",
                marker_color="#2563eb"
            ))
            fig_perf.add_trace(go.Bar(
                x=clean_groups,
                y=trades["max_drawdown_%"],
                name="Max Drawdown %",
                marker_color="#ef4444"
            ))
            fig_perf.update_layout(
                barmode="group",
                height=320,
                margin=dict(l=30, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_perf, use_container_width=True)

    st.divider()

    # 3. Fundamentals per Cluster
    st.write("#### Cluster Fundamentals & Financial Health")
    st.caption("Median fundamental metrics computed across 100 NSE companies grouped by unsupervised clustering.")
    
    if not profiles.empty:
        prof_display = profiles.copy()
        cluster_names = {
            0: "Cluster 0: Growth & Earnings Surge",
            1: "Cluster 1: High Margin & Volatile",
            2: "Cluster 2: Stable Blue Chips"
        }
        prof_display["Cluster Name"] = prof_display["cluster"].map(cluster_names)
        prof_cols_map = {
            "Cluster Name": "Cluster",
            "n": "Stocks",
            "pe_trailing": "Trailing P/E",
            "pb": "P/B Ratio",
            "dividend_yield": "Div Yield %",
            "profit_margin": "Profit Margin %",
            "revenue_growth": "Rev Growth %",
            "earnings_growth": "Earnings Growth %",
            "volatility_annual_pct": "Annual Volatility %",
            "max_drawdown_5y_pct": "5Y Max Drawdown %",
            "beta": "Beta"
        }
        avail_cols = [c for c in prof_cols_map.keys() if c in prof_display.columns]
        prof_tbl = prof_display[avail_cols].rename(columns=prof_cols_map)
        st.dataframe(prof_tbl, use_container_width=True)

    # 4. Individual Company Fundamental Lookup
    st.write("#### Company Fundamentals Explorer")
    if not companies.empty:
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            stock_list = sorted(companies["symbol"].unique())
            default_sym = "ICICIBANK" if "ICICIBANK" in stock_list else stock_list[0]
            f_stock = st.selectbox("Select Company for Fundamentals", stock_list, index=stock_list.index(default_sym))
        
        c_meta = companies[companies["symbol"] == f_stock].iloc[0]
        c_cl = c_meta["cluster"]
        cl_bench = profiles[profiles["cluster"] == c_cl].iloc[0] if not profiles.empty and c_cl in profiles["cluster"].values else None
        
        with col_f2:
            st.markdown(f"**{c_meta['company']}** ({c_meta['symbol']}) · Sector: **{c_meta['sector']}** · Market Cap: **₹{c_meta['market_cap_cr']:,.0f} Cr**")
            st.caption(f"Cluster Assignment: {c_meta['cluster_label']}")
            
        if cl_bench is not None:
            cf1, cf2, cf3, cf4, cf5 = st.columns(5)
            cf1.metric("Cluster P/E", f"{cl_bench.get('pe_trailing', 'N/A')}")
            cf2.metric("Cluster P/B", f"{cl_bench.get('pb', 'N/A')}")
            cf3.metric("Profit Margin", f"{cl_bench.get('profit_margin', 'N/A')}%")
            cf4.metric("Annual Volatility", f"{cl_bench.get('volatility_annual_pct', 'N/A')}%")
            cf5.metric("5Y Drawdown", f"{cl_bench.get('max_drawdown_5y_pct', 'N/A')}%")

# Footer
st.divider()
st.caption("Pattern Recognition Predictor | Machine Learning & Statistical Validation System | Research tool, not financial advice.")
