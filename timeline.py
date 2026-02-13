# streamlit_crypto_dashboard_multi.py
"""
Multi-dashboard Crypto Analytics Streamlit App (single file)
Contains:
1. Forecast Analytics (ARIMA, SARIMA, Prophet, LSTM, Actual)
2. Technical Indicators
3. Volatility & Risk
4. Correlation Matrix
5. Candlestick & Volume
6. Returns & Drawdown
7. Sentiment Analysis (dummy)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import datetime as dt
import joblib
from pathlib import Path

import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Try importing ML libs gracefully
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    ARIMA = SARIMAX = None

try:
    from prophet import Prophet
except Exception:
    Prophet = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except Exception:
    tf = None

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# --------------------------
# Utilities & Data Fetching
# --------------------------
CACHE_DIR = Path("./model_cache")
CACHE_DIR.mkdir(exist_ok=True)

@st.cache_data(ttl=3600)
def fetch_data(ticker="BTC-USD", period="2y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=["ds","y"])
    df = df.reset_index()[["Date","Close"]]
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.dropna()
    return df

def ensure_index(df):
    df = df.copy()
    df = df.set_index("ds")
    df.index = pd.to_datetime(df.index)
    return df

# --------------------------
# Fallback dummy forecast
# --------------------------
def dummy_forecast(df, horizon=30, seed=42):
    """Simple extrapolation: linear fit + small noise"""
    np.random.seed(seed)
    # df could be either DataFrame with 'y' or Series
    if isinstance(df, pd.Series):
        y = df.values
        last_index = df.index
    else:
        y = df["y"].values
        last_index = pd.to_datetime(df.index)
    if len(y) < 2:
        preds = np.repeat(0.0, horizon)
    else:
        x = np.arange(len(y))
        coef = np.polyfit(x, y, 1)
        slope, intercept = coef[0], coef[1]
        future_x = np.arange(len(y), len(y) + horizon)
        preds = intercept + slope * future_x
    noise = np.random.normal(scale=np.std(y)*0.01 if len(y)>0 else 1.0, size=horizon)
    preds = preds + noise
    try:
        start = last_index[-1] + pd.Timedelta(days=1)
    except Exception:
        start = pd.Timestamp.now().normalize()
    future_index = pd.date_range(start=start, periods=horizon, freq='D')
    return pd.Series(preds, index=future_index, name="dummy")

# --------------------------
# ARIMA & SARIMA
# --------------------------
def train_arima(df, horizon=30, order=(5,1,0), cache_name="arima.pkl"):
    cache_path = CACHE_DIR / cache_name
    idx = df.index
    try:
        if cache_path.exists():
            fitted = joblib.load(cache_path)
        else:
            if ARIMA is None:
                raise RuntimeError("statsmodels ARIMA not available")
            model = ARIMA(df["y"], order=order)
            fitted = model.fit()
            joblib.dump(fitted, cache_path)
        preds = fitted.forecast(steps=horizon)
        future_idx = pd.date_range(start=idx[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
        return pd.Series(preds, index=future_idx, name="ARIMA_Forecast")
    except Exception:
        return dummy_forecast(df, horizon)

def train_sarima(df, horizon=30, order=(1,1,1), seasonal_order=(1,1,1,12), cache_name="sarima.pkl"):
    cache_path = CACHE_DIR / cache_name
    idx = df.index
    try:
        if cache_path.exists():
            fitted = joblib.load(cache_path)
        else:
            if SARIMAX is None:
                raise RuntimeError("statsmodels SARIMAX not available")
            model = SARIMAX(df["y"], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit(disp=False)
            joblib.dump(fitted, cache_path)
        preds = fitted.get_forecast(steps=horizon).predicted_mean
        future_idx = pd.date_range(start=idx[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
        return pd.Series(preds, index=future_idx, name="SARIMA_Forecast")
    except Exception:
        return dummy_forecast(df, horizon)

# --------------------------
# Prophet
# --------------------------
def train_prophet(df, horizon=30, cache_name="prophet.pkl"):
    cache_path = CACHE_DIR / cache_name
    try:
        if Prophet is None:
            raise RuntimeError("Prophet not installed")
        if cache_path.exists():
            m = joblib.load(cache_path)
        else:
            m = Prophet(daily_seasonality=True)
            m.fit(df.rename(columns={"ds":"ds","y":"y"}))
            joblib.dump(m, cache_path)
        future = m.make_future_dataframe(periods=horizon)
        forecast = m.predict(future)
        preds = forecast.set_index("ds")["yhat"].iloc[-horizon:]
        preds.index = pd.to_datetime(preds.index)
        preds.name = "Prophet_Forecast"
        return preds
    except Exception:
        # If df is DataFrame with ds index
        try:
            return dummy_forecast(df.set_index("ds"), horizon)
        except Exception:
            return dummy_forecast(pd.Series(dtype=float), horizon)

# --------------------------
# LSTM
# --------------------------
def prepare_lstm_series(series, n_steps=30):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps])
    X = np.array(X)
    y = np.array(y)
    if X.size == 0:
        return np.array([]), np.array([])
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def train_lstm(df, horizon=30, n_steps=30, epochs=10, batch_size=16, cache_name="lstm.pkl"):
    cache_path = CACHE_DIR / cache_name
    idx = pd.to_datetime(df.index)
    try:
        if tf is None:
            raise RuntimeError("TensorFlow not installed")
        scaler = MinMaxScaler()
        arr = df["y"].values.reshape(-1,1)
        scaled = scaler.fit_transform(arr).flatten()
        # prepare
        X, y = prepare_lstm_series(scaled, n_steps=n_steps)
        if X.shape[0] < 10:
            raise RuntimeError("Not enough data for LSTM")
        if cache_path.exists():
            model, scaler = joblib.load(cache_path)
        else:
            model = Sequential([
                LSTM(64, input_shape=(n_steps,1)),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
            joblib.dump((model, scaler), cache_path)
        # forecasting iteratively
        last_seq = scaled[-n_steps:].tolist()
        preds_scaled = []
        for _ in range(horizon):
            x_in = np.array(last_seq[-n_steps:]).reshape((1,n_steps,1))
            pred = model.predict(x_in, verbose=0)[0][0]
            preds_scaled.append(pred)
            last_seq.append(pred)
        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
        future_idx = pd.date_range(start=idx[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
        return pd.Series(preds, index=future_idx, name="LSTM_Forecast")
    except Exception:
        return dummy_forecast(df, horizon)

# --------------------------
# Orchestrator
# --------------------------
def generate_all_forecasts(df, horizon=30, selected_models=None, retrain=False):
    """
    df: dataframe with columns ['ds','y']
    returns dict of pd.Series {model_name: series}
    """
    df_indexed = df.copy().set_index("ds")
    results = {}
    results["Actual_Close"] = df_indexed["y"].copy()
    if selected_models is None:
        selected_models = ["ARIMA_Forecast","SARIMA_Forecast","Prophet_Forecast","LSTM_Forecast"]
    # optionally clear cache for retrain
    if retrain:
        for f in CACHE_DIR.glob("*.pkl"):
            try:
                f.unlink()
            except:
                pass

    # compute each selected
    if "ARIMA_Forecast" in selected_models:
        results["ARIMA_Forecast"] = train_arima(df_indexed, horizon=horizon)
    if "SARIMA_Forecast" in selected_models:
        results["SARIMA_Forecast"] = train_sarima(df_indexed, horizon=horizon)
    if "Prophet_Forecast" in selected_models:
        df_prophet = df[["ds","y"]]
        results["Prophet_Forecast"] = train_prophet(df_prophet, horizon=horizon)
    if "LSTM_Forecast" in selected_models:
        results["LSTM_Forecast"] = train_lstm(df_indexed, horizon=horizon)
    # ensure all are pandas Series
    for k,v in list(results.items()):
        if not isinstance(v, pd.Series):
            results[k] = pd.Series(v)
    return results

# --------------------------
# ----------------- Page Config -----------------
st.set_page_config(page_title="Cryptocurrency Dashboard", layout="wide")

# ----------------- Scrolling Title -----------------
st.markdown("""
<div style="background-color:black; padding:10px;">
<marquee behavior="scroll" direction="left" scrollamount="10" style="color:green; font-size:30px; font-weight:bold;">
Cryptocurrency Dashboard 
</marquee>
</div>
""", unsafe_allow_html=True)

# ----------------- Dark Theme -----------------
st.markdown("""
<style>
.stApp {background-color:#0b0f14; color:red;}
</style>
""", unsafe_allow_html=True)

# ----------------- Layout Helper -----------------
def dark_layout(fig, title=None):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0b0f14",
        paper_bgcolor="#0b0f14",
        font=dict(color="#cbd5e1"),
        margin=dict(l=40, r=20, t=40, b=40),
        title=title
    )
    return fig

# ----------------- Pages -----------------
pages = [
    "Forecast Analytics",
    "Technical Indicators",
    "Volatility & Risk",
    "Correlation Matrix",
    "Candlestick & Volume",
    "Returns & Drawdown",
    "Sentiment Analysis",
    "Market Dashboard"
]
page = st.selectbox("Select Dashboard:", pages)

# ----------------- User Inputs -----------------
ticker = st.text_input("Ticker (yfinance)", "BTC-USD")
period = st.selectbox("History period", ["6mo","1y","2y","5y"], 2)
horizon = st.number_input("Forecast horizon (days)", 7, 365, 90)

# ----------------- Fetch main data -----------------
@st.cache_data(ttl=3600)
def fetch_data(ticker="BTC-USD", period="2y"):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df.empty:
        return pd.DataFrame(columns=["ds","y"])
    df = df.reset_index()[["Date","Close"]]
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    return df

df = fetch_data(ticker, period)
if df.empty:
    st.error(f"No data found for {ticker}.")
    st.stop()


# ---------- PAGE: Forecast Analytics ----------
if page == "Forecast Analytics":
    st.markdown("<h2 style='color:#ffffff; font-weight:700;'>Forecast Analytics Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<small style='color:#aab4c0'>Select Forecast Models</small>", unsafe_allow_html=True)

    # Model multi-select (show tags like screenshot)
    model_options = ["ARIMA_Forecast", "SARIMA_Forecast", "Prophet_Forecast", "LSTM_Forecast"]
    selected = st.multiselect("Choose models to display", ["Actual_Close"] + model_options, default=["Actual_Close","LSTM_Forecast","ARIMA_Forecast"])

    st.metric(label="Last Updated", value=pd.Timestamp.now(tz=None).strftime("%Y-%m-%d %H:%M:%S"))

    # Generate forecasts
    with st.spinner("Training/Loading models and generating forecasts (some models may be cached)..."):
        forecasts = generate_all_forecasts(df, horizon=horizon, selected_models=selected if selected else None,)

    # Build Plotly figure
    fig = go.Figure()
    actual = forecasts.get("Actual_Close")
    if isinstance(actual, pd.Series):
        fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode='lines', name='Actual_Close', line=dict(width=2)))
    else:
        ts = df.set_index("ds")["y"]
        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='Actual_Close', line=dict(width=2)))

    for m in selected:
        if m == "Actual_Close":
            continue
        series = forecasts.get(m)
        if series is None:
            continue
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=m))

    dark_layout(fig, title="Forecast vs Actual")
    st.plotly_chart(fig, use_container_width=True)

    # Add a small summary / model diagnostics
    st.markdown("---")
    cols = st.columns(3)
    with cols[0]:
        st.write("Data points")
        st.write(len(df))
    with cols[1]:
        st.write("Forecast horizon (days)")
        st.write(horizon)
    with cols[2]:
        st.write("Models displayed")
        st.write(", ".join(selected))

    # Show sample forecast table
    st.markdown("### Forecast sample (next 10 days)")
    rows = []
    for m in selected:
        if m == "Actual_Close":
            continue
        s = forecasts.get(m)
        if s is None: continue
        sample = s.iloc[:10].rename(m)
        rows.append(sample)
    if rows:
        combined = pd.concat(rows, axis=1)
        st.dataframe(combined)

    st.markdown("<div style='color:#9aa7b2'>Tip: if a model fails to load/training fails or a package isn't installed, a robust dummy forecast is used so the dashboard stays interactive.</div>", unsafe_allow_html=True)

# ---------- PAGE: Technical Indicators ----------
elif page == "Technical Indicators":
    st.markdown("<h2 style='color:#ffffff; font-weight:700;'>Technical Indicators Dashboard</h2>", unsafe_allow_html=True)
    df2 = df.set_index("ds").copy()
    df2["SMA_20"] = df2["y"].rolling(window=20).mean()
    df2["SMA_50"] = df2["y"].rolling(window=50).mean()
    df2["EMA_20"] = df2["y"].ewm(span=20, adjust=False).mean()

    # RSI
    delta = df2["y"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.rolling(14).mean() / down.rolling(14).mean()
    df2["RSI"] = 100 - (100 / (1 + rs))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df2.index, y=df2["y"], name="Price"))
    fig.add_trace(go.Scatter(x=df2.index, y=df2["SMA_20"], name="SMA 20"))
    fig.add_trace(go.Scatter(x=df2.index, y=df2["SMA_50"], name="SMA 50"))
    fig.add_trace(go.Scatter(x=df2.index, y=df2["EMA_20"], name="EMA 20"))

    dark_layout(fig, title="Moving Averages")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("RSI (14-day)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df2.index, y=df2["RSI"], name="RSI"))
    fig2.update_layout(template="plotly_dark", yaxis=dict(range=[0,100]))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("Key indicator values (latest):")
    latest = df2.iloc[-1]
    st.write({
        "Price": float(latest["y"]),
        "SMA_20": float(latest["SMA_20"]) if pd.notna(latest["SMA_20"]) else None,
        "SMA_50": float(latest["SMA_50"]) if pd.notna(latest["SMA_50"]) else None,
        "RSI": float(latest["RSI"]) if pd.notna(latest["RSI"]) else None
    })

# ---------- PAGE: Volatility & Risk ----------
elif page == "Volatility & Risk":
    st.markdown("<h2 style='color:#ffffff; font-weight:700;'>Volatility & Risk Dashboard</h2>", unsafe_allow_html=True)
    df2 = df.set_index("ds").copy()
    df2["Returns"] = df2["y"].pct_change()
    df2["Volatility_30"] = df2["Returns"].rolling(window=30).std() * np.sqrt(30)

    st.subheader("Rolling Volatility (30 days)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df2.index, y=df2["Volatility_30"], name="Volatility"))
    dark_layout(fig, title="30-day Rolling Volatility")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Returns Histogram")
    returns = df2["Returns"].dropna()
    fig2 = px.histogram(returns, nbins=80, title="Daily Returns Distribution")
    dark_layout(fig2, title="Daily Returns Distribution")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("Volatility summary (latest):")
    st.write({
        "Latest daily return": float(returns.iloc[-1]) if len(returns)>0 else None,
        "30-day volatility": float(df2["Volatility_30"].iloc[-1]) if pd.notna(df2["Volatility_30"].iloc[-1]) else None
    })

# ---------- PAGE: Correlation Matrix ----------
elif page == "Correlation Matrix":
    st.markdown("<h2 style='color:#ffffff; font-weight:700;'>Correlation Matrix Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("You can add more tickers (comma separated) in the input below. Default set provided.")
    cryptos_input = st.text_input("Additional tickers (comma separated)", value="ETH-USD,BNB-USD,ADA-USD,BTC-USD", help="E.g. ETH-USD,BNB-USD,BTC-USD")
    cryptos = ["BTC-USD"] + [t.strip() for t in cryptos_input.split(",") if t.strip()]
    period_corr = st.selectbox("Correlation period", ["6mo","1y","2y"], index=1)

    # fetch closes for each
    df_corr = pd.DataFrame()
    with st.spinner("Fetching data for correlation..."):
        for c in cryptos:
            try:
                d = yf.download(c, period=period_corr, interval="1d", progress=False)
                if d is not None and not d.empty:
                    df_corr[c] = d["Close"]
            except Exception:
                pass

    if df_corr.empty:
        st.error("No data for correlation tickers.")
    else:
        corr = df_corr.pct_change().corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale=px.colors.diverging.RdBu,
            title="Correlation Heatmap (returns)"
        )
        dark_layout(fig, title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        st.write("Correlation table:")
        st.dataframe(corr)

# ---------- PAGE: Candlestick & Volume ----------
elif page == "Candlestick & Volume":
    st.markdown("<h2 style='color:#ffffff; font-weight:700;'>Candlestick & Volume</h2>", unsafe_allow_html=True)

    # FIX TICKER
    if ticker.isalpha():
        ticker = ticker.upper() + "-USD"

    # DOWNLOAD DATA
    df_price = yf.download(ticker, period=period, interval="1d")

    # DEBUG
    st.write("Columns returned by Yahoo:", df_price.columns)
    st.write(df_price.head())

    if df_price.empty:
        st.error("No data returned from Yahoo Finance.")
        st.stop()

    # FLATTEN MultiIndex if needed
    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = [' '.join(col).strip() for col in df_price.columns.values]

    # STANDARDIZE COLUMN NAMES (lowercase)
    df_price.columns = [col.lower().strip() for col in df_price.columns]

    # MAP Yahoo columns to OHLCV
    col_map = {}
    for key in ["open", "high", "low", "close", "volume"]:
        for col in df_price.columns:
            if key in col:  # fuzzy match
                col_map[key] = col
                break

    # CHECK IF ALL REQUIRED COLUMNS EXIST
    if len(col_map) < 5:
        st.error(f"Could not find all required columns in Yahoo data. Found: {list(col_map.values())}")
        st.stop()

    # CLEAN DATA
    df_price = df_price[list(col_map.values())].dropna()
    df_price.rename(columns={v: k.capitalize() for k, v in col_map.items()}, inplace=True)
    df_price.reset_index(inplace=True)

    # PLOT CANDLESTICK
    fig = go.Figure(data=[
        go.Candlestick(
            x=df_price["Date"],
            open=df_price["Open"],
            high=df_price["High"],
            low=df_price["Low"],
            close=df_price["Close"],
            increasing_line_color="green",
            decreasing_line_color="red"
        )
    ])

    fig.update_layout(
        title=f"{ticker} Candlestick Chart",
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # VOLUME
    st.subheader("Volume")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df_price["Date"], y=df_price["Volume"]))
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)



# ---------- PAGE: Returns & Drawdown ---------- 
elif page == "Returns & Drawdown":
    st.markdown("<h2 style='color:#ffffff; font-weight:700;'>Returns, Drawdown & Sharpe Dashboard</h2>", unsafe_allow_html=True)
    df2 = df.set_index("ds").copy()
    df2["Returns"] = df2["y"].pct_change().fillna(0)
    df2["Cumulative"] = (1 + df2["Returns"]).cumprod()
    df2["Peak"] = df2["Cumulative"].cummax()
    df2["Drawdown"] = df2["Cumulative"] / df2["Peak"] - 1

    st.subheader("Cumulative Returns")
    st.line_chart(df2["Cumulative"])

    st.subheader("Drawdown")
    st.area_chart(df2["Drawdown"])

    # Sharpe ratio (annualized)
    try:
        daily_mean = df2["Returns"].mean()
        daily_std = df2["Returns"].std()
        sharpe = (daily_mean / daily_std) * np.sqrt(252) if daily_std != 0 else np.nan
    except Exception:
        sharpe = np.nan
    st.metric("Sharpe Ratio (annualized)", f"{sharpe:.2f}")

    st.markdown("Performance summary:")
    st.write({
        "Total return": float(df2["Cumulative"].iloc[-1]) - 1,
        "Max drawdown": float(df2["Drawdown"].min())
    })

# ---------- PAGE: Sentiment Analysis ----------
elif page == "Sentiment Analysis":
    import numpy as np
    import plotly.graph_objects as go

    st.markdown(
        "<h1 style='text-align:center; color:#ffffff;'>Sentiment Analysis Dashboard</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align:center; color:#d0d0d0;'>A visual breakdown of market sentiment using bull/bear icons, progress bars, and a sentiment gauge.</p>",
        unsafe_allow_html=True,
    )

    # Fake data (replace with API)
    bullish = 0.65
    neutral = 0.20
    bearish = 0.15

    sentiment_score = np.round((bullish - bearish), 2)
    sentiment_label = (
        "Bullish" if sentiment_score > 0.2
        else "Bearish" if sentiment_score < -0.2
        else "Neutral"
    )

    # Icons
    BULL_ICON = "https://cdn-icons-png.flaticon.com/512/7465/7465709.png"
    NEUTRAL_ICON = "https://cdn-icons-png.flaticon.com/512/179/179386.png"
    BEAR_ICON = "https://cdn-icons-png.flaticon.com/512/7465/7465715.png"

    col1, col2 = st.columns([1.2, 1])

    # --------------------------- LEFT ---------------------------
    with col1:
        st.markdown("<h2 style='color:#ffffff;'>Sentiment Breakdown</h2>", unsafe_allow_html=True)

        def sentiment_row(icon_url, label, value, bar_color):
            c1, c2, c3 = st.columns([0.2, 1, 0.2])
            with c1:
                st.image(icon_url, width=45)
            with c2:
                st.markdown(
                    f"<p style='color:#ffffff; font-size:20px;'>{label}</p>",
                    unsafe_allow_html=True,
                )
                st.progress(value)
            with c3:
                st.markdown(
                    f"<p style='color:#ffffff; font-size:20px; margin-top:20px;'>{int(value*100)}%</p>",
                    unsafe_allow_html=True,
                )

        sentiment_row(BULL_ICON, "Bullish", bullish, "#2ecc71")
        sentiment_row(NEUTRAL_ICON, "Neutral", neutral, "#95a5a6")
        sentiment_row(BEAR_ICON, "Bearish", bearish, "#c0392b")

    # --------------------------- RIGHT ---------------------------
    with col2:
        st.markdown("<h2 style='color:#ffffff;'>Sentiment Gauge</h2>", unsafe_allow_html=True)

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=sentiment_score,
                number={'font': {'color': 'white', 'size': 48}},
                gauge={
                    "axis": {"range": [-1, 1], "tickwidth": 0},
                    "bar": {"color": "rgba(0,0,0,0)"},
                    "steps": [
                        {"range": [-1, -0.3], "color": "#ff6b5c"},
                        {"range": [-0.3, 0.3], "color": "#f4d03f"},
                        {"range": [0.3, 1], "color": "#2ecc71"}
                    ],
                    "threshold": {"line": {"color": "white", "width": 4}, "value": sentiment_score},
                },
            )
        )

        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="#062b36",
            font=dict(color="white")
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<h3 style='color:#ffffff;'>Market Insight</h3>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style='color:#d0d0d0; font-size:18px;'>
                <strong>Current Sentiment:</strong> {sentiment_label}<br>
                The market currently shows <strong>{sentiment_label.lower()}</strong> tendencies.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------- PAGE: Market Dashboard ----------
elif page == "Market Dashboard":
    st.markdown("<h2 style='color:#ffffff; font-weight:700;'>Market Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#cccccc;'>Executive overview: price, volume, short-term momentum, and quick sentiment snapshot.</p>", unsafe_allow_html=True)

    # Data setup
    df_idx = df.set_index("ds").copy()
    latest_ts = df_idx.index.max()
    latest_price = float(df_idx["y"].iloc[-1])
    prev_price = float(df_idx["y"].iloc[-2]) if len(df_idx) > 1 else latest_price
    pct_1d = (latest_price / prev_price - 1) * 100 if prev_price != 0 else 0.0

    # 7-day change
    try:
        price_7d_ago = float(df_idx["y"].iloc[-8])
        pct_7d = (latest_price / price_7d_ago - 1) * 100
    except Exception:
        pct_7d = np.nan

    # Volume metrics (use raw yfinance fetch for volume series if available)
    vol_series = None
    try:
        raw = yf.download(ticker, period=period, interval="1d", progress=False)
        if not raw.empty and "Volume" in raw.columns:
            vol_series = raw["Volume"].dropna()
    except Exception:
        vol_series = None

    last_vol = int(vol_series.iloc[-1]) if vol_series is not None and len(vol_series) > 0 else None
    avg_vol_30 = int(vol_series.tail(30).mean()) if vol_series is not None and len(vol_series) >= 5 else None

    # Try to get some fundamentals (may not be complete for crypto tickers)
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info
    except Exception:
        info = {}

    market_cap = info.get("marketCap") or info.get("market_cap") or None
    circulating_supply = info.get("circulatingSupply") or info.get("circulating_supply") or None

    # Top KPI row
    k1, k2, k3, k4 = st.columns([2,2,2,2])
    k1.metric(label=f"{ticker} Price", value=f"${latest_price:,.2f}", delta=f"{pct_1d:.2f}%")
    k2.metric(label="7d Change", value=f"{pct_7d:.2f}%" if not np.isnan(pct_7d) else "n/a")
    k3.metric(label="24h Volume", value=f"{last_vol:,}" if last_vol is not None else "n/a")
    k4.metric(label="30d Avg Vol", value=f"{avg_vol_30:,}" if avg_vol_30 is not None else "n/a")

    # Middle area: price + volume chart and right column for sentiment & fundamentals
    left_col, right_col = st.columns([3,1], gap="large")

    # Price + Volume combined chart
    with left_col:
        fig = go.Figure()
        # Price line
        fig.add_trace(go.Scatter(
            x=df_idx.index, y=df_idx["y"], name="Price", mode="lines", line=dict(width=2)
        ))

        # add latest-day marker
        fig.add_trace(go.Scatter(
            x=[df_idx.index[-1]], y=[df_idx["y"].iloc[-1]],
            mode="markers+text", marker=dict(size=8),
            text=[f"${df_idx['y'].iloc[-1]:.2f}"], textposition="top right",
            showlegend=False
        ))

        # Volume as bars on secondary y
        if vol_series is not None:
            fig.add_trace(go.Bar(
                x=vol_series.index, y=vol_series.values, name="Volume",
                yaxis="y2", opacity=0.4
            ))
            fig.update_layout(
                yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Volume")
            )

        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=30, b=30)
        )
        dark_layout(fig, title=f"{ticker} Price & Volume")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Sparkline of returns (last 30 days)
        st.markdown("*Short-term momentum (30d)*")
        returns = df_idx["y"].pct_change().dropna()
        returns_30 = returns.tail(30)
        if len(returns_30) > 0:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=returns_30.index, y=(1+returns_30).cumprod()-1, mode="lines", name="30d Momentum"))
            dark_layout(fig2, title="30d Cumulative Return")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Not enough data for 30-day momentum sparkline.")

    # Right column: sentiment mini + fundamentals
    with right_col:
        # Small sentiment mini (reuse style)
        st.markdown("### Quick Sentiment")
        # synthetic sentiment to keep stateless
        raw_score = np.random.normal(loc=0.05, scale=0.35)
        sscore = float(np.clip(raw_score, -1, 1))
        bullish = max(sscore, 0) * 100
        bearish = abs(min(sscore, 0)) * 100
        neutral = 100 - (bullish + bearish)

        def _sent_row(emoji, label, value, color):
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                    <div style="font-size:20px;">{emoji}</div>
                    <div style="color:#ffffff; width:70px;">{label}</div>
                    <div style="flex-grow:1;">
                        <div style="background:#121316; height:8px; border-radius:6px;">
                            <div style="background:{color}; width:{value}%; height:8px; border-radius:6px;"></div>
                        </div>
                    </div>
                    <div style="color:#bfcbd6; width:40px; text-align:right;">{value:.0f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        _sent_row("😀", "Bullish", bullish, "#00ff88")
        _sent_row("😐", "Neutral", neutral, "#ffaa00")
        _sent_row("😟", "Bearish", bearish, "#5c8aff")

        # Small sentiment gauge (compact)
        figg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sscore,
            gauge={'axis': {'range': [-1, 1], 'tickcolor': '#888'}, 'bar': {'color': '#00ff88' if sscore>0 else "#ff4b4b"}}
        ))
        dark_layout(figg, title="Sentiment (mini)")
        figg.update_layout(height=200)
        st.plotly_chart(figg, use_container_width=True, config={"displayModeBar": False})

        # Fundamentals card
        st.markdown("### Quick Fundamentals")
        fundamentals_html = "<div style='background:#0e1518; padding:12px; border-radius:10px; border:1px solid #1f2b32; color:#d0d0d0;'>"
        fundamentals_html += f"<div><strong>Market cap:</strong> {f'${market_cap:,}' if market_cap else 'n/a'}</div>"
        fundamentals_html += f"<div><strong>Circulating supply:</strong> {f'{circulating_supply:,}' if circulating_supply else 'n/a'}</div>"
        fundamentals_html += f"<div><strong>Last updated:</strong> {latest_ts.strftime('%Y-%m-%d')}</div>"
        fundamentals_html += "</div>"
        st.markdown(fundamentals_html, unsafe_allow_html=True)

    # Footer quick table: last 5 rows
    st.markdown("### Recent Prices (last 5 days)")
    st.dataframe(df_idx[["y"]].tail(5).rename(columns={"y":"Close"}))

# ---------------------------
# LEFT: Sentiment Breakdown
# ---------------------------