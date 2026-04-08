from fastapi import FastAPI
import datetime
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import os
import asyncio
from ml import get_ml_predictions
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", 'https://talos-ui-ten.vercel.app'],
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
fred_key = os.getenv("FRED_KEY")
fmp_key = os.getenv("FMP_KEY")

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"message": "Talos Engine Online"}


def get_alpaca_history(ticker, timeframe="1Day", period_days=365):
    import datetime

    start_date = (
        (datetime.datetime.now() - datetime.timedelta(days=period_days))
        .date()
        .isoformat()
    )

    url = f"{BASE_URL}/stocks/{ticker.upper()}/bars?timeframe={timeframe}&start={start_date}&adjustment=all"

    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        return pd.DataFrame()

    data = response.json()
    if not data.get("bars"):
        return pd.DataFrame()

    df = pd.DataFrame(data["bars"])

    df = df.rename(
        columns={
            "t": "Date",
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume",
        }
    )

    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    return df


def port(tickers, num_port=3000):
    try:
        symbols_str = ",".join([t.upper() for t in tickers])
        start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).date().isoformat()
        
        url = f"{BASE_URL}/stocks/bars?timeframe=1Day&symbols={symbols_str}&start={start_date}&adjustment=all&limit=10000"
        response = requests.get(url, headers=HEADERS)
        data = response.json()

        if not data.get("bars"):
            print("DEBUG: No bars returned from Alpaca")
            return None, None

        all_bars = []
        for symbol, bars in data["bars"].items():
            temp_df = pd.DataFrame(bars)
            temp_df["symbol"] = symbol
            all_bars.append(temp_df)

        df_long = pd.concat(all_bars)
        prices = df_long.pivot(index="t", columns="symbol", values="c")
        
        if prices.empty or len(prices.columns) < 2:
            print(f"DEBUG: Not enough overlapping data. Columns found: {prices.columns}")
            return None, None

        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        prices = prices.ffill().dropna() 
        
        returns = np.log(prices / prices.shift(1)).dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        assets = len(prices.columns) 
        risk_free = 0.0422
        gene = np.random.default_rng()
        result = []
        
        for _ in range(num_port):
            w = gene.random(assets)
            w /= w.sum()
            portfolio_return = np.dot(w, mean_returns)
            portfolio_risk = np.sqrt(w.T @ cov_matrix.values @ w)
            sharpe = (portfolio_return - risk_free) / portfolio_risk
            result.append({"returns": portfolio_return, "risk": portfolio_risk, "sharpe": sharpe, "Weight": w})
            
        result_df = pd.DataFrame(result)
        return result_df.iloc[result_df["sharpe"].idxmax()], result_df.iloc[result_df["risk"].idxmin()]

    except Exception as e:
        print(f"PORTFOLIO FATAL ERROR: {e}")
        return None, None

@app.get("/stock/{ticker}/simulate")
def simulate(ticker: str, target_price: float = None):
    try:
        df = get_alpaca_history(ticker.upper())
        df = rsi(df)
        df = macd(df)
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_100"] = df["Close"].rolling(window=100).mean()
        df["Volatility"] = df["Close"].pct_change().rolling(window=20).std()
        if df.empty:
            return {"error": f"No data found for {ticker}"}
        predicted_price = get_ml_predictions(df)
        current_price = float(df["Close"].iloc[-1])
        ml_total_return = (predicted_price - current_price) / current_price
        drift = ml_total_return / 30
        price_path, p5, p50, p95 = sim(df, drift=drift)
        payload = []
        success_rate = 0
        if target_price is not None:
            success_rate = (price_path[-1] >= target_price).mean() * 100
            success_rate = round(success_rate, 2)
        for i in range(len(price_path)):
            payload.append(
                {
                    "Date": int(i+1),
                    "p5": float(p5[i]),
                    "p50": float(p50[i]),
                    "p95": float(p95[i]),
                }
            )
        return {"data": payload, "probability": success_rate, "ml_expected_price": round(ml_total_return, 2)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/portfolio")
def optimize(tickers: str):
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        result = port(ticker_list)
        if result is None:
            return {"error": "Could not optimize"}
        max_sharpe_df, min_vol = result
        return {
            "max_sharpe": {
                "return": float(max_sharpe_df["returns"]),
                "risk": float(max_sharpe_df["risk"]),
                "sharpe": float(max_sharpe_df["sharpe"]),
                "weights": {
                    t: float(w) for t, w in zip(ticker_list, max_sharpe_df["Weight"])
                },
            },
            "min_vol": {
                "return": float(min_vol["returns"]),
                "risk": float(min_vol["risk"]),
                "sharpe": float(min_vol["sharpe"]),
                "weights": {
                    t: float(w) for t, w in zip(ticker_list, min_vol["Weight"])
                },
            },
        }
    except Exception as e:
        return {"error": f"{e}"}


@app.get("/stock/{ticker}")
def get_stock(ticker: str):
    try:
        ticker = ticker.upper()
        quote_resp = requests.get(
            f"{BASE_URL}/stocks/{ticker}/quotes/latest", headers=HEADERS
        )
        start_date = (
            (datetime.datetime.now() - datetime.timedelta(days=365)).date().isoformat()
        )
        bar_resp = requests.get(
            f"{BASE_URL}/stocks/{ticker}/bars?timeframe=1Day&start={start_date}&adjustment=all&limit=500",
            headers=HEADERS,
        )

        if quote_resp.status_code != 200 or bar_resp.status_code != 200:
            return {"error": f"Alpaca API error: {quote_resp.status_code}"}
        q_data = quote_resp.json()
        b_data = bar_resp.json()
        if "quote" not in q_data or "bars" not in b_data or not b_data["bars"]:
            return {"error": "No data found for ticker"}
        quote = q_data.get("quote")
        bar = b_data.get("bars")
        if not quote or not bar:
            return {"error": "No data found for this ticker"}
        daily = bar[0] if bar else None
        highs = [b["h"] for b in bar] if bar else [None]
        lows = [b["l"] for b in bar] if bar else [None]

        max_low = min(lows) if lows else None
        max_high = max(highs) if highs else None
        return {
            'ticker': ticker,
            "price": quote.get("ap"),
            "open": daily.get("o"),
            "high": daily.get("h"),
            "low": daily.get("l"),
            "volume": daily.get("v"),
            "close": daily.get("c"),
            "max_low": max_low,
            "max_high": max_high,
        }
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return {"error": str(e)}


@app.get("/stock/{ticker}/history")
def hist(ticker: str, period_days: int = 730):
    try:
        start_date = (
            (datetime.datetime.now() - datetime.timedelta(days=period_days))
            .date()
            .isoformat()
        )
        url = f"{BASE_URL}/stocks/{ticker.upper()}/bars?timeframe=1Day&start={start_date}&adjustment=all&limit=1000"

        res = requests.get(url, headers=HEADERS)
        data = res.json()
        bars = data.get("bars", [])
        return [{"Date": b["t"].split("T")[0], "Close": b["c"]} for b in bars]
    except:
        return []


def get_macro(series_id, fred_key):
    try:

        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": fred_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 10,
        }
        response = requests.get(url=url, params=params)
        data = response.json()
        obsv = data["observations"]
        real_data = obsv[0]["value"]
        if real_data == ".":
            return None
        return real_data
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.Timeout:
        return None
    except AttributeError:
        return None
    except Exception:
        return None


@app.get("/macro")
async def macro():
    try:
        loop = asyncio.get_event_loop()
        results = [
            loop.run_in_executor(None, get_macro, "A191RL1Q225SBEA", fred_key),
            loop.run_in_executor(None, get_macro, "CPIAUCSL", fred_key),
            loop.run_in_executor(None, get_macro, "FEDFUNDS", fred_key),
            loop.run_in_executor(None, get_macro, "UNRATE", fred_key),
            loop.run_in_executor(None, get_macro, "DGS10", fred_key),
            loop.run_in_executor(None, get_macro, "SP500", fred_key),
        ]
        results = await asyncio.gather(*results)
        data = {
            "gdp_growth": results[0],
            "inflation": results[1],
            "fed_funds": results[2],
            "unemployment": results[3],
            "treasury_yield": results[4],
            "sp500": results[5],
        }
        return data
    except Exception as e:
        return {"error": str(e)}


def wrap(df):
    df = df.copy()
    try:
        df["Ty"] = (df["High"] + df["Low"] + df["Close"]) / 3
    except ZeroDivisionError:
        return None

    df["Cum_TP_Vol"] = (df["Ty"] * df["Volume"]).cumsum()

    df["Cum_Vol"] = df["Volume"].cumsum()

    df["VWAP"] = df["Cum_TP_Vol"] / df["Cum_Vol"]

    return df


def atr(df, period=14):
    df = df.copy()
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
    low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()

    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1)
    tr = tr.max(axis=1)

    atr = tr.rolling(window=period).mean()
    return atr.iloc[-1]


def sharpness(df, risk_free):
    returns = df["Close"].dropna().pct_change().dropna()
    daily_rf = (1 + risk_free) ** (1 / 252) - 1
    mean = returns.mean()
    vola = returns.std()
    if vola == 0:
        return np.nan
    sharpe_ratio = ((mean - daily_rf) / vola) * np.sqrt(252)

    return sharpe_ratio


def sim(df, drift=None):
    returns = df["Close"].dropna().pct_change()
    price = df["Close"].iloc[-1]

    vola = returns.std()
    ret = drift if drift is not None else returns.mean()

    rng = np.random.default_rng()

    noise = rng.normal(ret, vola, (30, 1000))

    price_path = price * (1 + noise).cumprod(axis=0)

    p5 = np.percentile(price_path, 5, axis=1)
    p50 = np.percentile(price_path, 50, axis=1)
    p95 = np.percentile(price_path, 95, axis=1)

    return price_path, p5, p50, p95


def bollinger(df, window=20, num_std=2):

    df = df.copy()
    df["SMA_20"] = df["Close"].rolling(window=window).mean()
    df["BB_Up"] = df["SMA_20"] + num_std * df["Close"].rolling(window=window).std()
    df["BB_Down"] = df["SMA_20"] - num_std * df["Close"].rolling(window=window).std()
    return df


def macd(df):
    emal12 = df["Close"].ewm(span=12, adjust=False).mean()
    emal26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = emal12 - emal26
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]
    return df


def rsi(df, period=14):
    df = df.dropna()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


BASE_URL = "https://data.alpaca.markets/v2"
HEADERS = {
    "APCA-API-KEY-ID": os.getenv("ALPACA_KEY"),
    "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET"),
}


def get_multiple_prices(symbols):

    url = f"{BASE_URL}/stocks/quotes/latest?symbols={symbols}"
    response = requests.get(url, headers=HEADERS)
    return response.json()


@app.get("/analyze/{ticker}")
def analyse(ticker: str):
    try:
        df = get_alpaca_history(ticker.upper())
        if df.empty:
            return {"error": f"No data found for {ticker}"}
        spy = get_alpaca_history("SPY")
        df = rsi(df)
        df = macd(df)
        df = bollinger(df)
        df = wrap(df)
        current_rsi = float(df["RSI"].iloc[-1])
        current_macd = float(df["MACD"].iloc[-1])
        signal_line = float(df["Signal_Line"].iloc[-1])
        current_price = float(df["Close"].iloc[-1])
        sma50 = float(df["Close"].rolling(50).mean().iloc[-1])

        sma100 = float(df["Close"].rolling(100).mean().iloc[-1])
        annual_vol = float(df["Close"].pct_change().std() * (252**0.5) * 100)

        def cagr(df, price_col):
            start = df[price_col].iloc[0]
            end = df[price_col].iloc[-1]
            days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days
            if days == 0 or start == 0:
                return 0
            return ((end / start) ** (365.25 / days) - 1) * 100

        signal = current_rsi
        if signal <= 30:
            signal = "Buy"
        elif signal > 30 and signal < 70:
            signal = "Hold"
        else:
            signal = "Sell"
        spy_cagr = cagr(spy, "Close")
        stock_cagr = cagr(df, "Close")
        risk_free = 0.0422
        sharpe = float(sharpness(df, risk_free))

        return {
            "rsi": round(float(current_rsi)),
            "macd": round(float(current_macd)),
            "signal_line": signal_line,
            "price": current_price,
            "sma50": sma50,
            "sma100": sma100,
            "vola": round(float(annual_vol)),
            "rsi_signal": signal,
            "stock_cagr": stock_cagr,
            "spy_cagr": spy_cagr,
            "sharpe": sharpe,
        }
    except Exception as e:
        return {"error": str(e)}
