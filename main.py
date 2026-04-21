from fastapi import FastAPI, Depends
import datetime
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import httpx
import anyio
from sqlalchemy.orm import Session
import asyncio
from contextlib import asynccontextmanager
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from ml import get_ml_predictions
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fastapi.middleware.cors import CORSMiddleware
from database import SessionLocal
import models

# changed: added a small in-memory cache for repeated history requests
HISTORY_TTL_SECONDS = 30
HISTORY_CACHE = {}
PRICE_HISTORY_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: client is already created globally or create it here
    yield
    # Shutdown: Close connections gracefully
    await client.aclose()
app = FastAPI(lifespan=lifespan)
analyzer = SentimentIntensityAnalyzer()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", 'https://talos-ui-ten.vercel.app'],
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
client = httpx.AsyncClient(
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
    timeout=httpx.Timeout(15.0) 
)
fred_key = os.getenv("FRED_KEY")
fmp_key = os.getenv("FMP_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True  # Vital: Checks if connection is alive before using it
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"message": "Talos Engine Online"}
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
@app.post("/stock/target")
def save_stock_target(data: dict, db: Session = Depends(get_db)):
    ticker = data.get("ticker").upper()
    user_id = data.get("user_id")
    target_price = data.get("target_price")

    db_target = db.query(models.UserStockTarget).filter(
        models.UserStockTarget.user_id == user_id,
        models.UserStockTarget.ticker == ticker
    ).first()

    if db_target:
        db_target.target_price = target_price
        db_target.updated_at = datetime.datetime.now(datetime.timezone.utc)
    else:
        db_target = models.UserStockTarget(
            user_id=user_id,
            ticker=ticker,
            target_price=target_price
        )
        db.add(db_target)
    
    db.commit()
    return {"status": "saved", "ticker": ticker, "target": target_price}
@app.api_route("/health", methods=["GET", "HEAD"])

def health():
    return {"status": "ok"}

async def get_alpaca_history(ticker, timeframe="1Day", period_days=365):
    # changed: reuse recent ticker history instead of refetching immediately
    cache_key = (ticker.upper(), timeframe, period_days)
    cached = HISTORY_CACHE.get(cache_key)
    now_ts = asyncio.get_running_loop().time()
    if cached and now_ts - cached[0] < HISTORY_TTL_SECONDS:
        return cached[1].copy(deep=False)

    start_date = (
        (datetime.datetime.now() - datetime.timedelta(days=period_days))
        .date()
        .isoformat()
    )

    url = f"{BASE_URL}/stocks/{ticker.upper()}/bars?timeframe={timeframe}&start={start_date}&adjustment=all"

    response = await client.get(url, headers=HEADERS)
    if response.status_code != 200:
        return pd.DataFrame()

    data = response.json()
    if not data.get("bars"):
        return pd.DataFrame()

    # changed: build only the columns we need instead of renaming a full frame
    df = pd.DataFrame.from_records(
        data["bars"],
        columns=["t", "o", "h", "l", "c", "v"]
    )
    df.columns = PRICE_HISTORY_COLUMNS
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    HISTORY_CACHE[cache_key] = (now_ts, df)

    return df


async def port(tickers, num_port=3000):
    try:
        symbols_str = ",".join([t.upper() for t in tickers])
        start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).date().isoformat()
        
        url = f"{BASE_URL}/stocks/bars?timeframe=1Day&symbols={symbols_str}&start={start_date}&adjustment=all&limit=10000"
        response = await client.get(url, headers=HEADERS)
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
        
        weights = gene.random((num_port, assets))
        weights /= weights.sum(axis=1)[:, np.newaxis]
        
        portfolio_returns = weights @ mean_returns.values
        portfolio_risks = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix.values, weights))
        portfolio_sharpe = (portfolio_returns - risk_free) / portfolio_risks
        idx_max_sharpe = np.argmax(portfolio_sharpe)
        idx_min_vol = np.argmin(portfolio_risks)
        max_sharpe = {
            "returns": portfolio_returns[idx_max_sharpe],
            "risk": portfolio_risks[idx_max_sharpe],
            "sharpe": portfolio_sharpe[idx_max_sharpe],
            "Weight": weights[idx_max_sharpe]
        }
        
        min_vol = {
            "returns": portfolio_returns[idx_min_vol],
            "risk": portfolio_risks[idx_min_vol],
            "sharpe": portfolio_sharpe[idx_min_vol],
            "Weight": weights[idx_min_vol]
        }
        return max_sharpe, min_vol

    except Exception as e:
        print(f"PORTFOLIO FATAL ERROR: {e}")
        return None, None
@app.get("/stock/{ticker}/simulate")
async def simulate(ticker: str, target_price: float = None):
    try:
        # 1. Fetch History
        df = await get_alpaca_history(ticker.upper())
        if df.empty: 
            return {"error": "No data found"}

        # 2. Define the synchronous math block
        def run_sim_logic(df_in):
            # Technicals
            df_tech = run_all_technicals(df_in)
            df_tech["SMA_50"] = df_tech["Close"].rolling(50).mean()
            df_tech["SMA_100"] = df_tech["Close"].rolling(100).mean()
            df_tech["Volatility"] = df_tech["Close"].pct_change().rolling(20).std()
            clean = df_tech.dropna(subset=["RSI", "MACD", "SMA_50", "SMA_100", "Volatility"])
            
            # --- CRITICAL FIX START ---
            # ml_output is the percentage (e.g., 2.5)
            ml_output = float(get_ml_predictions(clean, ticker.upper()))
            
            # ml_move_decimal is the decimal (e.g., 0.025)
            ml_move_decimal = ml_output / 100.0
            
            # Drift must be decimal-based
            daily_drift = ml_move_decimal / 30
            # --- CRITICAL FIX END ---

            # Monte Carlo Simulation
            paths, p5, p50, p95 = sim(clean, daily_drift)
            
            # Target success probability (using simulated prices vs dollar target)
            prob = 0
            if target_price:
                prob = round(float((paths[-1] >= target_price).mean() * 100), 2)
            
            return p5, p50, p95, ml_output, prob

        # 3. Execute math in ONE background thread jump
        p5, p50, p95, ml_output, success_prob = await anyio.to_thread.run_sync(run_sim_logic, df)

        # 4. Build JSON Payload
        payload = [
            {"Date": i + 1, "p5": float(p5_i), "p50": float(p50_i), "p95": float(p95_i)}
            for i, (p5_i, p50_i, p95_i) in enumerate(zip(p5, p50, p95))
        ]

        return {
            "data": payload, 
            "probability": f"{success_prob}%", # Formatting as string prevents UI currency bugs
            "ml_expected_price": round(ml_output, 2) # Returns the clean percentage like 2.5
        }
        
    except Exception as e:
        return {"error": f"Sim Error: {str(e)}"}
@app.get("/portfolio")
async def optimize(tickers: str):
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        # Await the async function directly
        result = await port(ticker_list) 
        if result == (None, None): # Ensure you handle the tuple return correctly
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
async def get_stock(ticker: str):
    try:
        start_date = (
            (datetime.datetime.now() - datetime.timedelta(days=365)).date().isoformat()
        )
        
        ticker = ticker.upper()
        quote_task = client.get(f"{BASE_URL}/stocks/{ticker}/quotes/latest", headers=HEADERS)
        bar_task = client.get(
            f"{BASE_URL}/stocks/{ticker}/bars?timeframe=1Day&start={start_date}&adjustment=all", 
            headers=HEADERS
        )
        quote_resp, bar_resp = await asyncio.gather(quote_task, bar_task)
        
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
async def hist(ticker: str, period_days: int = 730):
    try:
        start_date = (
            (datetime.datetime.now() - datetime.timedelta(days=period_days))
            .date()
            .isoformat()
        )
        url = f"{BASE_URL}/stocks/{ticker.upper()}/bars?timeframe=1Day&start={start_date}&adjustment=all&limit=1000"

        res = await client.get(url, headers=HEADERS)
        data = res.json()
        bars = data.get("bars", [])
        return [{"Date": b["t"].split("T")[0], "Close": b["c"]} for b in bars]
    except:
        return []


async def get_macro(series_id, fred_key):
    try:

        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": fred_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 10,
        }
        response = await client.get(url, params=params)
        data = response.json()
        obsv = data["observations"]
        real_data = obsv[0]["value"]
        if real_data == ".":
            return None
        return real_data
    except AttributeError:
        return None
    except Exception:
        return None


@app.get("/macro")
async def macro():
    try:
        tasks = [
            get_macro("A191RL1Q225SBEA", fred_key),
            get_macro("CPIAUCSL", fred_key),
            get_macro("FEDFUNDS", fred_key),
            get_macro("UNRATE", fred_key),
            get_macro("DGS10", fred_key),
            get_macro("SP500", fred_key),
        ]
        results = await asyncio.gather(*tasks)
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
    # changed: removed one extra DataFrame copy from the technical pipeline
    try:
        df["SMA_50"] = df["Close"].rolling(50).mean()
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
    # changed: use NumPy arrays directly for the simulation math
    close = df["Close"].to_numpy(copy=False)
    if np.isnan(close).any():
        close = close[~np.isnan(close)]
    returns = close[1:] / close[:-1] - 1
    price = close[-1]

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
    # changed: compute the rolling stats once and reuse them
    rolling = df["Close"].rolling(window=window)
    sma_20 = rolling.mean()
    close_std = rolling.std()
    df["SMA_20"] = sma_20
    df["BB_Up"] = sma_20 + num_std * close_std
    df["BB_Down"] = sma_20 - num_std * close_std
    return df


def macd(df):
    emal12 = df["Close"].ewm(span=12, adjust=False).mean()
    emal26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = emal12 - emal26
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]
    return df


def rsi(df, period=14):
    # changed: make one explicit copy and reuse the converted Close series
    df = df.dropna().copy()
    close = pd.to_numeric(df["Close"], errors="coerce")
    df["Close"] = close
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def run_all_technicals(df):
    # changed: copy once at the pipeline boundary instead of inside each helper
    df = df.copy()
    df = rsi(df)
    df = macd(df)
    df = bollinger(df)
    return wrap(df)

BASE_URL = "https://data.alpaca.markets/v2"
HEADERS = {
    "APCA-API-KEY-ID": os.getenv("ALPACA_KEY"),
    "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET"),
}


async def get_multiple_prices(symbols):

    url = f"{BASE_URL}/stocks/quotes/latest?symbols={symbols}"
    response = await client.get(url, headers=HEADERS)
    return response.json()


@app.get("/analyze/{ticker}")
async def analyse(ticker: str):
    try:
        df, spy = await asyncio.gather(
            get_alpaca_history(ticker.upper()),
            get_alpaca_history("SPY")
        )
        if df.empty:
            return {"error": f"No data found for {ticker}"}
        df = await anyio.to_thread.run_sync(run_all_technicals, df)
        print(df.columns.tolist())
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

        rsi_val = df["RSI"].iloc[-1]
        macd_val = df["MACD"].iloc[-1]
        signal_line_val = df["Signal_Line"].iloc[-1]
        price = df["Close"].iloc[-1]
        sma50_val = df["SMA_50"].iloc[-1]
        bb_lower = df["BB_Down"].iloc[-1]
        bb_upper = df["BB_Up"].iloc[-1]
        
        score = 0
        
        
        if price > sma50_val: score += 1  
        else: score -= 1             
        
        if macd_val > signal_line_val: score += 1
        else: score -= 1
        
        
        if price < bb_lower: score += 2  
        if price > bb_upper: score -= 2 
        
        if rsi_val < 35: score += 1
        if rsi_val > 65: score -= 1
        sig = "Neutral"
        if score >= 3:    sig = "Strong Buy"
        elif score >= 1:  sig = "Buy"
        elif score <= -3: sig = "Strong Sell"
        elif score <= -1: sig = "Sell"
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
            "rsi_signal": sig,
            "stock_cagr": stock_cagr,
            "spy_cagr": spy_cagr,
            "sharpe": sharpe,
        }
    except Exception as e:
        return {"error": str(e)}


def backtest(df, buy_rsi = 30, sell_rsi=70, starter_cash = 10000):
    try:
        rsi = df['RSI'].values
        close = df['Close'].values
        signals = np.where(rsi < buy_rsi, 1, np.where(rsi > sell_rsi, -1, 0))
        position = np.clip(np.cumsum(signals), 0, 1)
        returns = np.diff(close) / close[:-1]
        position = position[:-1]

        strat_return = position * returns


        portfolio = starter_cash * np.cumprod(1 + strat_return)

        tot_returns = (portfolio[-1] - starter_cash) * 100 / starter_cash

        sharpe = np.mean(strat_return) / np.std(strat_return) * np.sqrt(252)
        
        if np.std(strat_return) == 0:
            sharpe = 0.0
    
        buy = len(np.where(signals == 1)[0])
        sell = len(np.where(signals == -1)[0])
        return {
            "portfolio": portfolio,
            "total_return": float(tot_returns),
            "sharpe": float(sharpe),
            "buy": buy,
            'sell':sell
        }
    except KeyError:
        return {
            "portfolio":'N/A',
            "total_return":'N/A',
            "sharpe":'N/A',
            "buy": 'N/A',
            'sell':'N/A'
        }
    

@app.get("/stock/{ticker}/backtest")
async def backtester(ticker: str, buy_rsi: float = 30, sell_rsi: float = 70, starter_cash: float = 10000):
    try:
        df = await get_alpaca_history(ticker.upper(), period_days=365*2)
        if df.empty:
            return {"error": f"No data found for {ticker}"}
        df = await anyio.to_thread.run_sync(run_all_technicals, df)
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df.dropna(inplace=True)
        result = backtest(df, buy_rsi, sell_rsi, starter_cash)
        close = df['Close'].values
        returns = np.diff(close)/close[:-1]
        buy_hold = starter_cash * np.cumprod(1 + returns)
        portfolio = np.array(result["portfolio"])
        peak = np.maximum.accumulate(portfolio)
        drawdown = (portfolio - peak) / peak
        buy_hold_return = round((buy_hold[-1] - starter_cash) * 100 / starter_cash, 2)
        max_drawdown = round(float(np.min(drawdown) * 100), 2)
        return {
            "total_return": round(result["total_return"], 2),
            "buy_hold_return": buy_hold_return,
            "sharpe": round(float(result["sharpe"]), 2),
            "max_drawdown": max_drawdown,
            "buy_signals": result["buy"],
            "sell_signals": result["sell"],
            "portfolio": result["portfolio"].tolist(),
            "buy_hold": buy_hold.tolist(),
        }
    except Exception as e:
        return {"error": f"Endpoint Error: {str(e)}"}
@app.get("/stock/{ticker}/sentiment")
async def get_sentiment(ticker):
    try:
        NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
        
     
        params = {
            "symbols": ticker.upper(),
            "limit": 10
        }
        
   
        response = await client.get(NEWS_URL, headers=HEADERS, params=params)
        if response.status_code != 200:
            return {"error": f"Alpaca API error: {response.status_code}"}
        news_data = response.json().get("news", [])
        tasks = [
        anyio.to_thread.run_sync(analyzer.polarity_scores, f"{a['headline']} {a['summary']}") 
        for a in news_data
    ]
        results = await asyncio.gather(*tasks)
        total_score = 0
        articles_output = []
        for article, sentiment_result in zip(news_data, results):
            score = sentiment_result["compound"] 
            total_score += score
            articles_output.append({
                "headline": article["headline"],
                "url": article["url"],
                "sentiment": "Bullish" if score > 0.05 else "Bearish" if score < -0.05 else "Neutral"
            })
        avg_score = total_score / len(news_data) if news_data else 0
        label = "Neutral"
        if avg_score > 0.15: label = "Strong Bullish"
        elif avg_score > 0.05: label = "Bullish"
        elif avg_score < -0.15: label = "Strong Bearish"
        elif avg_score < -0.05: label = "Bearish"

        return {
            "score": round(avg_score, 2), # -1 to 1
            "label": label,
            "articles": articles_output
        }
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return {"error": str(e)}
    
@app.get("/stock/{ticker}/target/{user_id}")
async def get_stock_target(ticker: str, user_id: str, db: Session = Depends(get_db)):
    target = db.query(models.UserStockTarget).filter(
        models.UserStockTarget.user_id == user_id,
        models.UserStockTarget.ticker == ticker.upper()
    ).first()
    
    if target:
        return {"target_price": target.target_price}
    return {"target_price": None}
