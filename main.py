from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import datetime
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import websockets
import httpx
import anyio
from sqlalchemy.orm import Session
import asyncio
from contextlib import asynccontextmanager
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from ml import get_ml_predictions
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()
from database import SessionLocal
import models
#changed: avoid the deprecated ORJSONResponse wrapper by using a tiny JSONResponse subclass backed by orjson when available.
class ConnectionManager:
    def __init__(self):
        # We store active connections in a list
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
       
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        

    async def broadcast_tick(self, tick_data: dict):
        """Sends a live price tick to EVERY connected user"""
        # We use orjson (already in your code) for ultra-fast serialization
        message = orjson.dumps(tick_data)
        for connection in self.active_connections:
            try:
                await connection.send_bytes(message)
            except Exception:
                # If a connection is dead, we'll clean it up later via disconnect
                pass
manager = ConnectionManager()
try:
    import orjson

    #changed: keep fast JSON serialization without depending on FastAPI's deprecated ORJSONResponse helper.
    class FastJSONResponse(JSONResponse):
        media_type = "application/json"

        def render(self, content) -> bytes:
            return orjson.dumps(content)

    DEFAULT_RESPONSE_CLASS = FastJSONResponse
except ImportError:
    DEFAULT_RESPONSE_CLASS = JSONResponse

# This software is provided as-is for educational and research purposes. The developer is not responsible for any financial losses incurred through the use of this code.
#changed: added a small in-memory cache for repeated history requests
HISTORY_TTL_SECONDS = 30
HISTORY_CACHE = {}
#changed: cache repeated portfolio and macro responses for short bursts of traffic
RESPONSE_TTL_SECONDS = 30
PORTFOLIO_CACHE = {}
MACRO_CACHE = {}
#changed: cache computed technical frames and sentiment payloads so repeated requests avoid redoing CPU-heavy work.
TECHNICAL_CACHE = {}
SENTIMENT_CACHE = {}
#changed: cache hot user target lookups to avoid repeated round-trips for the same key
TARGET_CACHE_TTL_SECONDS = 60
TARGET_CACHE = {}
#changed: reuse one shared feature list anywhere we need fully-computed technical columns
TECHNICAL_REQUIRED_COLUMNS = ["RSI", "MACD", "SMA_50", "SMA_100", "Volatility"]
#changed: dedupe identical in-flight async work so concurrent traffic shares one fetch or calculation.
INFLIGHT_HISTORY_TASKS = {}
INFLIGHT_PORTFOLIO_TASKS = {}
INFLIGHT_TECHNICAL_TASKS = {}
INFLIGHT_SENTIMENT_TASKS = {}
INFLIGHT_MACRO_TASKS = {}
def get_all_active_tickers(db: Session):
    # Use your optimized SQLAlchemy logic to get unique tickers
    result = db.execute(select(models.UserStockTarget.ticker).distinct()).scalars().all()
    # Always include a high-volume "heartbeat" like BTC/USD to keep the pipe warm
    tickers = list(set(result + ["BTC/USD"]))
    return [t.upper() for t in tickers]
async def alpaca_to_shield_bridge():
    url = "wss://stream.data.alpaca.markets/v2/iex"
    
    while True:
        try:
            async with websockets.connect(url) as ws:
                # 1. Auth
                await ws.send(orjson.dumps({
                    "action": "auth",
                    "key": os.getenv("ALPACA_KEY"),
                    "secret": os.getenv("ALPACA_SECRET")
                }))
                
                # 2. Get tickers from DB (using a temporary session)
                with SessionLocal() as db:
                    active_tickers = get_all_active_tickers(db)
                
                # 3. Dynamic Subscribe
                await ws.send(orjson.dumps({
                    "action": "subscribe",
                    "trades": active_tickers,
                    "quotes": active_tickers # Adding quotes for more density
                }))

                print(f"🛡️  Shield Bridge: Subscribed to {len(active_tickers)} tickers")

                while True:
                    raw_data = await ws.recv()
                    data = orjson.loads(raw_data)
                    
                    for msg in data:
                        if msg.get("T") in ["t", "q"]:
                            # 1. Broadcast to UI as usual
                            await manager.broadcast_tick(msg)
                            
                            # 2. THE FIX: Extract price and check targets
                            # 'p' is Trade Price, 'ap' is Ask Price (for quotes)
                            current_price = msg.get("p") or msg.get("ap")
                            ticker = msg.get("S") # 'S' is the Symbol
                            
                            if current_price and ticker:
                                # We use create_task so the check doesn't block the stream
                                asyncio.create_task(check_shield_activation(ticker, current_price))

        except Exception as e:
            print(f"🛡️  Shield Bridge Error: {e}. Retrying...")
            await asyncio.sleep(5)
LAST_ALERT_TIME = {}
async def check_shield_activation(ticker: str, current_price: float):
    #changed: use the shared target cache for quick lookups and keep the DB as the source of truth for cache misses.
    for (user_id, cached_ticker), (ts, target_price) in TARGET_CACHE.items():
        if cached_ticker == ticker and abs(current_price - target_price) / target_price <= 0.01:
            alert_key = (user_id, ticker)
            now = time.time()
            if now - LAST_ALERT_TIME.get(alert_key, 0) < 60:
                continue 
            if current_price >= target_price:
                LAST_ALERT_TIME[alert_key] = now
                print(f"!!! ALERT for {user_id}: {ticker} hit {current_price} (Target: {target_price})")
                
                # Push an alert back through the WebSocket to the specific user
                alert_msg = {
                    "type": "SHIELD_ALERT",
                    "ticker": ticker,
                    "price": current_price,
                    "msg": f"Target of {target_price} reached!"
                }
                await manager.broadcast_tick(alert_msg)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: client is already created globally or create it here
    stream_task = asyncio.create_task(alpaca_to_shield_bridge())
    yield
    # Shutdown: Close connections gracefully
    await client.aclose()
#changed: use the custom fast JSON response application-wide when orjson is installed.
app = FastAPI(lifespan=lifespan, default_response_class=DEFAULT_RESPONSE_CLASS)
analyzer = SentimentIntensityAnalyzer()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", 'https://talos-ui-ten.vercel.app'],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = httpx.AsyncClient(
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
    timeout=httpx.Timeout(15.0) 
)
fred_key = os.getenv("FRED_KEY")
@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"message": "Talos Engine Online"}
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

#changed: centralize the hot target cache bookkeeping so reads and writes share one fast path
def get_cached_target_price(user_id: str, ticker: str):
    cache_key = (user_id, ticker)
    cached = TARGET_CACHE.get(cache_key)
    if not cached:
        return None, False
    now_ts = time.monotonic()
    if now_ts - cached[0] >= TARGET_CACHE_TTL_SECONDS:
        TARGET_CACHE.pop(cache_key, None)
        return None, False
    return cached[1], True

#changed: keep target cache writes tiny and explicit
def set_cached_target_price(user_id: str, ticker: str, target_price):
    TARGET_CACHE[(user_id, ticker)] = (time.monotonic(), target_price)

#changed: standardize TTL cache reads so the hot caches share one expiration path.
def get_ttl_cache_value(cache: dict, cache_key, ttl_seconds: float):
    cached = cache.get(cache_key)
    if not cached:
        return None, False
    now_ts = time.monotonic()
    if now_ts - cached[0] >= ttl_seconds:
        cache.pop(cache_key, None)
        return None, False
    return cached[1], True

#changed: keep generic TTL cache writes tiny and reusable for data, analytics, and response payloads.
def set_ttl_cache_value(cache: dict, cache_key, value):
    cache[cache_key] = (time.monotonic(), value)

#changed: share identical in-flight async work so concurrent requests await one task instead of duplicating it.
async def get_or_create_task_result(task_cache: dict, cache_key, coroutine_factory):
    task = task_cache.get(cache_key)
    if task is None:
        task = asyncio.create_task(coroutine_factory())
        task_cache[cache_key] = task
    try:
        return await task
    finally:
        if task_cache.get(cache_key) is task and task.done():
            task_cache.pop(cache_key, None)

@app.post("/stock/target")
def save_stock_target(data: dict, db: Session = Depends(get_db)):
    ticker = data.get("ticker").upper()
    user_id = data.get("user_id")
    target_price = data.get("target_price")

    #changed: use a single database round-trip for save/update when the backend supports upsert
    target_table = models.UserStockTarget.__table__
    if db.bind and db.bind.dialect.name == "postgresql":
        stmt = pg_insert(target_table).values(
            user_id=user_id,
            ticker=ticker,
            target_price=target_price,
        ).on_conflict_do_update(
            index_elements=["user_id", "ticker"],
            set_={
                "target_price": target_price,
                "updated_at": datetime.datetime.now(datetime.timezone.utc),
            },
        )
        db.execute(stmt)
    elif db.bind and db.bind.dialect.name == "sqlite":
        stmt = sqlite_insert(target_table).values(
            user_id=user_id,
            ticker=ticker,
            target_price=target_price,
        ).on_conflict_do_update(
            index_elements=["user_id", "ticker"],
            set_={
                "target_price": target_price,
                "updated_at": datetime.datetime.now(datetime.timezone.utc),
            },
        )
        db.execute(stmt)
    else:
        #changed: keep the fallback path lean by selecting only the row we may mutate
        db_target = db.execute(
            select(models.UserStockTarget).where(
                models.UserStockTarget.user_id == user_id,
                models.UserStockTarget.ticker == ticker,
            )
        ).scalar_one_or_none()
        if db_target:
            db_target.target_price = target_price
            db_target.updated_at = datetime.datetime.now(datetime.timezone.utc)
        else:
            db.add(
                models.UserStockTarget(
                    user_id=user_id,
                    ticker=ticker,
                    target_price=target_price
                )
            )

    db.commit()
    #changed: update the hot cache immediately after a successful write
    set_cached_target_price(user_id, ticker, target_price)
    return {"status": "saved", "ticker": ticker, "target": target_price}
@app.api_route("/health", methods=["GET", "HEAD"])

def health():
    return {"status": "ok"}

async def get_alpaca_history(ticker, timeframe="1Day", period_days=365):
    #changed: reuse recent ticker history instead of refetching immediately.
    upper_ticker = ticker.upper()
    cache_key = (upper_ticker, timeframe, period_days)
    cached, found = get_ttl_cache_value(HISTORY_CACHE, cache_key, HISTORY_TTL_SECONDS)
    if found:
        return cached

    async def fetch_history():
        start_date = (
            (datetime.datetime.now() - datetime.timedelta(days=period_days))
            .date()
            .isoformat()
        )
        url = f"{BASE_URL}/stocks/{upper_ticker}/bars?timeframe={timeframe}&start={start_date}&adjustment=all"
        response = await client.get(url, headers=HEADERS)
        if response.status_code != 200:
            return pd.DataFrame()

        bars = response.json().get("bars")
        if not bars:
            return pd.DataFrame()

        #changed: build typed columns once so downstream technical and portfolio math stays numeric without extra coercion.
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime([bar["t"] for bar in bars], utc=True).tz_localize(None),
                "Open": np.fromiter((bar["o"] for bar in bars), dtype=np.float64, count=len(bars)),
                "High": np.fromiter((bar["h"] for bar in bars), dtype=np.float64, count=len(bars)),
                "Low": np.fromiter((bar["l"] for bar in bars), dtype=np.float64, count=len(bars)),
                "Close": np.fromiter((bar["c"] for bar in bars), dtype=np.float64, count=len(bars)),
                "Volume": np.fromiter((bar["v"] for bar in bars), dtype=np.float64, count=len(bars)),
            }
        )
        set_ttl_cache_value(HISTORY_CACHE, cache_key, df)
        return df

    return await get_or_create_task_result(INFLIGHT_HISTORY_TASKS, cache_key, fetch_history)


#changed: cache fully-computed technical frames so simulation, analysis, and backtests reuse the same indicator work.
async def get_technical_history(ticker, timeframe="1Day", period_days=365):
    upper_ticker = ticker.upper()
    cache_key = (upper_ticker, timeframe, period_days)
    cached, found = get_ttl_cache_value(TECHNICAL_CACHE, cache_key, HISTORY_TTL_SECONDS)
    if found:
        return cached

    async def build_technical_history():
        history_df = await get_alpaca_history(upper_ticker, timeframe=timeframe, period_days=period_days)
        if history_df.empty:
            return history_df
        technical_df = await anyio.to_thread.run_sync(run_all_technicals, history_df)
        set_ttl_cache_value(TECHNICAL_CACHE, cache_key, technical_df)
        return technical_df

    return await get_or_create_task_result(INFLIGHT_TECHNICAL_TASKS, cache_key, build_technical_history)


async def port(tickers, num_port=3000):
    try:
        #changed: normalize once, reject underspecified requests early, and cache identical portfolio optimizations briefly.
        normalized_tickers = tuple(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
        if len(normalized_tickers) < 2:
            return None, None
        cache_key = (normalized_tickers, num_port)
        cached, found = get_ttl_cache_value(PORTFOLIO_CACHE, cache_key, RESPONSE_TTL_SECONDS)
        if found:
            return cached

        async def build_portfolio():
            symbols_str = ",".join(normalized_tickers)
            start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).date().isoformat()
            url = f"{BASE_URL}/stocks/bars?timeframe=1Day&symbols={symbols_str}&start={start_date}&adjustment=all&limit=10000"
            response = await client.get(url, headers=HEADERS)
            if response.status_code != 200:
                return None, None

            bars_by_symbol = response.json().get("bars")
            if not bars_by_symbol:
                print("DEBUG: No bars returned from Alpaca")
                return None, None

            #changed: build the close-price matrix directly and keep the final ticker ordering aligned with the optimized weights.
            prices = pd.DataFrame(
                {
                    symbol: pd.Series(
                        np.fromiter((bar["c"] for bar in bars), dtype=np.float64, count=len(bars)),
                        index=pd.to_datetime([bar["t"] for bar in bars], utc=True).tz_localize(None),
                    )
                    for symbol, bars in bars_by_symbol.items()
                    if bars
                }
            ).sort_index()
            if prices.empty or len(prices.columns) < 2:
                print(f"DEBUG: Not enough overlapping data. Columns found: {prices.columns}")
                return None, None

            prices = prices.ffill().dropna()
            price_matrix = prices.to_numpy(dtype=np.float64, copy=False)
            if price_matrix.shape[0] < 2 or price_matrix.shape[1] < 2:
                return None, None

            #changed: run the portfolio math on dense NumPy arrays to avoid extra pandas allocation overhead.
            log_returns = np.diff(np.log(price_matrix), axis=0)
            mean_returns = log_returns.mean(axis=0) * 252
            cov_matrix = np.cov(log_returns, rowvar=False) * 252
            assets = price_matrix.shape[1]
            risk_free = 0.0422
            gene = np.random.default_rng()
            weights = gene.random((num_port, assets))
            weights /= weights.sum(axis=1, keepdims=True)
            portfolio_returns = weights @ mean_returns
            portfolio_risks = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_matrix, weights, optimize=True))
            portfolio_sharpe = np.divide(
                portfolio_returns - risk_free,
                portfolio_risks,
                out=np.full(portfolio_returns.shape, -np.inf),
                where=portfolio_risks > 0,
            )
            idx_max_sharpe = int(np.argmax(portfolio_sharpe))
            idx_min_vol = int(np.argmin(portfolio_risks))
            portfolio_tickers = prices.columns.to_list()
            max_sharpe = {
                "returns": portfolio_returns[idx_max_sharpe],
                "risk": portfolio_risks[idx_max_sharpe],
                "sharpe": portfolio_sharpe[idx_max_sharpe],
                "Weight": weights[idx_max_sharpe],
                "tickers": portfolio_tickers,
            }
            min_vol = {
                "returns": portfolio_returns[idx_min_vol],
                "risk": portfolio_risks[idx_min_vol],
                "sharpe": portfolio_sharpe[idx_min_vol],
                "Weight": weights[idx_min_vol],
                "tickers": portfolio_tickers,
            }
            result = (max_sharpe, min_vol)
            set_ttl_cache_value(PORTFOLIO_CACHE, cache_key, result)
            return result

        return await get_or_create_task_result(INFLIGHT_PORTFOLIO_TASKS, cache_key, build_portfolio)

    except Exception as e:
        print(f"PORTFOLIO FATAL ERROR: {e}")
        return None, None
@app.get("/stock/{ticker}/simulate")
async def simulate(ticker: str, target_price: float = None):
    try:
        #changed: reuse cached technical history so simulation only pays for the Monte Carlo and ML steps.
        technical_df = await get_technical_history(ticker.upper())
        if technical_df.empty:
            return {"error": "No data found"}

        #changed: keep the CPU-heavy simulation work inside one background-thread hop.
        def run_sim_logic(df_in):
            clean = df_in.dropna(subset=TECHNICAL_REQUIRED_COLUMNS)
            if clean.empty:
                raise ValueError("Not enough clean data for simulation")
            ml_output = float(get_ml_predictions(clean, ticker.upper()))
            ml_move_decimal = ml_output / 100.0
            daily_drift = ml_move_decimal / 30
            paths, p5, p50, p95 = sim(clean, daily_drift)
            prob = 0
            if target_price:
                prob = round(float((paths[-1] >= target_price).mean() * 100), 2)
            return p5, p50, p95, ml_output, prob

        p5, p50, p95, ml_output, success_prob = await anyio.to_thread.run_sync(run_sim_logic, technical_df)

        #changed: build the response payload from the already-vectorized percentile arrays.
        payload = [
            {"Date": day, "p5": float(p5_i), "p50": float(p50_i), "p95": float(p95_i)}
            for day, p5_i, p50_i, p95_i in zip(range(1, len(p5) + 1), p5, p50, p95)
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
        #changed: normalize ticker input once so the optimizer and response payload use the same ordering.
        ticker_list = tuple(dict.fromkeys(t.strip().upper() for t in tickers.split(",") if t.strip()))
        result = await port(ticker_list)
        if result == (None, None):
            return {"error": "Could not optimize"}
        max_sharpe_df, min_vol = result
        return {
            "max_sharpe": {
                "return": float(max_sharpe_df["returns"]),
                "risk": float(max_sharpe_df["risk"]),
                "sharpe": float(max_sharpe_df["sharpe"]),
                "weights": {
                    t: float(w) for t, w in zip(max_sharpe_df["tickers"], max_sharpe_df["Weight"])
                },
            },
            "min_vol": {
                "return": float(min_vol["returns"]),
                "risk": float(min_vol["risk"]),
                "sharpe": float(min_vol["sharpe"]),
                "weights": {
                    t: float(w) for t, w in zip(min_vol["tickers"], min_vol["Weight"])
                },
            },
        }
    except Exception as e:
        return {"error": f"{e}"}


@app.get("/stock/{ticker}")
async def get_stock(ticker: str):
    try:
        ticker = ticker.upper()
        #changed: reuse shared history fetching so repeated stock and history calls hit the same cache
        history_task = get_alpaca_history(ticker, period_days=365)
        quote_task = client.get(f"{BASE_URL}/stocks/{ticker}/quotes/latest", headers=HEADERS)
        history_df, quote_resp = await asyncio.gather(history_task, quote_task)
        
        if quote_resp.status_code != 200:
            return {"error": f"Alpaca API error: {quote_resp.status_code}"}
        q_data = quote_resp.json()
        if "quote" not in q_data or history_df.empty:
            return {"error": "No data found for ticker"}
        quote = q_data.get("quote")
        if not quote:
            return {"error": "No data found for this ticker"}
        #changed: read the latest daily row and extrema straight from the cached DataFrame
        daily = history_df.iloc[-1]
        max_low = float(history_df["Low"].min())
        max_high = float(history_df["High"].max())
        return {
            'ticker': ticker,
            "price": quote.get("ap"),
            "open": float(daily["Open"]),
            "high": float(daily["High"]),
            "low": float(daily["Low"]),
            "volume": float(daily["Volume"]),
            "close": float(daily["Close"]),
            "max_low": max_low,
            "max_high": max_high,
        }
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return {"error": str(e)}


@app.get("/stock/{ticker}/history")
async def hist(ticker: str, period_days: int = 730):
    try:
        #changed: serve history from the shared cached frame instead of issuing another API request
        df = await get_alpaca_history(ticker, period_days=period_days)
        if df.empty:
            return []
        dates = df["Date"].dt.strftime("%Y-%m-%d").to_list()
        closes = df["Close"].to_list()
        return [{"Date": date, "Close": close} for date, close in zip(dates, closes)]
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
        #changed: short-cache the combined macro payload and dedupe concurrent refreshes into a single upstream fan-out.
        cached, found = get_ttl_cache_value(MACRO_CACHE, "macro", RESPONSE_TTL_SECONDS)
        if found:
            return cached

        async def build_macro():
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
            set_ttl_cache_value(MACRO_CACHE, "macro", data)
            return data

        return await get_or_create_task_result(INFLIGHT_MACRO_TASKS, "macro", build_macro)
    except Exception as e:
        return {"error": str(e)}


def wrap(df):
    #changed: removed one extra DataFrame copy from the technical pipeline
    try:
        #changed: compute shared rolling features once so both simulation and ML can reuse them
        close = df["Close"]
        df["SMA_50"] = close.rolling(50).mean()
        df["SMA_100"] = close.rolling(100).mean()
        df["Volatility"] = close.pct_change().rolling(20).std()
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
    #changed: compute Sharpe directly from the close-price array to avoid extra pandas allocations.
    close = df["Close"].to_numpy(dtype=np.float64, copy=False)
    close = close[~np.isnan(close)]
    if close.size < 2:
        return np.nan
    returns = close[1:] / close[:-1] - 1
    daily_rf = (1 + risk_free) ** (1 / 252) - 1
    mean = returns.mean()
    vola = returns.std()
    if vola == 0:
        return np.nan
    sharpe_ratio = ((mean - daily_rf) / vola) * np.sqrt(252)

    return sharpe_ratio


def sim(df, drift=None):
    #changed: use typed NumPy arrays directly for the simulation math.
    close = df["Close"].to_numpy(dtype=np.float64, copy=False)
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
    #changed: compute the rolling stats once and reuse them
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
    #changed: make one explicit copy and reuse the converted Close series
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
    #changed: collapse the technical pipeline into one copy so repeated indicator work stays cache-friendly and vectorized.
    df = df.copy()
    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")
    df["Close"] = close
    df["High"] = high
    df["Low"] = low
    df["Volume"] = volume
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]
    rolling_20 = close.rolling(window=20)
    sma_20 = rolling_20.mean()
    close_std_20 = rolling_20.std()
    df["SMA_20"] = sma_20
    df["BB_Up"] = sma_20 + 2 * close_std_20
    df["BB_Down"] = sma_20 - 2 * close_std_20
    df["SMA_50"] = close.rolling(50).mean()
    df["SMA_100"] = close.rolling(100).mean()
    df["Volatility"] = close.pct_change().rolling(20).std()
    typical_price = (high + low + close) / 3
    cumulative_volume = volume.cumsum()
    df["Ty"] = typical_price
    df["Cum_TP_Vol"] = (typical_price * volume).cumsum()
    df["Cum_Vol"] = cumulative_volume
    df["VWAP"] = df["Cum_TP_Vol"] / cumulative_volume.replace(0, np.nan)
    return df

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
        #changed: reuse the cached technical frame for the ticker and keep the market benchmark fetch in parallel.
        df, spy = await asyncio.gather(
            get_technical_history(ticker.upper()),
            get_alpaca_history("SPY")
        )
        if df.empty or spy.empty:
            return {"error": f"No data found for {ticker}"}

        #changed: run the rest of the analytics in one worker thread and reuse the already-computed technical columns.
        def run_analysis(stock_df, spy_df):
            def cagr(frame):
                start = float(frame["Close"].iat[0])
                end = float(frame["Close"].iat[-1])
                days = int((frame["Date"].iat[-1] - frame["Date"].iat[0]).days)
                if days == 0 or start == 0:
                    return 0.0
                return ((end / start) ** (365.25 / days) - 1) * 100

            current_rsi = float(stock_df["RSI"].iat[-1])
            current_macd = float(stock_df["MACD"].iat[-1])
            signal_line = float(stock_df["Signal_Line"].iat[-1])
            current_price = float(stock_df["Close"].iat[-1])
            sma50 = float(stock_df["SMA_50"].iat[-1])
            sma100 = float(stock_df["SMA_100"].iat[-1])
            #changed: derive annual volatility from the raw close-price array instead of allocating another pandas Series.
            close = stock_df["Close"].to_numpy(dtype=np.float64, copy=False)
            returns = np.diff(close) / close[:-1]
            rsi_val = current_rsi
            macd_val = current_macd
            signal_line_val = signal_line
            price = current_price
            sma50_val = sma50
            bb_lower = float(stock_df["BB_Down"].iat[-1])
            bb_upper = float(stock_df["BB_Up"].iat[-1])
            annual_vol = float(returns.std() * (252**0.5) * 100) if returns.size else 0.0
            score = 0
            if price > sma50_val:
                score += 1
            else:
                score -= 1
            if macd_val > signal_line_val:
                score += 1
            else:
                score -= 1
            if price < bb_lower:
                score += 2
            if price > bb_upper:
                score -= 2
            if rsi_val < 35:
                score += 1
            if rsi_val > 65:
                score -= 1
            sig = "Neutral"
            if score >= 3:
                sig = "Strong Buy"
            elif score >= 1:
                sig = "Buy"
            elif score <= -3:
                sig = "Strong Sell"
            elif score <= -1:
                sig = "Sell"
            spy_cagr = cagr(spy_df)
            stock_cagr = cagr(stock_df)
            risk_free = 0.0422
            sharpe = float(sharpness(stock_df, risk_free))
            bull_reasons = []
            if price > sma50_val:
                bull_reasons.append("price is trading above the 50-day moving average, which supports the current uptrend")
            if macd_val > signal_line_val:
                bull_reasons.append("MACD is above the signal line, showing bullish momentum is still in place")
            if rsi_val < 35:
                bull_reasons.append("RSI is near oversold territory, which can create room for a rebound")
            if price < bb_lower:
                bull_reasons.append("price is below the lower Bollinger Band, which can signal a stretched downside move")
            if stock_cagr > spy_cagr:
                bull_reasons.append("The stock has outperformed SPY on a CAGR basis, showing stronger longer-term trend strength")
            if sharpe > 1:
                bull_reasons.append("The Sharpe ratio is healthy, which suggests recent returns have been efficient relative to risk")
            bear_reasons = []
            if price < sma50_val:
                bear_reasons.append("Price is below the 50-day moving average, which points to weak near-term trend structure")
            if macd_val < signal_line_val:
                bear_reasons.append("MACD is below the signal line, showing momentum has weakened")
            if rsi_val > 65:
                bear_reasons.append("RSI is near overbought territory, which raises the chance of a pullback")
            if price > bb_upper:
                bear_reasons.append("Price is above the upper Bollinger Band, which can signal an overheated move")
            if stock_cagr < spy_cagr:
                bear_reasons.append("The stock has lagged SPY on a CAGR basis, which weakens the relative-strength story")
            if annual_vol > 40:
                bear_reasons.append("The annualized volatility is elevated, which makes the setup more fragile")
            if sharpe < 0:
                bear_reasons.append("The Sharpe ratio is negative, meaning recent risk-adjusted performance has been poor")
            if not bull_reasons:
                bull_reasons.append("There is no standout bullish signal right now, so the upside case depends on momentum improving from here")
            if not bear_reasons:
                bear_reasons.append("There is no standout bearish signal right now, so the downside case mainly depends on trend deterioration")
            bull_case = "Bull case: " + ". ".join([r for r in bull_reasons[:3]]) + "."
            bear_case = "Bear case: " + ". ".join([r for r in bear_reasons[:3]]) + "."
            return {
                "rsi": round(current_rsi),
                "macd": round(current_macd),
                "signal_line": signal_line,
                "price": current_price,
                "sma50": sma50,
                "sma100": sma100,
                "vola": round(annual_vol),
                "rsi_signal": sig,
                "stock_cagr": stock_cagr,
                "spy_cagr": spy_cagr,
                "sharpe": sharpe,
                "bull_case": bull_case,
                "bear_case": bear_case,
            }

        return await anyio.to_thread.run_sync(run_analysis, df, spy)
    except Exception as e:
        return {"error": str(e)}


def backtest(df, buy_rsi = 30, sell_rsi=70, starter_cash = 10000):
    try:
        #changed: keep the backtest on NumPy arrays and use fast counts instead of repeated boolean indexing.
        rsi = df["RSI"].to_numpy(dtype=np.float64, copy=False)
        close = df["Close"].to_numpy(dtype=np.float64, copy=False)
        signals = np.where(rsi < buy_rsi, 1, np.where(rsi > sell_rsi, -1, 0))
        position = np.clip(np.cumsum(signals), 0, 1)
        returns = np.diff(close) / close[:-1]
        position = position[:-1]
        strat_return = position * returns
        portfolio = starter_cash * np.cumprod(1 + strat_return)
        if portfolio.size == 0:
            return {
                "portfolio": np.array([], dtype=np.float64),
                "total_return": 0.0,
                "sharpe": 0.0,
                "buy": 0,
                "sell": 0,
            }
        tot_returns = (portfolio[-1] - starter_cash) * 100 / starter_cash
        strat_std = np.std(strat_return)
        sharpe = 0.0 if strat_std == 0 else np.mean(strat_return) / strat_std * np.sqrt(252)
        buy = int(np.count_nonzero(signals == 1))
        sell = int(np.count_nonzero(signals == -1))
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
        #changed: reuse cached technical history and keep the rest of the backtest math in one worker-thread jump.
        technical_df = await get_technical_history(ticker.upper(), period_days=365 * 2)
        if technical_df.empty:
            return {"error": f"No data found for {ticker}"}

        def run_backtest_logic(df_in):
            clean_df = df_in.dropna(subset=["Close", "RSI", "SMA_50"])
            if clean_df.empty:
                raise ValueError("Not enough clean data for backtest")
            result = backtest(clean_df, buy_rsi, sell_rsi, starter_cash)
            close = clean_df["Close"].to_numpy(dtype=np.float64, copy=False)
            returns = np.diff(close) / close[:-1]
            buy_hold = starter_cash * np.cumprod(1 + returns)
            portfolio = np.asarray(result["portfolio"], dtype=np.float64)
            peak = np.maximum.accumulate(portfolio)
            drawdown = (portfolio - peak) / peak
            buy_hold_return = round((buy_hold[-1] - starter_cash) * 100 / starter_cash, 2) if buy_hold.size else 0.0
            max_drawdown = round(float(np.min(drawdown) * 100), 2) if drawdown.size else 0.0
            return {
                "total_return": round(result["total_return"], 2),
                "buy_hold_return": buy_hold_return,
                "sharpe": round(float(result["sharpe"]), 2),
                "max_drawdown": max_drawdown,
                "buy_signals": result["buy"],
                "sell_signals": result["sell"],
                "portfolio": portfolio.tolist(),
                "buy_hold": buy_hold.tolist(),
            }

        return await anyio.to_thread.run_sync(run_backtest_logic, technical_df)
    except Exception as e:
        return {"error": f"Endpoint Error: {str(e)}"}
@app.get("/stock/{ticker}/sentiment")
async def get_sentiment(ticker):
    try:
        #changed: cache ticker sentiment briefly and reuse one background thread to score all fetched articles.
        upper_ticker = ticker.upper()
        cached, found = get_ttl_cache_value(SENTIMENT_CACHE, upper_ticker, RESPONSE_TTL_SECONDS)
        if found:
            return cached

        NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
        params = {
            "symbols": upper_ticker,
            "limit": 10
        }

        async def build_sentiment():
            response = await client.get(NEWS_URL, headers=HEADERS, params=params)
            if response.status_code != 200:
                return {"error": f"Alpaca API error: {response.status_code}"}
            news_data = response.json().get("news", [])

            def score_articles(articles):
                total_score = 0.0
                articles_output = []
                for article in articles:
                    score = analyzer.polarity_scores(
                        f"{article.get('headline', '')} {article.get('summary', '')}"
                    )["compound"]
                    total_score += score
                    articles_output.append(
                        {
                            "headline": article["headline"],
                            "url": article["url"],
                            "sentiment": "Bullish" if score > 0.05 else "Bearish" if score < -0.05 else "Neutral",
                        }
                    )
                avg_score = total_score / len(articles) if articles else 0.0
                label = "Neutral"
                if avg_score > 0.15:
                    label = "Strong Bullish"
                elif avg_score > 0.05:
                    label = "Bullish"
                elif avg_score < -0.15:
                    label = "Strong Bearish"
                elif avg_score < -0.05:
                    label = "Bearish"
                return {
                    "score": round(avg_score, 2),
                    "label": label,
                    "articles": articles_output,
                }

            payload = await anyio.to_thread.run_sync(score_articles, news_data)
            set_ttl_cache_value(SENTIMENT_CACHE, upper_ticker, payload)
            return payload

        return await get_or_create_task_result(INFLIGHT_SENTIMENT_TASKS, upper_ticker, build_sentiment)
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return {"error": str(e)}
    
@app.get("/stock/{ticker}/target/{user_id}")
async def get_stock_target(ticker: str, user_id: str, db: Session = Depends(get_db)):
    ticker = ticker.upper()
    #changed: short-circuit hot reads from memory before touching the database
    cached_target, found = get_cached_target_price(user_id, ticker)
    if found:
        return {"target_price": cached_target}

    #changed: fetch only the target_price scalar instead of hydrating a full ORM object
    target_price = db.execute(
        select(models.UserStockTarget.target_price).where(
            models.UserStockTarget.user_id == user_id,
            models.UserStockTarget.ticker == ticker
        )
    ).scalar_one_or_none()

    #changed: backfill the cache after a database read so repeated requests stay fast
    set_cached_target_price(user_id, ticker, target_price)
    return {"target_price": target_price}
@app.websocket("/ws/shield")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # This keeps the pipe open. We can also receive commands 
            # from the frontend here (like "Switch to TSLA")
            data = await websocket.receive_text()
            # For now, we just acknowledge receipt (ping-pong)
            await websocket.send_text(f"Shield Heartbeat: Received {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)