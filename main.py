from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, Request, APIRouter
import weasyprint
from fastapi.responses import JSONResponse
import tempfile
from fastapi.responses import FileResponse
import datetime
import time
import numpy as np
import redis
import pandas as pd
from dotenv import load_dotenv
import os
import websockets
import httpx
from concurrent.futures import ProcessPoolExecutor
import anyio
from sqlalchemy.orm import Session
import asyncio
from contextlib import asynccontextmanager, suppress
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from ml import get_ml_predictions
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fastapi.middleware.cors import CORSMiddleware
executor = ProcessPoolExecutor(max_workers=2)
#changed: use numba for the hottest numeric loops when available, while keeping a no-extra-dependency fallback.
try:
    from numba import njit, prange

    NUMBA_ENABLED = True
except ImportError:
    NUMBA_ENABLED = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

#changed: use uvloop when available so FastAPI, websockets, and httpx run on a faster event loop without changing endpoint code.
try:
    import uvloop

    uvloop.install()
except ImportError:
    uvloop = None

load_dotenv()
from database import SessionLocal
import models
#changed: avoid the deprecated ORJSONResponse wrapper by using a tiny JSONResponse subclass backed by orjson when available.
class ConnectionManager:
    def __init__(self):
        #changed: store sockets in a set so connect/disconnect stay O(1) as the audience grows.
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
       
    def disconnect(self, websocket: WebSocket):
        #changed: discard avoids raising when a socket has already been cleaned up elsewhere.
        self.active_connections.discard(websocket)
        

    async def broadcast_tick(self, tick_data: dict):
        """Sends a live price tick to EVERY connected user"""
        #changed: bail out early for empty audiences and skip gather overhead for the common single-client case.
        if not self.active_connections:
            return
        # We use orjson (already in your code) for ultra-fast serialization
        message = orjson.dumps(tick_data)
        if len(self.active_connections) == 1:
            connection = next(iter(self.active_connections))
            try:
                await connection.send_bytes(message)
            except Exception:
                self.active_connections.discard(connection)
            return
        #changed: fan out writes concurrently and prune dead sockets immediately so one slow client does not stall the rest.
        connections = tuple(self.active_connections)
        results = await asyncio.gather(
            *(connection.send_bytes(message) for connection in connections),
            return_exceptions=True,
        )
        for connection, result in zip(connections, results):
            if isinstance(result, Exception):
                self.active_connections.discard(connection)
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
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: client is already created globally or create it here
    stream_task = asyncio.create_task(alpaca_to_shield_bridge())
  
    try:
        yield
    finally:
        #changed: cancel the background bridge on shutdown so it does not linger past app teardown.
        stream_task.cancel()
        with suppress(asyncio.CancelledError):
            await stream_task
        await client.aclose()
#changed: use the custom fast JSON response application-wide when orjson is installed.
class FastORJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return orjson.dumps(
            content, 
            # This is the "magic" that makes it fast
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_OMIT_MICROSECONDS
        )

pool = redis.ConnectionPool.from_url(
    os.getenv("REDIS_URL"), 
    max_connections=20, 
    decode_responses=True
)
db = redis.Redis(connection_pool=pool)  
# This pulls the URL from the Render Environment safely
REDIS_URL = os.getenv("REDIS_URL")

# If you use the rediss:// URL, Upstash handles the password automatically
pool = redis.ConnectionPool.from_url(REDIS_URL, max_connections=20)
db = redis.Redis(connection_pool=pool)

# 1. Setup your connection
# Note: Use decode_responses=False if you want to handle the raw bytes from orjson
pool = redis.ConnectionPool.from_url(os.getenv("REDIS_URL"), max_connections=20)
db = redis.Redis(connection_pool=pool)

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
    #changed: skip expensive numeric coercion when history data is already typed coming out of the shared fetch cache.
    close = df["Close"] if pd.api.types.is_float_dtype(df["Close"]) else pd.to_numeric(df["Close"], errors="coerce")
    high = df["High"] if pd.api.types.is_float_dtype(df["High"]) else pd.to_numeric(df["High"], errors="coerce")
    low = df["Low"] if pd.api.types.is_float_dtype(df["Low"]) else pd.to_numeric(df["Low"], errors="coerce")
    volume = df["Volume"] if pd.api.types.is_float_dtype(df["Volume"]) else pd.to_numeric(df["Volume"], errors="coerce")
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

app = FastAPI(lifespan=lifespan, default_response_class=FastORJSONResponse)
async def analyse_stock(ticker: str):
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
@app.get("/analyze/{ticker}")
async def get_analysis_endpoint(ticker: str):
    ticker = ticker.upper()
    cache_key = f"analysis:{ticker}"

    # 1. Use your new helper!
    cached_data = await get_from_cache(cache_key)
    if cached_data:
        return cached_data

    # 2. Run the math
    analysis_results = await analyse_stock(ticker) 

    # 3. Save using your new helper!
    if "error" not in analysis_results:
        await save_to_cache(cache_key, analysis_results, ttl=600)

    return analysis_results

async def get_processed_metrics(ticker: str):
    try:
    # Ensure the ticker is always uppercase so 'asml' and 'ASML' use the same cache

    # Simply call your endpoint logic to avoid repeating code
        return await get_analysis_endpoint(ticker)
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return {"error": "Could not retrieve market data"}
async def get_from_cache(key: str):
    """Checks Redis for data and returns it as a Python object."""
    try:
        data = db.get(key)
        if data:
            return orjson.loads(data)
    except Exception as e:
        print(f"Cache lookup error: {e}")
    return None

async def save_to_cache(key: str, value: any, ttl=600):
    """Saves any object (including NumPy/Pandas results) to Redis."""
    try:
        db.setex(
            key, 
            ttl, 
            orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY)
        )
    except Exception as e:
        print(f"Cache save error: {e}")
# This software is provided as-is for educational and research purposes. The developer is not responsible for any financial losses incurred through the use of this code.
#changed: added a small in-memory cache for repeated history requests
HISTORY_TTL_SECONDS = 30

#changed: cache repeated portfolio and macro responses for short bursts of traffic
RESPONSE_TTL_SECONDS = 30
PORTFOLIO_CACHE = {}
#changed: cache computed technical frames and sentiment payloads so repeated requests avoid redoing CPU-heavy work.

#changed: cache simulate endpoint responses to avoid repeated CPU-heavy Monte Carlo and ML work.
SIMULATE_CACHE_TTL_SECONDS = 60
#changed: dedupe in-flight simulate tasks for identical concurrent requests.
INFLIGHT_SIMULATE_TASKS = {}
#changed: cache hot user target lookups to avoid repeated round-trips for the same key
TARGET_CACHE_TTL_SECONDS = 60
TARGET_CACHE = {}
#changed: index cached targets by ticker so the live stream only checks relevant alerts per symbol.
TARGETS_BY_TICKER = {}
#changed: reuse one shared feature list anywhere we need fully-computed technical columns
TECHNICAL_REQUIRED_COLUMNS = ["RSI", "MACD", "SMA_50", "SMA_100", "Volatility"]
#changed: dedupe identical in-flight async work so concurrent traffic shares one fetch or calculation.
INFLIGHT_HISTORY_TASKS = {}
INFLIGHT_PORTFOLIO_TASKS = {}
INFLIGHT_TECHNICAL_TASKS = {}
INFLIGHT_SENTIMENT_TASKS = {}
INFLIGHT_MACRO_TASKS = {}
#changed: reuse one shared monotonic clock binding so hot-path cache reads avoid repeated global lookups.
MONOTONIC = time.monotonic
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
            async with websockets.connect(
    url, 
    ping_interval=20, 
    ping_timeout=20
            ) as ws:
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
                                #changed: await the lightweight in-memory alert check directly to avoid spawning a task per tick.
                                await check_shield_activation(ticker, current_price)

        except Exception as e:
            print(f"🛡️  Shield Bridge Error: {e}. Retrying...")
            await asyncio.sleep(5)
LAST_ALERT_TIME = {}
async def check_shield_activation(ticker: str, current_price: float):
    #changed: read only the ticker-specific target bucket so live ticks do not scan the full cache.
    ticker_targets = TARGETS_BY_TICKER.get(ticker)
    if not ticker_targets:
        return

    for user_id, target_price in tuple(ticker_targets.items()):
        #changed: prune expired target cache entries in-line and replace division with a cheaper range check.
        cached_target, found = get_cached_target_price(user_id, ticker)
        if not found or cached_target is None:
            continue
        target_price = cached_target
        if target_price <= current_price <= target_price * 1.01:
            alert_key = (user_id, ticker)
            now = MONOTONIC()
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

analyzer = SentimentIntensityAnalyzer()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", 'https://talos-ui-ten.vercel.app'],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = httpx.AsyncClient(
    #changed: allow HTTP/2 reuse when the upstream supports it to reduce request overhead on repeated API calls.
    http2=True,
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
    now_ts = MONOTONIC()
    if now_ts - cached[0] >= TARGET_CACHE_TTL_SECONDS:
        #changed: keep the secondary ticker index in sync when a hot target cache entry expires.
        set_cached_target_price(user_id, ticker, None)
        return None, False
    return cached[1], True

#changed: keep target cache writes tiny and explicit
def set_cached_target_price(user_id: str, ticker: str, target_price):
    cache_key = (user_id, ticker)
    if target_price is None:
        TARGET_CACHE.pop(cache_key, None)
        targets_for_ticker = TARGETS_BY_TICKER.get(ticker)
        if targets_for_ticker is not None:
            targets_for_ticker.pop(user_id, None)
            if not targets_for_ticker:
                TARGETS_BY_TICKER.pop(ticker, None)
        return
    TARGET_CACHE[cache_key] = (MONOTONIC(), target_price)
    TARGETS_BY_TICKER.setdefault(ticker, {})[user_id] = target_price

#changed: standardize TTL cache reads so the hot caches share one expiration path.
def get_ttl_cache_value(cache: dict, cache_key, ttl_seconds: float):
    cached = cache.get(cache_key)
    if not cached:
        return None, False
    now_ts = MONOTONIC()
    if now_ts - cached[0] >= ttl_seconds:
        cache.pop(cache_key, None)
        return None, False
    return cached[1], True

#changed: keep generic TTL cache writes tiny and reusable for data, analytics, and response payloads.
def set_ttl_cache_value(cache: dict, cache_key, value):
    cache[cache_key] = (MONOTONIC(), value)

#changed: trim the leading indicator warmup rows with one contiguous slice instead of copying via dropna on every endpoint call.
def get_valid_technical_slice(df: pd.DataFrame, required_columns: list[str]):
    required = df[required_columns].to_numpy(dtype=np.float64, copy=False)
    valid_mask = np.isfinite(required).all(axis=1)
    if not valid_mask.any():
        return df.iloc[0:0]
    first_valid_idx = int(np.argmax(valid_mask))
    return df.iloc[first_valid_idx:]

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

async def get_alpaca_history(ticker, timeframe="1Day", period_days=365, start_override=None, end_override=None):
    #changed: reuse recent ticker history instead of refetching immediately.
    upper_ticker = ticker.upper()
    # Create a unique key for this specific request
    cache_key = f"hist:{upper_ticker}:{timeframe}:{period_days}:{start_override}:{end_override}"
    
    # --- NEW REDIS CHECK ---
    # We check Redis instead of the HISTORY_CACHE dictionary
    cached_data = await get_from_cache(cache_key)
    if cached_data:
        # We store DataFrames as lists of dicts in Redis. 
        # This converts them back into a DataFrame for your math.
        df = pd.DataFrame(cached_data)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date']) # Ensure dates are correct
        return df

    async def fetch_history():
        # LOGIC: If we have an override, use it; otherwise, look back from now
        if start_override:
            start_date = start_override
        else:
            start_date = (datetime.datetime.now() - datetime.timedelta(days=period_days)).date().isoformat()
            
        url = f"{BASE_URL}/stocks/{upper_ticker}/bars?timeframe={timeframe}&start={start_date}"
        
        # Add end_date if we are targeting a specific historical window
        if end_override:
            url += f"&end={end_override}"
            
        url += "&adjustment=all"
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
        if not df.empty:
            # --- NEW REDIS SAVE ---
            # Save the DataFrame to Redis so other functions can use it
            await save_to_cache(
                cache_key, 
                df.to_dict(orient="records"), 
                ttl=300 # Cache for 5 minutes
            )
        return df

    return await get_or_create_task_result(INFLIGHT_HISTORY_TASKS, cache_key, fetch_history)

async def get_black_swan_signature(ticker):
    # March 2020 COVID Window
    crash_df = await get_alpaca_history(
        ticker, 
        start_override="2020-02-01", 
        end_override="2020-05-01"
    )
    
    if crash_df.empty:
        return None

    # Calculate daily returns during the crash
    returns = crash_df['Close'].pct_change().dropna().values
    
    # THE SIGNATURE:
    # 1. Max Drawdown (The 'Depth' of the swan)
    peak = crash_df['Close'].max()
    trough = crash_df['Close'].min()
    max_drawdown = (trough - peak) / peak
    
    # 2. Realized Volatility (The 'Chaos' during the swan)
    realized_vol = np.std(returns) * np.sqrt(252) # Annualized
    
    return {
        "max_drawdown": float(max_drawdown),
        "crash_volatility": float(realized_vol),
        "recovery_speed": len(crash_df) # How many days it took
    }

@njit(parallel=True)
def run_black_swan_simulation(current_price, days, mu, sigma, shock_factor, num_sims=1000):
    """
    current_price: Starting price today
    days: How long the simulation runs (e.g., 30 or 60 days)
    mu: Historical average return (drift)
    sigma: Historical volatility (the 'Signature' you just extracted)
    shock_factor: Multiplier for volatility (usually 1.5 to 2.0 for a crash)
    """
    dt = 1 / 252  # Time step (daily)
    results = np.zeros((num_sims, days))
    
    for s in prange(num_sims):
        prices = np.zeros(days)
        prices[0] = current_price
        for t in range(1, days):
            # The 'Shock' is applied to the volatility component
            epsilon = np.random.normal(0, 1)
            drift = (mu - 0.5 * (sigma**2)) * dt
            diffusion = (sigma * shock_factor) * np.sqrt(dt) * epsilon
            
            prices[t] = prices[t-1] * np.exp(drift + diffusion)
        results[s, :] = prices
        
    return results

@app.get("/stock/{ticker}/black-swan")
async def black_swan(ticker: str):
    try:
        signature = await get_black_swan_signature(ticker)
        if signature is None:
            return {"error": "Not enough data for black swan analysis"}
        df_recent = await get_alpaca_history(ticker, period_days=5)
        if df_recent.empty:
            return {"error": "Not enough recent data for simulation"}
        current_price = df_recent['Close'].iat[-1]
        sim_paths = run_black_swan_simulation(
            current_price=float(current_price),
            days=30,
            mu=-0.05,  # We assume a negative drift during a crash
            sigma=signature['crash_volatility'],
            shock_factor=1.5,
            num_sims=500
    )
        worst_case_path = np.percentile(sim_paths, 5, axis=0)  # 5th percentile for worst-case scenario
        return {
            "ticker": ticker,
            "stress_label": "COVID-19 Impact Overlay",
            "historical_drawdown": signature['max_drawdown'],
            "projected_path": worst_case_path.tolist(),
            "vaR_percent": float((worst_case_path[-1] - current_price) / current_price)
        }
    except Exception as e:
        import logging
        logging.error(f"Black Swan Analysis Error for {ticker}: {e}")
        return {"error": str(e)}
#changed: cache fully-computed technical frames so simulation, analysis, and backtests reuse the same indicator work.
async def get_technical_history(ticker: str):
    ticker = ticker.upper()
    cache_key = f"tech_df:{ticker}"

    # 1. Check Redis for the ALREADY CALCULATED indicators
    cached = await get_from_cache(cache_key)
    if cached:
        # Convert back to DataFrame so your score logic works
        return pd.DataFrame(cached)

    # 2. If not in Redis, get the raw history (This now uses the Redis cache you just built!)
    df = await get_alpaca_history(ticker)
    
    if df.empty:
        return df

    # 3. Run the math (CPU-heavy part)
    # This uses your run_all_technicals function
    df = run_all_technicals(df)

    # 4. Save the "Finished Product" to Redis
    # We save this for 300 seconds (5 mins)
    await save_to_cache(
        cache_key, 
        df.to_dict(orient="records"), 
        ttl=300
    )
    
    return df


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
def calculate_prediction_cone(close_prices, drift, days=30):
    # Pure math - runs in microseconds
    last_price = float(close_prices[-1])
    returns = np.diff(close_prices) / close_prices[:-1]
    vola = np.std(returns)
    
    # Time array (1 to 30)
    t = np.arange(1, days + 1)
    
    # 1. Expected Price (Median/p50)
    # Using the compounding drift over time
    p50 = last_price * (1 + drift)**t
    
    # 2. Volatility Expansion (The "Cone")
    # 1.645 is the Z-score for a 90% confidence interval (5% to 95%)
    # The width of the cone grows with the square root of time
    margin = 1.645 * vola * np.sqrt(t)
    
    # Calculate boundaries using exponential growth/decay
    p5 = p50 * np.exp(-margin)
    p95 = p50 * np.exp(margin)
    
    return p5, p50, p95

def calculate_probability(target_price, last_price, drift, vola, days=30):
    if not target_price:
        return 0
    # Standard normal distribution formula for price probability
    # ln(Target/Last) - (Drift - 0.5 * Vola^2) * Days / (Vola * sqrt(Days))
    from scipy.stats import norm
    d2 = (np.log(target_price / last_price) - (drift - 0.5 * vola**2) * days) / (vola * np.sqrt(days))
    prob = (1 - norm.cdf(d2)) * 100
    return round(max(0, min(100, prob)), 2)

# --- 3. FASTAPI ENDPOINT ---
@app.get("/stock/{ticker}/simulate")
async def simulate(ticker: str, target_price: float = None):
    try:
        ticker_upper = ticker.upper()
        # 1. Redis Cache Check
        # We include target_price in the key because the probability changes based on it
        cache_key = f"sim:{ticker_upper}:{target_price}"
        
        cached = await get_from_cache(cache_key)
        if cached:
            return cached

        # 2. Build the Simulation (The "Slow" Work)
        async def run_simulation():
            # This uses your Redis-backed history fetcher!
            df = await get_alpaca_history(ticker_upper)
            if df is None or df.empty:
                return {"error": f"No data found for {ticker}"}

            # 3. Technical Indicators (Note: Consider using your run_all_technicals(df) here)
            df = rsi(df)
            df = macd(df)
            df["SMA_50"] = df["Close"].rolling(window=50).mean()
            df["SMA_100"] = df["Close"].rolling(window=100).mean()
            df["Volatility"] = df["Close"].pct_change().rolling(window=20).std()
            df_clean = df.dropna(subset=["RSI", "MACD", "SMA_50", "SMA_100", "Volatility"])

            if len(df_clean) < 10:
                return {"error": "Insufficient data for simulation"}

            # 4. ML Prediction 
            predicted_price = get_ml_predictions(df_clean, ticker_upper)
            current_price = float(df["Close"].iat[-1])
            
            ml_total_return = (predicted_price - current_price) / current_price
            drift = ml_total_return / 30

            # 5. Analytical Math
            close_array = df_clean["Close"].to_numpy(dtype=np.float32)
            returns = np.diff(close_array) / close_array[:-1]
            vola = np.std(returns)

            p5, p50, p95 = calculate_prediction_cone(close_array, drift)
            prob = calculate_probability(target_price, current_price, drift, vola)

            # 6. Payload Generation
            payload = [
                {
                    "Date": i + 1,
                    "p5": round(float(p5[i]), 2),
                    "p50": round(float(p50[i]), 2),
                    "p95": round(float(p95[i]), 2)
                }
                for i in range(30)
            ]

            return {
                "data": payload,
                "probability": f"{prob}%",
                "ml_expected_price": round(predicted_price, 2)
            }

        # 7. Execute with In-Flight de-duplication
        result = await get_or_create_task_result(INFLIGHT_SIMULATE_TASKS, cache_key, run_simulation)

        # 8. Save to Redis
        # Simulation is heavy, so we cache it for 10 minutes (600s)
        if "error" not in result:
            await save_to_cache(cache_key, result, ttl=600)

        return result

    except Exception as e:
        import logging
        logging.error(f"Calculation Error for {ticker}: {e}")
        return {"error": f"Calculation Error: {str(e)}"}
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

@app.get("/macro/{series_id}")
async def get_macro_data(series_id: str):
    try:
        # 1. Normalize the key
        series_id = series_id.upper()
        cache_key = f"macro:{series_id}"
    
        # 2. Check Redis (The bookshelf)
        cached = await get_from_cache(cache_key)
        if cached:
            return cached

        # 3. Logic to fetch from FRED if not cached
        async def fetch_macro():
            # Example FRED URL
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={fred_key}&file_type=json"
            response = await client.get(url)
            
            if response.status_code != 200:
                return {"error": f"FRED API error: {response.status_code}"}
                
            data = response.json()
            # Clean up the data here if needed (e.g., getting the latest observation)
            return data

        # 4. Use In-Flight logic (to prevent "Thundering Herd" on startup)
        payload = await get_or_create_task_result(INFLIGHT_MACRO_TASKS, cache_key, fetch_macro)
        
        # 5. Save to Redis for 24 hours (86400 seconds)
        if "error" not in payload:
            await save_to_cache(cache_key, payload, ttl=86400)
    
        return payload

    except Exception as e:
        print(f"Macro Error for {series_id}: {e}")
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


#changed: JIT-compile the Monte Carlo core so repeated simulations spend less time inside Python loops.
@njit(cache=True, fastmath=True)
def _simulate_price_paths(last_price, expected_return, volatility, days, path_count):
    price_path = np.empty((days, path_count), dtype=np.float64)
    for path_idx in range(path_count):
        running_price = last_price
        for day_idx in range(days):
            running_price *= 1.0 + np.random.normal(expected_return, volatility)
            price_path[day_idx, path_idx] = running_price
    return price_path


#changed: compute all percentiles in one pass for better cache efficiency.
@njit(cache=True, fastmath=True)
def _compute_percentiles(price_path):
    days = price_path.shape[0]
    p5 = np.empty(days, dtype=np.float64)
    p50 = np.empty(days, dtype=np.float64)
    p95 = np.empty(days, dtype=np.float64)
    
    for day_idx in range(days):
        sorted_prices = np.sort(price_path[day_idx, :])
        n = sorted_prices.size
        p5[day_idx] = sorted_prices[int(n * 0.05)]
        p50[day_idx] = sorted_prices[int(n * 0.50)]
        p95[day_idx] = sorted_prices[int(n * 0.95)]
    
    return p5, p50, p95



#changed: JIT-compile the strategy loop so repeated backtests avoid Python overhead on signal and portfolio updates.
@njit(cache=True, fastmath=True)
def _run_backtest_kernel(rsi, close, buy_rsi, sell_rsi, starter_cash):
    trade_count = close.size - 1
    if trade_count <= 0:
        return np.empty(0, dtype=np.float64), 0.0, 0.0, 0, 0

    portfolio = np.empty(trade_count, dtype=np.float64)
    strat_returns = np.empty(trade_count, dtype=np.float64)
    cash_value = starter_cash
    position = 0
    buy_count = 0
    sell_count = 0

    for idx in range(trade_count):
        signal = 0
        if rsi[idx] < buy_rsi:
            signal = 1
            buy_count += 1
        elif rsi[idx] > sell_rsi:
            signal = -1
            sell_count += 1

        position += signal
        if position < 0:
            position = 0
        elif position > 1:
            position = 1

        day_return = position * (close[idx + 1] / close[idx] - 1.0)
        strat_returns[idx] = day_return
        cash_value *= 1.0 + day_return
        portfolio[idx] = cash_value

    total_return = (portfolio[-1] - starter_cash) * 100.0 / starter_cash
    mean_return = strat_returns.mean()
    std_return = strat_returns.std()
    sharpe = 0.0 if std_return == 0.0 else mean_return / std_return * np.sqrt(252.0)
    return portfolio, total_return, sharpe, buy_count, sell_count


def backtest(df, buy_rsi = 30, sell_rsi=70, starter_cash = 10000):
    try:
        #changed: keep the backtest on NumPy arrays and use fast counts instead of repeated boolean indexing.
        rsi = df["RSI"].to_numpy(dtype=np.float64, copy=False)
        close = df["Close"].to_numpy(dtype=np.float64, copy=False)
        #changed: route the heavy numeric work through numba when present and preserve the existing return shape.
        if NUMBA_ENABLED:
            portfolio, tot_returns, sharpe, buy, sell = _run_backtest_kernel(
                rsi, close, buy_rsi, sell_rsi, starter_cash
            )
        else:
            signals = np.where(rsi < buy_rsi, 1, np.where(rsi > sell_rsi, -1, 0))
            position = np.clip(np.cumsum(signals), 0, 1)
            returns = np.diff(close) / close[:-1]
            position = position[:-1]
            strat_return = position * returns
            portfolio = starter_cash * np.cumprod(1 + strat_return)
            tot_returns = (portfolio[-1] - starter_cash) * 100 / starter_cash if portfolio.size else 0.0
            strat_std = np.std(strat_return)
            sharpe = 0.0 if strat_std == 0 else np.mean(strat_return) / strat_std * np.sqrt(252)
            buy = int(np.count_nonzero(signals == 1))
            sell = int(np.count_nonzero(signals == -1))
        if portfolio.size == 0:
            return {
                "portfolio": np.array([], dtype=np.float64),
                "total_return": 0.0,
                "sharpe": 0.0,
                "buy": 0,
                "sell": 0,
            }
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
            #changed: reuse the contiguous valid indicator slice here too so backtests avoid another full-frame copy.
            clean_df = get_valid_technical_slice(df_in, ["Close", "RSI", "SMA_50"])
            if clean_df.empty:
                raise ValueError("Not enough clean data for backtest")
            if len(clean_df) < 2:
                raise ValueError("Insufficient data after cleaning indicators")
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
                "total_return": float(result["total_return"]),
                "buy_hold_return": float(buy_hold_return),
                "sharpe": float(result["sharpe"]),
                "max_drawdown": float(max_drawdown),
                #changed: read the existing compact signal counts returned by backtest instead of reindexing missing keys.
                "buy_signals": int(result["buy"]),
                "sell_signals": int(result["sell"]),
                #changed: rely on NumPy's built-in list conversion, which is faster than a Python float loop here.
                "portfolio": portfolio.tolist(),
                "buy_hold": buy_hold.tolist(),
            }

        return await anyio.to_thread.run_sync(run_backtest_logic, technical_df)
    except Exception as e:
        return {"error": f"Endpoint Error: {str(e)}"}
@app.get("/stock/{ticker}/sentiment")
async def get_sentiment(ticker: str):
    try:
        upper_ticker = ticker.upper()
        cache_key = f"sentiment:{upper_ticker}"
    
        # 1. Check Redis (The "Silent Operator")
        cached = await get_from_cache(cache_key)
        if cached:
            return cached

        # 2. Fetch from Alpaca if not in cache
        NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
        params = {"symbols": upper_ticker, "limit": 10}

        async def build_sentiment():
            response = await client.get(NEWS_URL, headers=HEADERS, params=params)
            if response.status_code != 200:
                return {"error": f"Alpaca API error: {response.status_code}"}
            
            news_data = response.json().get("news", [])
            if not news_data:
                return {"score": 0, "label": "No News Found", "articles": []}

            def score_articles(articles):
                total_score = 0.0
                articles_output = []
                for article in articles:
                    # Combine headline and summary for better context
                    text = f"{article.get('headline', '')} {article.get('summary', '')}"
                    score = analyzer.polarity_scores(text)["compound"]
                    total_score += score
                    
                    articles_output.append({
                        "headline": article["headline"],
                        "url": article["url"],
                        "sentiment": "Bullish" if score > 0.05 else "Bearish" if score < -0.05 else "Neutral",
                    })
                
                avg_score = total_score / len(articles)
                label = "Neutral"
                if avg_score > 0.15: label = "Strong Bullish"
                elif avg_score > 0.05: label = "Bullish"
                elif avg_score < -0.15: label = "Strong Bearish"
                elif avg_score < -0.05: label = "Bearish"
                
                return {
                    "score": round(avg_score, 2),
                    "label": label,
                    "articles": articles_output,
                }

            # Run CPU-intensive scoring in a separate thread
            return await anyio.to_thread.run_sync(score_articles, news_data)

        # 3. Use your In-Flight task logic to prevent double-fetching
        payload = await get_or_create_task_result(INFLIGHT_SENTIMENT_TASKS, cache_key, build_sentiment)
        
        # 4. Save to Redis for 1 hour (3600s)
        if "error" not in payload:
            await save_to_cache(cache_key, payload, ttl=3600)
    
        return payload

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
@njit
def _calculate_performance(prices, signals):
    # 1. Calculate daily percentage changes
    # Equivalent to (prices[1:] / prices[:-1]) - 1
    returns = np.zeros(len(prices) - 1)
    for i in range(len(prices) - 1):
        returns[i] = (prices[i+1] - prices[i]) / prices[i]
    
    # 2. Apply signals (shift signals by 1 so we don't cheat by knowing today's price)
    strategy_returns = returns * signals[:-1]
    
    # 3. Calculate metrics
    total_return = np.sum(strategy_returns)
    
    # Sharpe Ratio: Mean / Std Dev (annualized)
    if np.std(strategy_returns) == 0:
        return 0.0
    
    sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)
    
    return sharpe # We optimize for Sharpe because it rewards stability
@njit
def _calculate_rsi_numba(prices, period):
    n = len(prices)
    rsi = np.full(n, np.nan)
    
    # Calculate price changes (deltas)
    deltas = np.zeros(n - 1)
    for i in range(n - 1):
        deltas[i] = prices[i+1] - prices[i]
    
    # Calculate initial average gain/loss
    up = 0.0
    down = 0.0
    for i in range(period):
        if deltas[i] > 0:
            up += deltas[i]
        else:
            down += -deltas[i]
    
    up /= period
    down /= period
    
    # First RSI value
    if down == 0: rsi[period] = 100
    else: rsi[period] = 100 - 100 / (1 + up / down)

    # Smooth the rest (Wilder's Smoothing or Simple Moving Average)
    for i in range(period + 1, n):
        delta = deltas[i - 1]
        up_val = delta if delta > 0 else 0.0
        down_val = -delta if delta < 0 else 0.0
            
        up = (up * (period - 1) + up_val) / period
        down = (down * (period - 1) + down_val) / period
        
        if down == 0: rsi[i] = 100
        else: rsi[i] = 100 - 100 / (1 + up / down)
        
    return rsi
@njit
def _generate_rsi_signals(prices, period, low_threshold, high_threshold):
    period = int(period) # Ensure period is an integer for numba
    n = len(prices)
    signals = np.zeros(n)
    rsi_values = _calculate_rsi_numba(prices, period) # You'll need this helper too!
    
    current_position = 0 # 0 = Cash, 1 = Long (Holding Stock)
    
    for i in range(period, n):
        rsi = rsi_values[i]
        
        # LOGIC: Buy if RSI crosses below the low threshold
        if rsi < low_threshold and current_position == 0:
            current_position = 1
            
        # LOGIC: Sell if RSI crosses above the high threshold
        elif rsi > high_threshold and current_position == 1:
            current_position = 0
            
        signals[i] = current_position
        
    return signals
@njit(parallel=True)
def run_rsi_search(close_prices, period, rsi_low_range, rsi_high_range):
    results = np.zeros((len(rsi_low_range), len(rsi_high_range)))
    for i in prange(len(rsi_low_range)): # Parallel
        low_val = rsi_low_range[i]
        for j in range(len(rsi_high_range)): # Regular range here!
            high_val = rsi_high_range[j]
            current_signals = _generate_rsi_signals(close_prices, period, low_val, high_val)
            results[i, j] = _calculate_performance(close_prices, current_signals)

    return results


@app.post("/optimize")
async def optimize_strategy(request: Request):
    try:
        # 1. Manually pull the data from the React JSON envelope
        data = await request.json()
        ticker = data.get("ticker")
        period = int(data.get("period", 14)) # Numba NEEDS this to be an int

        if not ticker:
            return {"error": "Ticker is required"}

        # 2. Get data
        df = await get_alpaca_history(ticker, period_days=365)  # ✅ We can reuse the existing history fetch logic, which is already cached and optimized.
        

        # 3. Prepare data for Numba (Force float64)
        prices = df["Close"].values.astype(np.float64)
        low_range = np.arange(20, 41, 1).astype(np.float64)
        high_range = np.arange(60, 81, 1).astype(np.float64)

        # 4. Run the high-speed engine
        grid = run_rsi_search(prices, period, low_range, high_range)
        best_idx = np.unravel_index(np.argmax(grid), grid.shape)
        
        return {
            "best_low": float(low_range[best_idx[0]]),
            "best_high": float(high_range[best_idx[1]]),
            "max_sharpe": float(grid[best_idx]),
            "heatmap": grid.tolist()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


router = APIRouter()

@router.post("/generate-report")
async def generate_report(data: dict):
    # 'data' will contain ticker, price, cagr, etc., sent from Next.js
    
    # 1. Create your HTML string (use the template I gave you)
    # You can use Jinja2 here to inject the 'data' values into the HTML
    html_content = f"""
    <html>
        <body>
            <h1>Investment Report: {data['ticker']}</h1>
            <p>Price: {data['price']}</p>
            </body>
    </html>
    """

    # 2. Generate PDF in a temporary file
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    weasyprint.HTML(string=html_content).write_pdf(temp_pdf.name)
    
    # 3. Return the file to the user
    return FileResponse(
        temp_pdf.name, 
        media_type='application/pdf', 
        filename=f"{data['ticker']}_Report.pdf"
    )