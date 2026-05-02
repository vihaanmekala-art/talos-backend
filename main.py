from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, Request, APIRouter, HTTPException
from fastapi.responses import JSONResponse
import tempfile
from fastapi.responses import FileResponse
import datetime
import time
import numpy as np
import redis
from hmmlearn import hmm
import pandas as pd
import polars as pl
from dotenv import load_dotenv
import os
import websockets
import httpx
from groq import Groq, AsyncGroq
from concurrent.futures import ProcessPoolExecutor
from persona import TECHNICAL_ANALYST_PROMPT, MACRO_STRATEGIST_PROMPT, RISK_MANAGER_PROMPT
import anyio
from sqlalchemy.orm import Session
import re
import asyncio
from contextlib import asynccontextmanager, suppress
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from ml import get_ml_predictions
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fastapi.middleware.cors import CORSMiddleware
from scipy.stats import norm
from models import BoardroomSession
executor = ProcessPoolExecutor(max_workers=2)

load_dotenv()

try:
    from numba import njit, prange

    NUMBA_ENABLED = True
except ImportError:
    NUMBA_ENABLED = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


# changed: use uvloop when available so FastAPI, websockets, and httpx run on a faster event loop without changing endpoint code.
try:
    import uvloop

    uvloop.install()
except ImportError:
    uvloop = None

from database import SessionLocal
import models


# changed: avoid the deprecated ORJSONResponse wrapper by using a tiny JSONResponse subclass backed by orjson when available.
class ConnectionManager:
    def __init__(self):
        # changed: store sockets in a set so connect/disconnect stay O(1) as the audience grows.
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        # changed: discard avoids raising when a socket has already been cleaned up elsewhere.
        self.active_connections.discard(websocket)

    async def broadcast_tick(self, tick_data: dict):
        """Sends a live price tick to EVERY connected user"""
        # changed: bail out early for empty audiences and skip gather overhead for the common single-client case.
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
        # changed: fan out writes concurrently and prune dead sockets immediately so one slow client does not stall the rest.
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

    # changed: keep fast JSON serialization without depending on FastAPI's deprecated ORJSONResponse helper.
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
        # changed: cancel the background bridge on shutdown so it does not linger past app teardown.
        stream_task.cancel()
        with suppress(asyncio.CancelledError):
            await stream_task
        await client.aclose()


# changed: use the custom fast JSON response application-wide when orjson is installed.
class FastORJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return orjson.dumps(
            content,
            # This is the "magic" that makes it fast
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_OMIT_MICROSECONDS,
        )


REDIS_URL = os.getenv("REDIS_URL")
try:
    # Adding socket_timeout=5 and ssl_cert_reqs=None handles 99% of cloud connection crashes
    pool = redis.ConnectionPool.from_url(
        REDIS_URL,
        max_connections=20,
        decode_responses=True,
        socket_timeout=5,
    )
    db = redis.Redis(connection_pool=pool)
    # This is critical: if it can't ping Redis, it won't kill the whole app
    db.ping()
    print("✅ Redis Connected")
except Exception as e:
    print(f"❌ Redis Connection Failed: {e}")
    db = None  # App will still run, just without cache
# 1. Setup your connection
# Note: Use decode_responses=False if you want to handle the raw bytes from orjson
pool = redis.ConnectionPool.from_url(os.getenv("REDIS_URL"), max_connections=20)
db = redis.Redis(connection_pool=pool)


def bollinger(df, window=20, num_std=2):
    frame = ensure_polars_frame(df)
    return frame.lazy().with_columns(
        [
            pl.col("Close").cast(pl.Float64, strict=False),
            pl.col("Close").cast(pl.Float64, strict=False).rolling_mean(
                window_size=window, min_samples=window
            ).alias("SMA_20"),
            (
                pl.col("Close").cast(pl.Float64, strict=False).rolling_mean(
                    window_size=window, min_samples=window
                )
                + num_std
                * pl.col("Close").cast(pl.Float64, strict=False).rolling_std(
                    window_size=window, min_samples=window
                )
            ).alias("BB_Up"),
            (
                pl.col("Close").cast(pl.Float64, strict=False).rolling_mean(
                    window_size=window, min_samples=window
                )
                - num_std
                * pl.col("Close").cast(pl.Float64, strict=False).rolling_std(
                    window_size=window, min_samples=window
                )
            ).alias("BB_Down"),
        ]
    ).collect()


def macd(df):
    frame = ensure_polars_frame(df)
    return (
        frame.lazy()
        .with_columns(
            pl.col("Close").cast(pl.Float64, strict=False).alias("Close")
        )
        .with_columns(
            (
                pl.col("Close").ewm_mean(span=12, adjust=False)
                - pl.col("Close").ewm_mean(span=26, adjust=False)
            ).alias("MACD")
        )
        .with_columns(pl.col("MACD").ewm_mean(span=9, adjust=False).alias("Signal_Line"))
        .with_columns((pl.col("MACD") - pl.col("Signal_Line")).alias("MACD_Histogram"))
        .collect()
    )


def rsi(df, period=14):
    frame = ensure_polars_frame(df).drop_nulls()
    return (
        frame.lazy()
        .with_columns(pl.col("Close").cast(pl.Float64, strict=False).alias("Close"))
        .with_columns(pl.col("Close").diff().alias("__delta"))
        .with_columns(
            [
                pl.when(pl.col("__delta") > 0)
                .then(pl.col("__delta"))
                .otherwise(0.0)
                .alias("__gain"),
                pl.when(pl.col("__delta") < 0)
                .then(-pl.col("__delta"))
                .otherwise(0.0)
                .alias("__loss"),
            ]
        )
        .with_columns(
            [
                pl.col("__gain")
                .rolling_mean(window_size=period, min_samples=period)
                .alias("__avg_gain"),
                pl.col("__loss")
                .rolling_mean(window_size=period, min_samples=period)
                .alias("__avg_loss"),
            ]
        )
        .with_columns(
            (
                100.0
                - (
                    100.0
                    / (
                        1.0
                        + pl.col("__avg_gain")
                        / (pl.col("__avg_loss") + pl.lit(1e-10))
                    )
                )
            ).alias("RSI")
        )
        .drop(["__delta", "__gain", "__loss", "__avg_gain", "__avg_loss"])
        .collect()
    )

def run_bayesian_analysis(returns):
    import pymc as pm

    # Ensure returns is a numpy array or pandas series with no NaNs
    with pm.Model() as model:
        # 1. Prior: How fast does volatility change?
        step_size = pm.Exponential("step_size", 1.0)
        
        # 2. Hidden State: The 'Random Walk' of log-volatility
        log_vol = pm.GaussianRandomWalk("log_vol", sigma=step_size, shape=len(returns))
        
        # 3. Deterministic: Transform log back to standard volatility for your Monte Carlo
        vol = pm.Deterministic("vol", pm.math.exp(log_vol))
        
        
        r = pm.StudentT("r", nu=4, sigma=vol, observed=returns)
        
        # 5. Inference: MCMC Sampling (This is the heavy lifting)
        trace = pm.sample(1000, tune=1000, target_accept=0.9, chains=2)
        
    return trace


# changed: estimate forward volatility from recent returns with a fast EWMA blend instead of per-request Bayesian sampling.
@njit(cache=True, fastmath=True)
def _estimate_annualized_volatility_numba(log_returns):
    n = log_returns.size
    if n == 0:
        return 0.0

    start = 0
    if n > 63:
        start = n - 63
    recent_count = n - start
    if recent_count <= 0:
        return 0.0

    mean_return = 0.0
    for idx in range(start, n):
        mean_return += log_returns[idx]
    mean_return /= recent_count

    realized_var = 0.0
    ewma_var = 0.0
    downside_var = 0.0
    downside_count = 0
    alpha = 0.94

    for idx in range(start, n):
        centered = log_returns[idx] - mean_return
        squared = centered * centered
        realized_var += squared
        ewma_var = alpha * ewma_var + (1.0 - alpha) * squared
        if centered < 0.0:
            downside_var += squared
            downside_count += 1

    realized_vol = np.sqrt(realized_var / recent_count)
    ewma_vol = np.sqrt(ewma_var)
    downside_vol = realized_vol
    if downside_count > 0:
        downside_vol = np.sqrt(downside_var / downside_count)

    blended = 0.5 * ewma_vol + 0.3 * realized_vol + 0.2 * downside_vol
    annualized = blended * np.sqrt(252.0)
    if annualized < 0.05:
        annualized = 0.05
    elif annualized > 2.5:
        annualized = 2.5
    return annualized


# changed: share one fast volatility estimator across simulation endpoints so they avoid expensive repeated inference.
def estimate_annualized_volatility_from_close(close_prices):
    if close_prices is None:
        return 0.0
    close_array = np.asarray(close_prices, dtype=np.float64)
    if close_array.size < 2:
        return 0.0
    valid_mask = np.isfinite(close_array) & (close_array > 0)
    close_array = close_array[valid_mask]
    if close_array.size < 2:
        return 0.0
    log_returns = np.diff(np.log(close_array))
    if log_returns.size == 0:
        return 0.0
    return float(_estimate_annualized_volatility_numba(log_returns))
def _run_all_technicals_pandas_reference(df):
    ref_df = ensure_pandas_frame(df).copy()
    close = (
        ref_df["Close"]
        if pd.api.types.is_float_dtype(ref_df["Close"])
        else pd.to_numeric(ref_df["Close"], errors="coerce")
    )
    high = (
        ref_df["High"]
        if pd.api.types.is_float_dtype(ref_df["High"])
        else pd.to_numeric(ref_df["High"], errors="coerce")
    )
    low = (
        ref_df["Low"]
        if pd.api.types.is_float_dtype(ref_df["Low"])
        else pd.to_numeric(ref_df["Low"], errors="coerce")
    )
    volume = (
        ref_df["Volume"]
        if pd.api.types.is_float_dtype(ref_df["Volume"])
        else pd.to_numeric(ref_df["Volume"], errors="coerce")
    )
    ref_df["Close"] = close
    ref_df["High"] = high
    ref_df["Low"] = low
    ref_df["Volume"] = volume
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    ref_df["RSI"] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ref_df["MACD"] = ema12 - ema26
    ref_df["Signal_Line"] = ref_df["MACD"].ewm(span=9, adjust=False).mean()
    ref_df["MACD_Histogram"] = ref_df["MACD"] - ref_df["Signal_Line"]
    rolling_20 = close.rolling(window=20)
    sma_20 = rolling_20.mean()
    close_std_20 = rolling_20.std()
    ref_df["SMA_20"] = sma_20
    ref_df["BB_Up"] = sma_20 + 2 * close_std_20
    ref_df["BB_Down"] = sma_20 - 2 * close_std_20
    ref_df["SMA_50"] = close.rolling(50).mean()
    ref_df["SMA_100"] = close.rolling(100).mean()
    ref_df["Volatility"] = close.pct_change().rolling(20).std()
    typical_price = (high + low + close) / 3
    cumulative_volume = volume.cumsum()
    ref_df["Ty"] = typical_price
    ref_df["Cum_TP_Vol"] = (typical_price * volume).cumsum()
    ref_df["Cum_Vol"] = cumulative_volume
    ref_df["VWAP"] = ref_df["Cum_TP_Vol"] / cumulative_volume.replace(0, np.nan)
    return ref_df


def _validate_polars_technical_output(source_df, result_df):
    reference = _run_all_technicals_pandas_reference(source_df)
    candidate = ensure_pandas_frame(result_df)
    if reference.columns.tolist() != candidate.columns.tolist():
        raise ValueError("Polars technical schema does not match pandas reference")
    if len(reference) != len(candidate):
        raise ValueError("Polars technical row count does not match pandas reference")
    for column in reference.columns:
        if column == "Date":
            left = pd.to_datetime(reference[column], errors="coerce")
            right = pd.to_datetime(candidate[column], errors="coerce")
            if not left.equals(right):
                raise ValueError(f"Polars technical values mismatch for column {column}")
            continue
        left = reference[column].to_numpy()
        right = candidate[column].to_numpy()
        if pd.api.types.is_numeric_dtype(reference[column]):
            if not np.allclose(left, right, equal_nan=True, atol=1e-8, rtol=1e-8):
                raise ValueError(f"Polars technical values mismatch for column {column}")
        else:
            if not pd.Series(left).equals(pd.Series(right)):
                raise ValueError(f"Polars technical values mismatch for column {column}")


def run_all_technicals(df):
    frame = ensure_polars_frame(df)
    result = (
        frame.lazy()
        .with_columns(
            [
                pl.col("Close").cast(pl.Float64, strict=False).alias("Close"),
                pl.col("High").cast(pl.Float64, strict=False).alias("High"),
                pl.col("Low").cast(pl.Float64, strict=False).alias("Low"),
                pl.col("Volume").cast(pl.Float64, strict=False).alias("Volume"),
            ]
        )
        .with_columns(pl.col("Close").diff().alias("__delta"))
        .with_columns(
            [
                pl.when(pl.col("__delta") > 0)
                .then(pl.col("__delta"))
                .otherwise(0.0)
                .alias("__gain"),
                pl.when(pl.col("__delta") < 0)
                .then(-pl.col("__delta"))
                .otherwise(0.0)
                .alias("__loss"),
                ((pl.col("High") + pl.col("Low") + pl.col("Close")) / 3.0).alias("Ty"),
            ]
        )
        .with_columns(
            [
                pl.col("__gain")
                .rolling_mean(window_size=14, min_samples=14)
                .alias("__avg_gain"),
                pl.col("__loss")
                .rolling_mean(window_size=14, min_samples=14)
                .alias("__avg_loss"),
                pl.col("Close").ewm_mean(span=12, adjust=False).alias("__ema12"),
                pl.col("Close").ewm_mean(span=26, adjust=False).alias("__ema26"),
                pl.col("Close")
                .rolling_mean(window_size=20, min_samples=20)
                .alias("SMA_20"),
                pl.col("Close")
                .rolling_std(window_size=20, min_samples=20)
                .alias("__close_std_20"),
                pl.col("Close")
                .rolling_mean(window_size=50, min_samples=50)
                .alias("SMA_50"),
                pl.col("Close")
                .rolling_mean(window_size=100, min_samples=100)
                .alias("SMA_100"),
                pl.col("Close").pct_change().rolling_std(window_size=20, min_samples=20).alias("Volatility"),
                (pl.col("Ty") * pl.col("Volume")).cum_sum().alias("Cum_TP_Vol"),
                pl.col("Volume").cum_sum().alias("Cum_Vol"),
            ]
        )
        .with_columns(
            [
                (
                    100.0
                    - (
                        100.0
                        / (
                            1.0
                            + pl.col("__avg_gain")
                            / (pl.col("__avg_loss") + pl.lit(1e-10))
                        )
                    )
                ).alias("RSI"),
                (pl.col("__ema12") - pl.col("__ema26")).alias("MACD"),
                (pl.col("SMA_20") + 2.0 * pl.col("__close_std_20")).alias("BB_Up"),
                (pl.col("SMA_20") - 2.0 * pl.col("__close_std_20")).alias("BB_Down"),
            ]
        )
        .with_columns(pl.col("MACD").ewm_mean(span=9, adjust=False).alias("Signal_Line"))
        .with_columns(
            [
                (pl.col("MACD") - pl.col("Signal_Line")).alias("MACD_Histogram"),
                pl.when(pl.col("Cum_Vol") != 0)
                .then(pl.col("Cum_TP_Vol") / pl.col("Cum_Vol"))
                .otherwise(None)
                .alias("VWAP"),
            ]
        )
        .drop(["__delta", "__gain", "__loss", "__avg_gain", "__avg_loss", "__ema12", "__ema26", "__close_std_20"])
        .collect()
    )
    if os.getenv("VALIDATE_POLARS_TECHNICALS") == "1":
        _validate_polars_technical_output(frame, result)
    return result


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
        # changed: reuse the cached technical frame for the ticker and keep the market benchmark fetch in parallel.
        df, spy = await asyncio.gather(
            get_technical_history(ticker.upper()), get_alpaca_history("SPY")
        )
        if is_empty_frame(df) or is_empty_frame(spy):
            return {"error": f"No data found for {ticker}"}

        # changed: run the rest of the analytics in one worker thread and reuse the already-computed technical columns.
        def run_analysis(stock_df, spy_df):
            def cagr(frame):
                start = float(frame.select(pl.col("Close").first()).item())
                end = float(frame.select(pl.col("Close").last()).item())
                start_date = frame.select(pl.col("Date").first()).item()
                end_date = frame.select(pl.col("Date").last()).item()
                days = int((end_date - start_date).days)
                if days == 0 or start == 0:
                    return 0.0
                return ((end / start) ** (365.25 / days) - 1) * 100

            current_rsi = float(stock_df.select(pl.col("RSI").last()).item())
            current_macd = float(stock_df.select(pl.col("MACD").last()).item())
            signal_line = float(stock_df.select(pl.col("Signal_Line").last()).item())
            current_price = float(stock_df.select(pl.col("Close").last()).item())
            sma50 = float(stock_df.select(pl.col("SMA_50").last()).item())
            sma100 = float(stock_df.select(pl.col("SMA_100").last()).item())
            close = stock_df.get_column("Close").to_numpy()
            returns = np.diff(close) / close[:-1]
            rsi_val = current_rsi
            macd_val = current_macd
            signal_line_val = signal_line
            price = current_price
            sma50_val = sma50
            bb_lower = float(stock_df.select(pl.col("BB_Down").last()).item())
            bb_upper = float(stock_df.select(pl.col("BB_Up").last()).item())
            annual_vol = (
                float(returns.std() * (252**0.5) * 100) if returns.size else 0.0
            )
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
                bull_reasons.append(
                    "price is trading above the 50-day moving average, which supports the current uptrend"
                )
            if macd_val > signal_line_val:
                bull_reasons.append(
                    "MACD is above the signal line, showing bullish momentum is still in place"
                )
            if rsi_val < 35:
                bull_reasons.append(
                    "RSI is near oversold territory, which can create room for a rebound"
                )
            if price < bb_lower:
                bull_reasons.append(
                    "price is below the lower Bollinger Band, which can signal a stretched downside move"
                )
            if stock_cagr > spy_cagr:
                bull_reasons.append(
                    "The stock has outperformed SPY on a CAGR basis, showing stronger longer-term trend strength"
                )
            if sharpe > 1:
                bull_reasons.append(
                    "The Sharpe ratio is healthy, which suggests recent returns have been efficient relative to risk"
                )
            bear_reasons = []
            if price < sma50_val:
                bear_reasons.append(
                    "Price is below the 50-day moving average, which points to weak near-term trend structure"
                )
            if macd_val < signal_line_val:
                bear_reasons.append(
                    "MACD is below the signal line, showing momentum has weakened"
                )
            if rsi_val > 65:
                bear_reasons.append(
                    "RSI is near overbought territory, which raises the chance of a pullback"
                )
            if price > bb_upper:
                bear_reasons.append(
                    "Price is above the upper Bollinger Band, which can signal an overheated move"
                )
            if stock_cagr < spy_cagr:
                bear_reasons.append(
                    "The stock has lagged SPY on a CAGR basis, which weakens the relative-strength story"
                )
            if annual_vol > 40:
                bear_reasons.append(
                    "The annualized volatility is elevated, which makes the setup more fragile"
                )
            if sharpe < 0:
                bear_reasons.append(
                    "The Sharpe ratio is negative, meaning recent risk-adjusted performance has been poor"
                )
            if not bull_reasons:
                bull_reasons.append(
                    "There is no standout bullish signal right now, so the upside case depends on momentum improving from here"
                )
            if not bear_reasons:
                bear_reasons.append(
                    "There is no standout bearish signal right now, so the downside case mainly depends on trend deterioration"
                )
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
        if db is None:
            return None
        data = await anyio.to_thread.run_sync(db.get, key)
        if data:
            return orjson.loads(data)
    except Exception as e:
        print(f"Cache lookup error: {e}")
    return None


async def save_to_cache(key: str, value: any, ttl=600):
    """Saves any object (including NumPy/Pandas results) to Redis."""
    try:
        if db is None:
            return
        payload = orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY)
        await anyio.to_thread.run_sync(db.setex, key, ttl, payload)
    except Exception as e:
        print(f"Cache save error: {e}")


# This software is provided as-is for educational and research purposes. The developer is not responsible for any financial losses incurred through the use of this code.
# changed: added a small in-memory cache for repeated history requests
HISTORY_TTL_SECONDS = 30
RAW_HISTORY_FRAME_CACHE = {}
TECHNICAL_HISTORY_FRAME_CACHE = {}
BACKTEST_FRAME_CACHE = {}

# changed: cache repeated portfolio and macro responses for short bursts of traffic
RESPONSE_TTL_SECONDS = 30
PORTFOLIO_CACHE = {}
MACRO_CACHE = {}
# changed: cache computed technical frames and sentiment payloads so repeated requests avoid redoing CPU-heavy work.

# changed: cache simulate endpoint responses to avoid repeated CPU-heavy Monte Carlo and ML work.
SIMULATE_CACHE_TTL_SECONDS = 60
SIMULATE_RESPONSE_CACHE = {}
# changed: keep hot backtest responses in memory so repeated slider tweaks and page refreshes return instantly.
BACKTEST_CACHE_TTL_SECONDS = 60
BACKTEST_RESPONSE_CACHE = {}
# changed: keep randomize endpoint responses in memory because the simulation payload is large and expensive to rebuild.
RANDOMIZE_CACHE_TTL_SECONDS = 60
RANDOMIZE_RESPONSE_CACHE = {}
# changed: keep sentiment payloads in memory so repeated dashboard renders avoid even Redis round-trips.
SENTIMENT_CACHE_TTL_SECONDS = 300
SENTIMENT_RESPONSE_CACHE = {}
# changed: dedupe in-flight simulate tasks for identical concurrent requests.
INFLIGHT_SIMULATE_TASKS = {}
INFLIGHT_BACKTEST_TASKS = {}
INFLIGHT_RANDOMIZE_TASKS = {}
# changed: cache hot user target lookups to avoid repeated round-trips for the same key
TARGET_CACHE_TTL_SECONDS = 60
TARGET_CACHE = {}
# changed: index cached targets by ticker so the live stream only checks relevant alerts per symbol.
TARGETS_BY_TICKER = {}
# changed: reuse one shared feature list anywhere we need fully-computed technical columns
TECHNICAL_REQUIRED_COLUMNS = ["RSI", "MACD", "SMA_50", "SMA_100", "Volatility"]
# changed: dedupe identical in-flight async work so concurrent traffic shares one fetch or calculation.
INFLIGHT_HISTORY_TASKS = {}
INFLIGHT_PORTFOLIO_TASKS = {}
INFLIGHT_TECHNICAL_TASKS = {}
INFLIGHT_SENTIMENT_TASKS = {}
INFLIGHT_MACRO_TASKS = {}
# changed: reuse one shared monotonic clock binding so hot-path cache reads avoid repeated global lookups.
MONOTONIC = time.monotonic


def ensure_polars_frame(df):
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df, include_index=False)
    if isinstance(df, list):
        return pl.from_dicts(df)
    if isinstance(df, dict):
        return pl.DataFrame(df)
    raise TypeError(f"Unsupported dataframe type: {type(df)!r}")


def ensure_pandas_frame(df):
    if isinstance(df, pd.DataFrame):
        return df
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    return pd.DataFrame(df)


def is_empty_frame(df):
    return ensure_polars_frame(df).is_empty()


def get_all_active_tickers(db: Session):
    # Use your optimized SQLAlchemy logic to get unique tickers
    result = (
        db.execute(select(models.UserStockTarget.ticker).distinct()).scalars().all()
    )
    # Always include a high-volume "heartbeat" like BTC/USD to keep the pipe warm
    tickers = list(set(result + ["BTC/USD"]))
    return [t.upper() for t in tickers]


async def alpaca_to_shield_bridge():
    url = "wss://stream.data.alpaca.markets/v2/iex"

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                # 1. Auth
                await ws.send(
                    orjson.dumps(
                        {
                            "action": "auth",
                            "key": os.getenv("ALPACA_KEY"),
                            "secret": os.getenv("ALPACA_SECRET"),
                        }
                    )
                )

                # 2. Get tickers from DB (using a temporary session)
                with SessionLocal() as db:
                    active_tickers = get_all_active_tickers(db)

                # 3. Dynamic Subscribe
                await ws.send(
                    orjson.dumps(
                        {
                            "action": "subscribe",
                            "trades": active_tickers,
                            "quotes": active_tickers,  # Adding quotes for more density
                        }
                    )
                )

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
                            ticker = msg.get("S")  # 'S' is the Symbol

                            if current_price and ticker:
                                # changed: await the lightweight in-memory alert check directly to avoid spawning a task per tick.
                                await check_shield_activation(ticker, current_price)

        except Exception as e:
            print(f"🛡️  Shield Bridge Error: {e}. Retrying...")
            await asyncio.sleep(5)


LAST_ALERT_TIME = {}


async def check_shield_activation(ticker: str, current_price: float):
    # changed: read only the ticker-specific target bucket so live ticks do not scan the full cache.
    ticker_targets = TARGETS_BY_TICKER.get(ticker)
    if not ticker_targets:
        return

    for user_id, target_price in tuple(ticker_targets.items()):
        # changed: prune expired target cache entries in-line and replace division with a cheaper range check.
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
                print(
                    f"!!! ALERT for {user_id}: {ticker} hit {current_price} (Target: {target_price})"
                )

                # Push an alert back through the WebSocket to the specific user
                alert_msg = {
                    "type": "SHIELD_ALERT",
                    "ticker": ticker,
                    "price": current_price,
                    "msg": f"Target of {target_price} reached!",
                }
                await manager.broadcast_tick(alert_msg)


analyzer = SentimentIntensityAnalyzer()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://talos-ui-ten.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = httpx.AsyncClient(
    # changed: allow HTTP/2 reuse when the upstream supports it to reduce request overhead on repeated API calls.
    http2=True,
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
    timeout=httpx.Timeout(15.0),
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


# changed: centralize the hot target cache bookkeeping so reads and writes share one fast path
def get_cached_target_price(user_id: str, ticker: str):
    cache_key = (user_id, ticker)
    cached = TARGET_CACHE.get(cache_key)
    if not cached:
        return None, False
    now_ts = MONOTONIC()
    if now_ts - cached[0] >= TARGET_CACHE_TTL_SECONDS:
        # changed: keep the secondary ticker index in sync when a hot target cache entry expires.
        set_cached_target_price(user_id, ticker, None)
        return None, False
    return cached[1], True


# changed: keep target cache writes tiny and explicit
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


# changed: standardize TTL cache reads so the hot caches share one expiration path.
def get_ttl_cache_value(cache: dict, cache_key, ttl_seconds: float):
    cached = cache.get(cache_key)
    if not cached:
        return None, False
    now_ts = MONOTONIC()
    if now_ts - cached[0] >= ttl_seconds:
        cache.pop(cache_key, None)
        return None, False
    return cached[1], True


# changed: keep generic TTL cache writes tiny and reusable for data, analytics, and response payloads.
def set_ttl_cache_value(cache: dict, cache_key, value):
    cache[cache_key] = (MONOTONIC(), value)


# changed: cache DataFrames in a compact columnar form so Redis payloads are smaller and reconstruct faster than row-wise records.
def dataframe_to_cache_payload(df):
    frame = ensure_polars_frame(df)
    payload = frame.to_dict(as_series=False)
    if "Date" in payload:
        payload["Date"] = [
            value.isoformat() if value is not None else None for value in payload["Date"]
        ]
    return payload


# changed: parse ISO timestamps as UTC explicitly so Polars does not guess formats or drop embedded timezone offsets.
def utc_date_parse_expr(column_name: str = "Date") -> pl.Expr:
    return (
        pl.col(column_name)
        .cast(pl.Utf8, strict=False)
        .str.to_datetime(time_zone="UTC", strict=False)
    )


# changed: support both the new compact columnar cache format and the legacy list-of-records format.
def dataframe_from_cache_payload(payload):
    if isinstance(payload, list):
        df = pl.from_dicts(payload)
    elif isinstance(payload, dict):
        df = pl.DataFrame(payload)
    else:
        return pl.DataFrame()
    if "Date" in df.columns and df.schema.get("Date") == pl.String:
        df = df.with_columns(utc_date_parse_expr())
    return df


# changed: trim the leading indicator warmup rows with one contiguous slice instead of copying via dropna on every endpoint call.
def get_valid_technical_slice(df, required_columns: list[str]):
    frame = ensure_polars_frame(df).with_row_index("__row_idx")
    valid_expr = pl.all_horizontal(
        [pl.col(column).is_not_null() & pl.col(column).is_finite() for column in required_columns]
    )
    first_valid_idx = frame.filter(valid_expr).select(pl.col("__row_idx").min()).item()
    if first_valid_idx is None:
        return frame.head(0).drop("__row_idx")
    return frame.filter(pl.col("__row_idx") >= first_valid_idx).drop("__row_idx")


# changed: share identical in-flight async work so concurrent requests await one task instead of duplicating it.
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

    # changed: use a single database round-trip for save/update when the backend supports upsert
    target_table = models.UserStockTarget.__table__
    if db.bind and db.bind.dialect.name == "postgresql":
        stmt = (
            pg_insert(target_table)
            .values(
                user_id=user_id,
                ticker=ticker,
                target_price=target_price,
            )
            .on_conflict_do_update(
                index_elements=["user_id", "ticker"],
                set_={
                    "target_price": target_price,
                    "updated_at": datetime.datetime.now(datetime.timezone.utc),
                },
            )
        )
        db.execute(stmt)
    elif db.bind and db.bind.dialect.name == "sqlite":
        stmt = (
            sqlite_insert(target_table)
            .values(
                user_id=user_id,
                ticker=ticker,
                target_price=target_price,
            )
            .on_conflict_do_update(
                index_elements=["user_id", "ticker"],
                set_={
                    "target_price": target_price,
                    "updated_at": datetime.datetime.now(datetime.timezone.utc),
                },
            )
        )
        db.execute(stmt)
    else:
        # changed: keep the fallback path lean by selecting only the row we may mutate
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
                    user_id=user_id, ticker=ticker, target_price=target_price
                )
            )

    db.commit()
    # changed: update the hot cache immediately after a successful write
    set_cached_target_price(user_id, ticker, target_price)
    return {"status": "saved", "ticker": ticker, "target": target_price}


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}


async def get_alpaca_history(
    ticker, timeframe="1Day", period_days=365, start_override=None, end_override=None
):
    # changed: reuse recent ticker history instead of refetching immediately.
    upper_ticker = ticker.upper()
    # Create a unique key for this specific request
    cache_key = (
        f"hist:{upper_ticker}:{timeframe}:{period_days}:{start_override}:{end_override}"
    )

    cached_frame, found = get_ttl_cache_value(
        RAW_HISTORY_FRAME_CACHE, cache_key, HISTORY_TTL_SECONDS
    )
    if found:
        return cached_frame

    # --- NEW REDIS CHECK ---
    # We check Redis instead of the HISTORY_CACHE dictionary
    cached_data = await get_from_cache(cache_key)
    if cached_data:
        cached_frame = dataframe_from_cache_payload(cached_data)
        set_ttl_cache_value(RAW_HISTORY_FRAME_CACHE, cache_key, cached_frame)
        return cached_frame

    async def fetch_history():
        # LOGIC: If we have an override, use it; otherwise, look back from now
        if start_override:
            start_date = start_override
        else:
            start_date = (
                (datetime.datetime.now() - datetime.timedelta(days=period_days))
                .date()
                .isoformat()
            )

        url = f"{BASE_URL}/stocks/{upper_ticker}/bars?timeframe={timeframe}&start={start_date}"

        # Add end_date if we are targeting a specific historical window
        if end_override:
            url += f"&end={end_override}"

        url += "&adjustment=all"
        response = await client.get(url, headers=HEADERS)
        if response.status_code != 200:
            return pl.DataFrame()

        bars = response.json().get("bars")
        if not bars:
            return pl.DataFrame()

        frame = pl.DataFrame(
            {
                "Date": [bar["t"] for bar in bars],
                "Open": [bar["o"] for bar in bars],
                "High": [bar["h"] for bar in bars],
                "Low": [bar["l"] for bar in bars],
                "Close": [bar["c"] for bar in bars],
                "Volume": [bar["v"] for bar in bars],
            }
        )
        df = frame.lazy().with_columns(
            [
                utc_date_parse_expr(),
                pl.col("Open").cast(pl.Float64, strict=False),
                pl.col("High").cast(pl.Float64, strict=False),
                pl.col("Low").cast(pl.Float64, strict=False),
                pl.col("Close").cast(pl.Float64, strict=False),
                pl.col("Volume").cast(pl.Float64, strict=False),
            ]
        ).collect()
        if not df.is_empty():
            set_ttl_cache_value(RAW_HISTORY_FRAME_CACHE, cache_key, df)
            await save_to_cache(cache_key, dataframe_to_cache_payload(df), ttl=300)
        return df

    return await get_or_create_task_result(
        INFLIGHT_HISTORY_TASKS, cache_key, fetch_history
    )


async def get_black_swan_signature(ticker):
    # March 2020 COVID Window
    crash_df = await get_alpaca_history(
        ticker, start_override="2020-02-01", end_override="2020-05-01"
    )

    if crash_df.is_empty():
        return None

    close = crash_df.get_column("Close").to_numpy()
    returns = np.diff(close) / close[:-1]
    peak = float(crash_df.select(pl.col("Close").max()).item())
    trough = float(crash_df.select(pl.col("Close").min()).item())
    max_drawdown = (trough - peak) / peak

    realized_vol = np.std(returns) * np.sqrt(252)  # Annualized

    return {
        "max_drawdown": float(max_drawdown),
        "crash_volatility": float(realized_vol),
        "recovery_speed": crash_df.height,  # How many days it took
    }


@njit(parallel=True)
def run_black_swan_simulation(
    current_price, days, mu, sigma, shock_factor, num_sims=1000
):
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

            prices[t] = prices[t - 1] * np.exp(drift + diffusion)
        results[s, :] = prices

    return results


@app.get("/stock/{ticker}/black-swan")
async def black_swan(ticker: str):
    try:
        signature = await get_black_swan_signature(ticker)
        if signature is None:
            return {"error": "Not enough data for black swan analysis"}
        df_recent = await get_alpaca_history(ticker, period_days=5)
        if df_recent.is_empty():
            return {"error": "Not enough recent data for simulation"}
        current_price = df_recent.select(pl.col("Close").last()).item()
        sim_paths = run_black_swan_simulation(
            current_price=float(current_price),
            days=30,
            mu=-0.05,  # We assume a negative drift during a crash
            sigma=signature["crash_volatility"],
            shock_factor=1.5,
            num_sims=500,
        )
        worst_case_path = np.percentile(
            sim_paths, 5, axis=0
        )  # 5th percentile for worst-case scenario
        return {
            "ticker": ticker,
            "stress_label": "COVID-19 Impact Overlay",
            "historical_drawdown": signature["max_drawdown"],
            "projected_path": worst_case_path.tolist(),
            "vaR_percent": float((worst_case_path[-1] - current_price) / current_price),
        }
    except Exception as e:
        import logging

        logging.error(f"Black Swan Analysis Error for {ticker}: {e}")
        return {"error": str(e)}


# changed: cache fully-computed technical frames so simulation, analysis, and backtests reuse the same indicator work.
async def get_technical_history(ticker: str, period_days: int = 365):
    ticker = ticker.upper()
    cache_key = f"tech_df:{ticker}:{period_days}"

    cached_frame, found = get_ttl_cache_value(
        TECHNICAL_HISTORY_FRAME_CACHE, cache_key, HISTORY_TTL_SECONDS
    )
    if found:
        return cached_frame

    # 1. Check Redis for the ALREADY CALCULATED indicators
    cached = await get_from_cache(cache_key)
    if cached:
        cached_frame = dataframe_from_cache_payload(cached)
        set_ttl_cache_value(TECHNICAL_HISTORY_FRAME_CACHE, cache_key, cached_frame)
        return cached_frame

    async def build_technical_history():
        # 2. If not in Redis, get the raw history (This now uses the Redis cache you just built!)
        df = await get_alpaca_history(ticker, period_days=period_days)
        if df.is_empty():
            return df

        # 3. Run the math (CPU-heavy part)
        df = run_all_technicals(df)

        # 4. Save the "Finished Product" to Redis
        set_ttl_cache_value(TECHNICAL_HISTORY_FRAME_CACHE, cache_key, df)
        await save_to_cache(cache_key, dataframe_to_cache_payload(df), ttl=300)
        return df

    return await get_or_create_task_result(
        INFLIGHT_TECHNICAL_TASKS, cache_key, build_technical_history
    )


def run_backtest_indicators(df):
    frame = ensure_polars_frame(df)
    return (
        frame.lazy()
        .with_columns(pl.col("Close").cast(pl.Float64, strict=False).alias("Close"))
        .with_columns(pl.col("Close").diff().alias("__delta"))
        .with_columns(
            [
                pl.when(pl.col("__delta") > 0)
                .then(pl.col("__delta"))
                .otherwise(0.0)
                .alias("__gain"),
                pl.when(pl.col("__delta") < 0)
                .then(-pl.col("__delta"))
                .otherwise(0.0)
                .alias("__loss"),
            ]
        )
        .with_columns(
            [
                pl.col("__gain")
                .rolling_mean(window_size=14, min_samples=14)
                .alias("__avg_gain"),
                pl.col("__loss")
                .rolling_mean(window_size=14, min_samples=14)
                .alias("__avg_loss"),
            ]
        )
        .with_columns(
            (
                100.0
                - (
                    100.0
                    / (
                        1.0
                        + pl.col("__avg_gain")
                        / (pl.col("__avg_loss") + pl.lit(1e-10))
                    )
                )
            ).alias("RSI")
        )
        .select(["Date", "Close", "RSI"])
        .collect()
    )


async def get_backtest_history(ticker: str, period_days: int = 365 * 2):
    ticker = ticker.upper()
    cache_key = f"backtest_df:{ticker}:{period_days}"
    cached_frame, found = get_ttl_cache_value(
        BACKTEST_FRAME_CACHE, cache_key, HISTORY_TTL_SECONDS
    )
    if found:
        return cached_frame

    cached = await get_from_cache(cache_key)
    if cached:
        cached_frame = dataframe_from_cache_payload(cached)
        set_ttl_cache_value(BACKTEST_FRAME_CACHE, cache_key, cached_frame)
        return cached_frame

    async def build_backtest_history():
        df = await get_alpaca_history(ticker, period_days=period_days)
        if df.is_empty():
            return df
        df = run_backtest_indicators(df)
        set_ttl_cache_value(BACKTEST_FRAME_CACHE, cache_key, df)
        await save_to_cache(cache_key, dataframe_to_cache_payload(df), ttl=300)
        return df

    return await get_or_create_task_result(
        INFLIGHT_TECHNICAL_TASKS, cache_key, build_backtest_history
    )


async def port(tickers, num_port=3000):
    try:
        # changed: normalize once, reject underspecified requests early, and cache identical portfolio optimizations briefly.
        normalized_tickers = tuple(
            dict.fromkeys(t.strip().upper() for t in tickers if t.strip())
        )
        if len(normalized_tickers) < 2:
            return None, None
        cache_key = (normalized_tickers, num_port)
        cached, found = get_ttl_cache_value(
            PORTFOLIO_CACHE, cache_key, RESPONSE_TTL_SECONDS
        )
        if found:
            return cached

        async def build_portfolio():
            symbols_str = ",".join(normalized_tickers)
            start_date = (
                (datetime.datetime.now() - datetime.timedelta(days=730))
                .date()
                .isoformat()
            )
            url = f"{BASE_URL}/stocks/bars?timeframe=1Day&symbols={symbols_str}&start={start_date}&adjustment=all&limit=10000"
            response = await client.get(url, headers=HEADERS)
            if response.status_code != 200:
                return None, None

            bars_by_symbol = response.json().get("bars")
            if not bars_by_symbol:
                print("DEBUG: No bars returned from Alpaca")
                return None, None

            symbol_frames = []
            for symbol, bars in bars_by_symbol.items():
                if not bars:
                    continue
                symbol_frames.append(
                    pl.DataFrame(
                        {
                            "Date": [bar["t"] for bar in bars],
                            symbol: [bar["c"] for bar in bars],
                        }
                    )
                    .lazy()
                    .with_columns(
                        [
                            utc_date_parse_expr(),
                            pl.col(symbol).cast(pl.Float64, strict=False),
                        ]
                    )
                )
            if len(symbol_frames) < 2:
                return None, None

            # join frames iteratively with unique suffixes to avoid repeated
            # collisions like 'Date_right' when Polars emits suffixes for
            # overlapping column names during multiple joins
            left = symbol_frames[0]
            for i, right in enumerate(symbol_frames[1:], start=1):
                left = left.join(right, on="Date", how="full", suffix=f"_r{i}")
            prices = left.sort("Date").collect().sort("Date")
            value_columns = [column for column in prices.columns if column != "Date"]
            if len(value_columns) < 2:
                return None, None
            prices = prices.lazy().sort("Date").fill_null(strategy="forward").drop_nulls().collect()
            price_matrix = prices.select(value_columns).to_numpy()
            if price_matrix.shape[0] < 2 or price_matrix.shape[1] < 2:
                return None, None

            # changed: run the portfolio math on dense NumPy arrays to avoid extra pandas allocation overhead.
            log_returns = np.diff(np.log(price_matrix), axis=0)
            mean_returns = log_returns.mean(axis=0) * 252
            cov_matrix = np.cov(log_returns, rowvar=False) * 252
            assets = price_matrix.shape[1]
            risk_free = 0.0422
            gene = np.random.default_rng()
            weights = gene.random((num_port, assets))
            weights /= weights.sum(axis=1, keepdims=True)
            portfolio_returns = weights @ mean_returns
            portfolio_risks = np.sqrt(
                np.einsum("ij,jk,ik->i", weights, cov_matrix, weights, optimize=True)
            )
            portfolio_sharpe = np.divide(
                portfolio_returns - risk_free,
                portfolio_risks,
                out=np.full(portfolio_returns.shape, -np.inf),
                where=portfolio_risks > 0,
            )
            idx_max_sharpe = int(np.argmax(portfolio_sharpe))
            idx_min_vol = int(np.argmin(portfolio_risks))
            portfolio_tickers = value_columns
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

        return await get_or_create_task_result(
            INFLIGHT_PORTFOLIO_TASKS, cache_key, build_portfolio
        )

    except Exception as e:
        print(f"PORTFOLIO FATAL ERROR: {e}")
        return None, None


def calculate_prediction_cone(last_price, drift, vola, days=30):
    # Time array (1 to 30)
    t = np.arange(1, days + 1)

    # Pre-compute common subexpressions for speed
    sqrt_t = np.sqrt(t)
    log_1_drift = np.log(1 + drift)

    # 1. Expected Price (Median/p50) - using exp(log()) form for numerical stability
    p50 = last_price * np.exp(t * log_1_drift)

    # 2. Volatility Expansion (The "Cone")
    # 1.645 is the Z-score for a 90% confidence interval (5% to 95%)
    # The width of the cone grows with the square root of time
    margin = 1.645 * vola * sqrt_t

    # Calculate boundaries using exponential growth/decay
    # Compute exp(margin) once, use reciprocal for p5
    exp_margin = np.exp(margin)
    p5 = p50 / exp_margin
    p95 = p50 * exp_margin

    return p5, p50, p95


def calculate_probability(target_price, last_price, drift, vola, days=30):
    if not target_price:
        return 0
    if vola <= 0:
        return 100.0 if target_price <= last_price else 0.0
    # Standard normal distribution formula for price probability
    # ln(Target/Last) - (Drift - 0.5 * Vola^2) * Days / (Vola * sqrt(Days))
    d2 = (np.log(target_price / last_price) - (drift - 0.5 * vola**2) * days) / (
        vola * np.sqrt(days)
    )
    prob = (1 - norm.cdf(d2)) * 100
    return round(max(0, min(100, prob)), 2)


# --- 3. FASTAPI ENDPOINT ---
@app.get("/stock/{ticker}/simulate")
async def simulate(ticker: str, target_price: float = None):
    try:
        ticker_upper = ticker.upper()
        normalized_target = None if target_price is None else round(float(target_price), 4)
        # 1. Redis Cache Check
        # We include target_price in the key because the probability changes based on it
        cache_key = f"sim:{ticker_upper}:{normalized_target}"

        cached, found = get_ttl_cache_value(
            SIMULATE_RESPONSE_CACHE, cache_key, SIMULATE_CACHE_TTL_SECONDS
        )
        if found:
            return cached

        cached = await get_from_cache(cache_key)
        if cached:
            set_ttl_cache_value(SIMULATE_RESPONSE_CACHE, cache_key, cached)
            return cached

        # 2. Build the Simulation (The "Slow" Work)
        async def run_simulation():
            # changed: keep only the minimum useful lookback for the ML horizon and indicators.
            df = await get_technical_history(ticker_upper, period_days=220)
            if df is None or df.is_empty():
                return {"error": f"No data found for {ticker}"}

            df_clean = get_valid_technical_slice(
                df, TECHNICAL_REQUIRED_COLUMNS + ["Close"]
            )
            if df_clean.height < 10:
                return {"error": "Insufficient data for simulation"}

            # 4. ML Prediction (runs on the cleaned technical frame)
            current_price = float(df_clean.select(pl.col("Close").last()).item())
            expected_return_30d = await anyio.to_thread.run_sync(
                get_ml_predictions, df_clean, ticker_upper
            )
            predicted_price = current_price * (1.0 + float(expected_return_30d))
            drift = float(expected_return_30d) / 30.0

            # 5. Analytical Math
            close_array = df_clean.get_column("Close").to_numpy().astype(
                np.float64, copy=False
            )
            if close_array.size < 2:
                return {"error": "Insufficient price history for simulation"}
            daily_vola = estimate_annualized_volatility_from_close(close_array) / np.sqrt(
                252.0
            )

            p5, p50, p95 = calculate_prediction_cone(current_price, drift, daily_vola)
            prob = calculate_probability(
                normalized_target, current_price, drift, daily_vola
            )

            # 6. Payload Generation
            p5 = np.round(p5, 2)
            p50 = np.round(p50, 2)
            p95 = np.round(p95, 2)
            payload = [
                {
                    "Date": day_idx + 1,
                    "p5": float(p5[day_idx]),
                    "p50": float(p50[day_idx]),
                    "p95": float(p95[day_idx]),
                }
                for day_idx in range(30)
            ]

            return {
                "data": payload,
                "probability": f"{prob}%",
                "ml_expected_price": round(predicted_price, 2),
            }

        # 7. Execute with In-Flight de-duplication
        result = await get_or_create_task_result(
            INFLIGHT_SIMULATE_TASKS, cache_key, run_simulation
        )

        # 8. Save to Redis
        # Simulation is heavy, so we cache it for 10 minutes (600s)
        if "error" not in result:
            set_ttl_cache_value(SIMULATE_RESPONSE_CACHE, cache_key, result)
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
        # changed: reuse shared history fetching so repeated stock and history calls hit the same cache
        history_task = get_alpaca_history(ticker, period_days=365)
        quote_task = client.get(
            f"{BASE_URL}/stocks/{ticker}/quotes/latest", headers=HEADERS
        )
        history_df, quote_resp = await asyncio.gather(history_task, quote_task)

        if quote_resp.status_code != 200:
            return {"error": f"Alpaca API error: {quote_resp.status_code}"}
        q_data = quote_resp.json()
        if "quote" not in q_data or history_df.is_empty():
            return {"error": "No data found for ticker"}
        quote = q_data.get("quote")
        if not quote:
            return {"error": "No data found for this ticker"}
        daily = history_df.select(
            [
                pl.col("Open").last().alias("Open"),
                pl.col("High").last().alias("High"),
                pl.col("Low").last().alias("Low"),
                pl.col("Volume").last().alias("Volume"),
                pl.col("Close").last().alias("Close"),
            ]
        ).row(0, named=True)
        max_low = float(history_df.select(pl.col("Low").min()).item())
        max_high = float(history_df.select(pl.col("High").max()).item())
        return {
            "ticker": ticker,
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
        # Fast path: read the same Redis cache key used by get_alpaca_history
        upper = ticker.upper()
        cache_key = f"hist:{upper}:1Day:{period_days}:None:None"
        cached = await get_from_cache(cache_key)
        if cached:
            df = dataframe_from_cache_payload(cached)
            if df.is_empty():
                return []
            formatted = df.select(
                [
                    pl.col("Date").dt.strftime("%Y-%m-%d").alias("Date"),
                    pl.col("Close"),
                ]
            )
            return formatted.to_dicts()

        # Fallback: compute via shared history fetch
        df = await get_alpaca_history(ticker, period_days=period_days)
        if df.is_empty():
            return []
        return df.select(
            [
                pl.col("Date").dt.strftime("%Y-%m-%d").alias("Date"),
                pl.col("Close"),
            ]
        ).to_dicts()
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
        response = await client.get(url=url, params=params)
        data = response.json()
        obsv = data["observations"]
        real_data = obsv[0]["value"]
        if real_data == ".":
            return None
        return real_data

    except Exception as e:
        print(f"Error fetching macro data for {series_id}: {e}")
        return {"error": str(e)}


@app.get("/macro")
async def macro():
    try:
        cache_key = "macro:bundle"
        cached, found = get_ttl_cache_value(
            MACRO_CACHE, cache_key, RESPONSE_TTL_SECONDS
        )
        if found:
            return cached

        async def build_macro():
            # Just call the async function directly to create coroutines
            tasks = [
                get_macro("A191RL1Q225SBEA", fred_key),
                get_macro("CPIAUCSL", fred_key),
                get_macro("FEDFUNDS", fred_key),
                get_macro("UNRATE", fred_key),
                get_macro("DGS10", fred_key),
                get_macro("SP500", fred_key),
            ]

            # asyncio.gather will run all these network requests concurrently
            results = await asyncio.gather(*tasks)
            payload = {
                "gdp_growth": results[0],
                "inflation": results[1],
                "fed_funds": results[2],
                "unemployment": results[3],
                "treasury_yield": results[4],
                "sp500": results[5],
            }
            set_ttl_cache_value(MACRO_CACHE, cache_key, payload)
            return payload

        return await get_or_create_task_result(
            INFLIGHT_MACRO_TASKS, cache_key, build_macro
        )
    except Exception as e:
        return {"error": str(e)}


def wrap(df):
    try:
        frame = ensure_polars_frame(df)
        return (
            frame.lazy()
            .with_columns(
                [
                    pl.col("Close")
                    .cast(pl.Float64, strict=False)
                    .rolling_mean(window_size=50, min_samples=50)
                    .alias("SMA_50"),
                    pl.col("Close")
                    .cast(pl.Float64, strict=False)
                    .rolling_mean(window_size=100, min_samples=100)
                    .alias("SMA_100"),
                    pl.col("Close")
                    .cast(pl.Float64, strict=False)
                    .pct_change()
                    .rolling_std(window_size=20, min_samples=20)
                    .alias("Volatility"),
                    ((pl.col("High") + pl.col("Low") + pl.col("Close")) / 3.0).alias("Ty"),
                ]
            )
            .with_columns(
                [
                    (pl.col("Ty") * pl.col("Volume")).cum_sum().alias("Cum_TP_Vol"),
                    pl.col("Volume").cum_sum().alias("Cum_Vol"),
                ]
            )
            .with_columns(
                pl.when(pl.col("Cum_Vol") != 0)
                .then(pl.col("Cum_TP_Vol") / pl.col("Cum_Vol"))
                .otherwise(None)
                .alias("VWAP")
            )
            .collect()
        )
    except ZeroDivisionError:
        return None


def atr(df, period=14):
    frame = ensure_polars_frame(df)
    result = (
        frame.lazy()
        .with_columns(
            [
                (pl.col("High") - pl.col("Low")).alias("__high_low"),
                (pl.col("High") - pl.col("Close").shift(1)).abs().alias("__high_prev_close"),
                (pl.col("Low") - pl.col("Close").shift(1)).abs().alias("__low_prev_close"),
            ]
        )
        .with_columns(
            pl.max_horizontal(
                ["__high_low", "__high_prev_close", "__low_prev_close"]
            ).alias("__tr")
        )
        .with_columns(
            pl.col("__tr").rolling_mean(window_size=period, min_samples=period).alias("__atr")
        )
        .select(pl.col("__atr").last())
        .collect()
    )
    return result.item()


def sharpness(df, risk_free):
    close = ensure_polars_frame(df).get_column("Close").to_numpy()
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


# changed: JIT-compile the Monte Carlo core so repeated simulations spend less time inside Python loops.
@njit(cache=True, fastmath=True)
def _simulate_price_paths(last_price, expected_return, volatility, days, path_count):
    price_path = np.empty((days, path_count), dtype=np.float64)
    for path_idx in range(path_count):
        running_price = last_price
        for day_idx in range(days):
            running_price *= 1.0 + np.random.normal(expected_return, volatility)
            price_path[day_idx, path_idx] = running_price
    return price_path


# changed: compute all percentiles in one pass for better cache efficiency.
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


# changed: JIT-compile the strategy loop so repeated backtests avoid Python overhead on signal and portfolio updates.
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


def backtest(df, buy_rsi=30, sell_rsi=70, starter_cash=10000):
    try:
        frame = ensure_polars_frame(df)
        rsi = frame.get_column("RSI").to_numpy()
        close = frame.get_column("Close").to_numpy()
        # changed: route the heavy numeric work through numba when present and preserve the existing return shape.
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
            tot_returns = (
                (portfolio[-1] - starter_cash) * 100 / starter_cash
                if portfolio.size
                else 0.0
            )
            strat_std = np.std(strat_return)
            sharpe = (
                0.0
                if strat_std == 0
                else np.mean(strat_return) / strat_std * np.sqrt(252)
            )
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
            "sell": sell,
        }
    except KeyError:
        return {
            "portfolio": "N/A",
            "total_return": "N/A",
            "sharpe": "N/A",
            "buy": "N/A",
            "sell": "N/A",
        }


@app.get("/stock/{ticker}/backtest")
async def backtester(
    ticker: str, buy_rsi: float = 30, sell_rsi: float = 70, starter_cash: float = 10000
):
    try:
        ticker_upper = ticker.upper()
        cache_key = (
            ticker_upper,
            round(float(buy_rsi), 4),
            round(float(sell_rsi), 4),
            round(float(starter_cash), 2),
        )
        cached, found = get_ttl_cache_value(
            BACKTEST_RESPONSE_CACHE, cache_key, BACKTEST_CACHE_TTL_SECONDS
        )
        if found:
            return cached

        async def build_backtest():
            # changed: compute only the RSI indicator this strategy uses so cold backtests avoid full technical analysis work.
            technical_df = await get_backtest_history(ticker_upper, period_days=365 * 2)
            if technical_df.is_empty():
                return {"error": f"No data found for {ticker}"}

            def run_backtest_logic(df_in):
                clean_df = get_valid_technical_slice(df_in, ["Close", "RSI"])
                if clean_df.is_empty():
                    raise ValueError("Not enough clean data for backtest")
                if clean_df.height < 2:
                    raise ValueError("Insufficient data after cleaning indicators")
                result = backtest(clean_df, buy_rsi, sell_rsi, starter_cash)
                close = clean_df.get_column("Close").to_numpy()
                returns = np.diff(close) / close[:-1]
                buy_hold = starter_cash * np.cumprod(1 + returns)
                portfolio = np.asarray(result["portfolio"], dtype=np.float64)
                peak = np.maximum.accumulate(portfolio)
                drawdown = (portfolio - peak) / peak
                buy_hold_return = (
                    round((buy_hold[-1] - starter_cash) * 100 / starter_cash, 2)
                    if buy_hold.size
                    else 0.0
                )
                max_drawdown = (
                    round(float(np.min(drawdown) * 100), 2) if drawdown.size else 0.0
                )
                return {
                    "total_return": float(result["total_return"]),
                    "buy_hold_return": float(buy_hold_return),
                    "sharpe": float(result["sharpe"]),
                    "max_drawdown": float(max_drawdown),
                    # changed: read the existing compact signal counts returned by backtest instead of reindexing missing keys.
                    "buy_signals": int(result["buy"]),
                    "sell_signals": int(result["sell"]),
                    # changed: rely on NumPy's built-in list conversion, which is faster than a Python float loop here.
                    "portfolio": portfolio.tolist(),
                    "buy_hold": buy_hold.tolist(),
                }

            return await anyio.to_thread.run_sync(run_backtest_logic, technical_df)

        result = await get_or_create_task_result(
            INFLIGHT_BACKTEST_TASKS, cache_key, build_backtest
        )
        if "error" not in result:
            set_ttl_cache_value(BACKTEST_RESPONSE_CACHE, cache_key, result)
        return result
    except Exception as e:
        return {"error": f"Endpoint Error: {str(e)}"}


@app.get("/stock/{ticker}/sentiment")
async def get_sentiment(ticker: str):
    try:
        upper_ticker = ticker.upper()
        cache_key = f"sentiment:{upper_ticker}"

        cached, found = get_ttl_cache_value(
            SENTIMENT_RESPONSE_CACHE, cache_key, SENTIMENT_CACHE_TTL_SECONDS
        )
        if found:
            return cached

        # 1. Check Redis (The "Silent Operator")
        cached = await get_from_cache(cache_key)
        if cached:
            set_ttl_cache_value(SENTIMENT_RESPONSE_CACHE, cache_key, cached)
            return cached

        # 2. Fetch from Alpaca if not in cache
        NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
        params = {"symbols": upper_ticker, "limit": 5, "sort": "desc"}

        async def build_sentiment():
            response = await client.get(
                NEWS_URL,
                headers=HEADERS,
                params=params,
                timeout=httpx.Timeout(2.5, connect=1.0),
            )
            if response.status_code != 200:
                return {"error": f"Alpaca API error: {response.status_code}"}

            news_data = orjson.loads(response.content).get("news", [])
            if not news_data:
                return {"score": 0, "label": "No News Found", "articles": []}

            total_score = 0.0
            articles_output = []
            for article in news_data:
                # changed: keep the score input compact so sentiment stays fast while preserving enough context.
                text = f"{article.get('headline', '')} {article.get('summary', '')[:240]}"
                score = analyzer.polarity_scores(text)["compound"]
                total_score += score
                articles_output.append(
                    {
                        "headline": article["headline"],
                        "url": article["url"],
                        "sentiment": (
                            "Bullish"
                            if score > 0.05
                            else "Bearish" if score < -0.05 else "Neutral"
                        ),
                    }
                )

            avg_score = total_score / len(news_data)
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

        # 3. Use your In-Flight task logic to prevent double-fetching
        payload = await get_or_create_task_result(
            INFLIGHT_SENTIMENT_TASKS, cache_key, build_sentiment
        )

        # 4. Save to Redis for 1 hour (3600s)
        if "error" not in payload:
            set_ttl_cache_value(SENTIMENT_RESPONSE_CACHE, cache_key, payload)
            await save_to_cache(cache_key, payload, ttl=3600)

        return payload

    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return {"error": str(e)}


@app.get("/stock/{ticker}/target/{user_id}")
async def get_stock_target(ticker: str, user_id: str, db: Session = Depends(get_db)):
    ticker = ticker.upper()
    # changed: short-circuit hot reads from memory before touching the database
    cached_target, found = get_cached_target_price(user_id, ticker)
    if found:
        return {"target_price": cached_target}

    # changed: fetch only the target_price scalar instead of hydrating a full ORM object
    target_price = db.execute(
        select(models.UserStockTarget.target_price).where(
            models.UserStockTarget.user_id == user_id,
            models.UserStockTarget.ticker == ticker,
        )
    ).scalar_one_or_none()

    # changed: backfill the cache after a database read so repeated requests stay fast
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
        returns[i] = (prices[i + 1] - prices[i]) / prices[i]

    # 2. Apply signals (shift signals by 1 so we don't cheat by knowing today's price)
    strategy_returns = returns * signals[:-1]

    # 3. Calculate metrics
    total_return = np.sum(strategy_returns)

    # Sharpe Ratio: Mean / Std Dev (annualized)
    if np.std(strategy_returns) == 0:
        return 0.0

    sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)

    return sharpe  # We optimize for Sharpe because it rewards stability


@njit
def _calculate_rsi_numba(prices, period):
    n = len(prices)
    rsi = np.full(n, np.nan)

    # Calculate price changes (deltas)
    deltas = np.zeros(n - 1)
    for i in range(n - 1):
        deltas[i] = prices[i + 1] - prices[i]

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
    if down == 0:
        rsi[period] = 100
    else:
        rsi[period] = 100 - 100 / (1 + up / down)

    # Smooth the rest (Wilder's Smoothing or Simple Moving Average)
    for i in range(period + 1, n):
        delta = deltas[i - 1]
        up_val = delta if delta > 0 else 0.0
        down_val = -delta if delta < 0 else 0.0

        up = (up * (period - 1) + up_val) / period
        down = (down * (period - 1) + down_val) / period

        if down == 0:
            rsi[i] = 100
        else:
            rsi[i] = 100 - 100 / (1 + up / down)

    return rsi


@njit
def _generate_rsi_signals(prices, period, low_threshold, high_threshold):
    period = int(period)  # Ensure period is an integer for numba
    n = len(prices)
    signals = np.zeros(n)
    rsi_values = _calculate_rsi_numba(prices, period)  # You'll need this helper too!

    current_position = 0  # 0 = Cash, 1 = Long (Holding Stock)

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
    for i in prange(len(rsi_low_range)):  # Parallel
        low_val = rsi_low_range[i]
        for j in range(len(rsi_high_range)):  # Regular range here!
            high_val = rsi_high_range[j]
            current_signals = _generate_rsi_signals(
                close_prices, period, low_val, high_val
            )
            results[i, j] = _calculate_performance(close_prices, current_signals)

    return results


@app.post("/optimize")
async def optimize_strategy(request: Request):
    try:
        # 1. Manually pull the data from the React JSON envelope
        data = await request.json()
        ticker = data.get("ticker")
        period = int(data.get("period", 14))  # Numba NEEDS this to be an int

        if not ticker:
            return {"error": "Ticker is required"}

        # 2. Get data
        df = await get_alpaca_history(
            ticker, period_days=365
        )  # ✅ We can reuse the existing history fetch logic, which is already cached and optimized.

        # 3. Prepare data for Numba (Force float64)
        prices = df.get_column("Close").to_numpy().astype(np.float64, copy=False)
        low_range = np.arange(20, 41, 1).astype(np.float64)
        high_range = np.arange(60, 81, 1).astype(np.float64)

        # 4. Run the high-speed engine
        grid = run_rsi_search(prices, period, low_range, high_range)
        best_idx = np.unravel_index(np.argmax(grid), grid.shape)

        return {
            "best_low": float(low_range[best_idx[0]]),
            "best_high": float(high_range[best_idx[1]]),
            "max_sharpe": float(grid[best_idx]),
            "heatmap": grid.tolist(),
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


router = APIRouter()


@router.post("/generate-report")
async def generate_report(data: dict):
    # 'data' will contain ticker, price, cagr, etc., sent from Next.js

    # Lazy import weasyprint only when needed
    try:
        import weasyprint
    except ImportError:
        return {"error": "PDF generation not available - weasyprint not installed"}

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
        media_type="application/pdf",
        filename=f"{data['ticker']}_Report.pdf",
    )


def compute_weights(data_dict):
    """
    Simulates the Efficient Frontier and returns the Max Sharpe and Min Volatility
    portfolios in the exact shape required by the Talos frontend.
    """
    price_frame = ensure_polars_frame(data_dict)
    value_columns = [column for column in price_frame.columns if column != "Date"]
    returns_frame = (
        price_frame.select(value_columns)
        .lazy()
        .with_columns([pl.col(column).pct_change().alias(column) for column in value_columns])
        .drop_nulls()
        .collect()
    )
    num_assets = len(value_columns)

    # 2. Annualize Stats
    # 252 trading days in a year
    returns_matrix = returns_frame.to_numpy()
    returns_annual = returns_matrix.mean(axis=0) * 252
    cov_annual = np.cov(returns_matrix, rowvar=False) * 252
    risk_free_rate = 0.0422  # Current 10-year Treasury yield approx

    # 3. Monte Carlo Simulation
    num_portfolios = 3000
    rng = np.random.default_rng()
    weights_record = rng.random((num_portfolios, num_assets))
    weights_record /= weights_record.sum(axis=1, keepdims=True)
    portfolio_returns = weights_record @ returns_annual
    portfolio_risks = np.sqrt(
        np.einsum(
            "ij,jk,ik->i", weights_record, cov_annual, weights_record, optimize=True
        )
    )
    sharpe_ratios = np.divide(
        portfolio_returns - risk_free_rate,
        portfolio_risks,
        out=np.full(num_portfolios, -np.inf, dtype=np.float64),
        where=portfolio_risks > 0,
    )

    # 4. Extract Key Portfolios
    # Find index of Max Sharpe and Min Volatility
    max_sharpe_idx = int(np.argmax(sharpe_ratios))
    min_vol_idx = int(np.argmin(portfolio_risks))
    tickers = value_columns

    def format_strategy(idx):
        return {
            "return": float(portfolio_returns[idx]),
            "risk": float(portfolio_risks[idx]),
            "sharpe": float(sharpe_ratios[idx]),
            "weights": {
                ticker: float(weights_record[idx, i]) for i, ticker in enumerate(tickers)
            },
        }

    # 5. Return the exact "Shape" the frontend expects
    return {
        "max_sharpe": format_strategy(max_sharpe_idx),
        "min_vol": format_strategy(min_vol_idx),
    }


@app.get("/portfolio")
async def get_portfolio_optimization(tickers: str):
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

        # Check Cache first
        cache_key = f"portfolio:{':'.join(sorted(ticker_list))}"
        cached = await get_from_cache(cache_key)
        if cached:
            return cached

        # Fetch data (reusing your existing history function)
        tasks = [get_alpaca_history(t) for t in ticker_list]
        histories = await asyncio.gather(*tasks)

        valid_frames = []
        for ticker, df in zip(ticker_list, histories):
            if not df.is_empty():
                valid_frames.append(df.select(["Date", pl.col("Close").alias(ticker)]).lazy())

        if len(valid_frames) < 2:
            return {"error": "Need at least 2 valid tickers"}

        # Iteratively join frames with unique suffixes to avoid duplicate
        # column name collisions (e.g. repeated 'Date_right' creations).
        left = valid_frames[0]
        for i, right in enumerate(valid_frames[1:], start=1):
            left = left.join(right, on="Date", how="full", suffix=f"_r{i}")

        joined_prices = (
            left.sort("Date").collect().lazy().sort("Date").fill_null(strategy="forward").drop_nulls().collect()
        )

        # HERE IS THE CONNECTION:
        # We run your math function inside anyio.to_thread
        results = await anyio.to_thread.run_sync(compute_weights, joined_prices)

        # Save to Redis and return
        await save_to_cache(cache_key, results, ttl=3600)
        return results

    except Exception as e:
        return {"error": str(e)}

async_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
def get_ai_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    return Groq(api_key=key)
def get_async_ai_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    
    async_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
    return async_client    

@app.post("/journal/review")
async def review_trade(ticker: str, thesis: str):
    try:
        client = get_ai_client()
        if not client:
            return {"error": "AI Service is temporarily unavailable (Missing Key)."}

        ticker = ticker.upper()
        # Fetch context from Redis
        tech_data = await get_from_cache(f"analysis:{ticker}")

        # Build the 'Skeptic' prompt
        prompt = f"Audit this {ticker} trade. Context: {tech_data}. Thesis: {thesis}"

        completion = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional trade auditor. Reason through the technicals before giving a verdict. Return your analysis in a valid JSON format with keys: 'rating', 'critique', and 'suggestion'.",
                },
                {"role": "user", "content": prompt},
            ],
            # Qwen 3 supports a specific 'reasoning_format' if you want to see its 'thoughts'
            # but for your current JSON UI, keep it simple first.
            response_format={"type": "json_object"},
        )

        return orjson.loads(completion.choices[0].message.content)

    except Exception as e:
        return {"error": str(e)}


def run_monte_carlo(current_price: int, volatility: float, days: int = 30, simulations: int = 1000):
    # Daily drift (assuming 0 for a neutral/short-term stress test)
    # or use historical CAGR / 252
    mu = 0
    daily_vol = volatility / np.sqrt(252)  # Annual to daily
    simulation_count = max(10, int(simulations))
    day_count = max(1, int(days))
    rng = np.random.default_rng()

    # changed: use antithetic shocks so the same simulation budget converges faster and stays more stable.
    half_count = (simulation_count + 1) // 2
    shocks = rng.standard_normal((half_count, day_count))
    returns = np.concatenate((shocks, -shocks), axis=0)[:simulation_count]
    returns = mu - 0.5 * daily_vol**2 + daily_vol * returns

    # Calculate price paths: S_t = S_0 * exp(cumsum(returns))
    price_paths = current_price * np.exp(np.cumsum(returns, axis=1))

    # Get the 5th, 50th, and 95th percentiles for the UI
    final_prices = price_paths[:, -1]
    return {
        "bull_case": float(np.percentile(final_prices, 95)),
        "base_case": float(np.percentile(final_prices, 50)),
        "bear_case": float(np.percentile(final_prices, 5)),
        "paths": price_paths[:50].tolist()  # Send 50 paths to the frontend to graph
    }


def detect_regimes(returns):
    returns_array = np.asarray(returns, dtype=np.float64)
    returns_array = returns_array[np.isfinite(returns_array)]
    if returns_array.size < 10:
        raise ValueError("Not enough valid return data to detect regimes")
    returns_array = returns_array[-252:]

    # changed: fit the HMM only on recent clean returns so regime detection stays fast on the request path.
    X = returns_array.reshape(-1, 1)

    # 2. Fit the model
    # In your boardroom analysis module
    model = hmm.GaussianHMM(
    n_components=3, 
    tol=0.1,        # Loosen from default (usually 0.01)
    n_iter=50,      # Cap iterations to prevent timeouts
    init_params="stmc"
)
    try:
        model.fit(X)
    except Exception as e:
    # Fallback verdict for the Executive Coordinator
        return "Neutral: Statistical models failed to reach consensus due to high volatility."
    
    # 3. Predict states
    states = model.predict(X)

    # --- ADD THE NEW LOGIC HERE ---
    # We extract the variance (volatility) of each state. 
    # Covars usually come in a shape like (n_components, n_features, n_features)
    # Since we have 1 feature (returns), we just take the [0][0] element of each matrix.
    var_state_0 = float(np.ravel(model.covars_[0])[0])
    var_state_1 = float(np.ravel(model.covars_[1])[0])
    
    # Identify which index belongs to the higher volatility regime
    bear_state = int(np.argmax([var_state_0, var_state_1]))
    # ------------------------------

    return states, model, bear_state

@app.get("/randomize")
async def randomize(ticker: str, days: int = 30, simulations: int = 1000):
    try:
        ticker_upper = ticker.upper()
        normalized_days = max(1, min(int(days), 365))
        normalized_simulations = max(100, min(int(simulations), 10000))
        cache_key = (ticker_upper, normalized_days, normalized_simulations)

        cached, found = get_ttl_cache_value(
            RANDOMIZE_RESPONSE_CACHE, cache_key, RANDOMIZE_CACHE_TTL_SECONDS
        )
        if found:
            return cached

        async def build_randomize():
            df = await get_alpaca_history(ticker_upper, period_days=365)
            if df.is_empty():
                return {"error": "No data found for ticker"}

            close = df.get_column("Close").to_numpy().astype(np.float64, copy=False)
            valid_close = close[np.isfinite(close) & (close > 0)]
            if valid_close.size < 30:
                return {"error": "Insufficient price history for randomize"}
            returns = np.diff(np.log(valid_close))

            # 1. Run HMM to detect the current market "Mood"
            states, model, bear_index = detect_regimes(returns)
            current_state = int(states[-1]) # Convert from numpy int to native int for JSON
            
            # Identify if we are in the 'Bear' (High Vol) or 'Bull' (Low Vol) state
            # NEW - explicit Python bool
            is_crisis_regime = bool(int(current_state) == int(bear_index))
            regime_label = "BEAR" if is_crisis_regime else "BULL"

            current_price = float(valid_close[-1])

            # 2. Get your Bayesian volatility
            smart_vola = estimate_annualized_volatility_from_close(valid_close)

            # 3. Run Monte Carlo as usual
            mc_result = await anyio.to_thread.run_sync(
                run_monte_carlo,
                current_price,
                smart_vola,
                normalized_days,
                normalized_simulations,
            )

            # 4. Inject the Regime Data into the final payload
            # 4. Inject the Regime Data into the final payload
            mc_result["regime"] = {
                "current_state": current_state,  # Already converted to int above
                "label": regime_label,  # Already a string
                "is_crisis": is_crisis_regime,  # Already a bool - don't wrap again!
                "stay_probability": float(model.transmat_[current_state][current_state])
            }

            return mc_result
            
        payload = await get_or_create_task_result(
            INFLIGHT_RANDOMIZE_TASKS, cache_key, build_randomize
        )
        if "error" not in payload:
            set_ttl_cache_value(RANDOMIZE_RESPONSE_CACHE, cache_key, payload)
        return payload
    except Exception as e:
        return {'error':str(e)}
async def get_agent_report(role_prompt, data_payload):
    # Uses the global async_client we defined above
    async_client = get_async_ai_client()
    response = await async_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": f"Analyze this data: {data_payload}"}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content
# Step 3.1: Connecting the Personas to the Engine
async def run_boardroom_debate(market_data):
    """
    This is where the personas are actually utilized to 
    generate the multi-perspective debate.
    """
    
    # In dev/debug mode, short-circuit the external AI calls so tests return fast.
    if os.getenv("DEV_FAST_BOARDROOM") == "1":
        return {
            "technical": "MOCK_TECHNICAL_REPORT",
            "macro": "MOCK_MACRO_REPORT",
            "risk": "MOCK_RISK_REPORT",
        }

    # We pass the constants from prompts.py into each specialized call
    technical_task = get_agent_report(
        role_prompt=TECHNICAL_ANALYST_PROMPT, 
        data_payload=market_data['tech_indicators'] # e.g., RSI, MACD
    )
    
    macro_task = get_agent_report(
        role_prompt=MACRO_STRATEGIST_PROMPT, 
        data_payload=market_data['macro_stats'] # e.g., CPI, GDP
    )
    
    risk_task = get_agent_report(
        role_prompt=RISK_MANAGER_PROMPT, 
        data_payload=market_data['risk_metrics'] # e.g., Monte Carlo, Sharpe
    )
    
    # Fire all 3 with their unique identities simultaneously
    reports = await asyncio.gather(technical_task, macro_task, risk_task)
    
    return {
        "technical": reports[0],
        "macro": reports[1],
        "risk": reports[2]
    }
async def run_executive_coordinator(reports: dict):
    """
    The Final Synthesis: Takes the debate and produces a unified verdict.
    """
    coordinator_prompt = """
    ## Role: Chief Investment Officer (Executive Coordinator)
    You are presiding over a board meeting. You have reports from Technical, Macro, and Risk agents.

    ## Your Task
    1. Summarize the key conflict or agreement between the agents.
    2. Provide a final 'Conviction Score' between 0.0 (Strong Sell) and 1.0 (Strong Buy).
    3. If the Risk Manager flags a 'Black Swan' or high risk, you MUST lower the conviction score.

    ## Output Format
    Final Score: [Score]
    Summary: [Your 2-sentence synthesis]
    """

    debate_text = f"""
    Technical Report: {reports['technical']}
    Macro Report: {reports['macro']}
    Risk Report: {reports['risk']}
    """
    # Dev short-circuit
    if os.getenv("DEV_FAST_BOARDROOM") == "1":
        return "Final Score: 0.75\nSummary: DEV MOCK — quick synthesis."

    async_client = get_async_ai_client()
    response = await async_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": coordinator_prompt},
            {"role": "user", "content": debate_text}
        ],
        temperature=0.1 # Very low for high consistency
    )

    return response.choices[0].message.content
    

def save_boardroom_session(db: Session, ticker: str, user_id: str, reports: dict, executive_output: str):
    """
    Persists the entire multi-agent debate and synthesis to PostgreSQL.
    """
    # Simple regex to pull the score from the Executive Coordinator's text
    # e.g., "Final Score: 0.85" -> 0.85
    score_match = re.search(r"Final Score:\s*([\d.]+)", executive_output)
    score = float(score_match.group(1)) if score_match else 0.5

    new_session = BoardroomSession(
        ticker=ticker,
        user_id=user_id,
        technical_analysis=reports['technical'],
        macro_analysis=reports['macro'],
        risk_analysis=reports['risk'],
        executive_summary=executive_output,
        conviction_score=score,
        status="PENDING"  # 2026 Compliance: Requires user validation
    )

    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session

async def _handle_boardroom(ticker: str, db: Session):
    # 1. Gather your existing project data (Placeholder for your current logic)
    market_data = {
        "tech_indicators": "RSI: 65, MACD: Bullish Crossover, Trend: Upward",
        "macro_stats": "GDP: 2.1%, CPI: 3.2%, Fed Rate: 5.25%",
        "risk_metrics": "Monte Carlo VaR: 5%, Sharpe Ratio: 1.8"
    }

    # 2. Run the Multi-Agent Debate (Async & Parallel)
    reports = await run_boardroom_debate(market_data)

    # 3. Synthesize the debate into a final verdict
    executive_output = await run_executive_coordinator(reports)

    # 4. Save to PostgreSQL for the "Audit Trail" (Value Multiplier)
    session_record = save_boardroom_session(
        db=db, ticker=ticker, user_id="user_123", reports=reports, executive_output=executive_output
    )

    return {"session_id": session_record.id, "verdict": executive_output}


@app.api_route("/boardroom/{ticker}", methods=["GET", "POST"])
async def analyze_stock_boardroom(ticker: str, db: Session = Depends(get_db)):
    return await _handle_boardroom(ticker, db)


@app.api_route("/boardroom", methods=["GET", "POST"])
async def analyze_stock_boardroom_root(request: Request, db: Session = Depends(get_db)):
    # Accept ticker via query param or JSON body for clients that POST to /boardroom
    try:
        ticker = request.query_params.get("ticker")
        if not ticker:
            try:
                body = await request.json()
                ticker = body.get("ticker")
            except Exception:
                ticker = None

        if not ticker:
            return JSONResponse({"error": "Missing 'ticker' parameter"}, status_code=400)

        return await _handle_boardroom(ticker.upper(), db)
    except Exception as e:
        import logging
        import traceback
        logging.error(traceback.format_exc()) # This prints the FULL error to Render logs
        raise HTTPException(status_code=500, detail=str(e))