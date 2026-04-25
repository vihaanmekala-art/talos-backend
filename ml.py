from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time

model_cache = {}
#changed: cache the final prediction for unchanged ticker/history inputs so repeat simulations skip feature engineering and inference.
prediction_cache = {}
#changed: hoisted feature names and weights so they are built once per process
FEATURE_COLUMNS = [
    'RSI', 'MACD', 'SMA_50', 'SMA_100', 'Volatility',
    'returns_1d', 'returns_5d', 'momentum',
    'volatility_5d', 'sma_ratio'
]
PREDICTION_WEIGHTS = np.array([1, 2, 3, 4, 5], dtype=np.float64)
#changed: precompute the denominator so prediction averaging stays on vector math only
PREDICTION_WEIGHT_SUM = PREDICTION_WEIGHTS.sum()
MODEL_CACHE_TTL_SECONDS = 3600
PREDICTION_CACHE_TTL_SECONDS = 300

#changed: key the hot prediction cache by the latest visible market data so identical requests reuse one result.
def get_prediction_cache_key(df, ticker):
    last_date = df["Date"].iat[-1]
    if hasattr(last_date, "value"):
        last_date = last_date.value
    return ticker, len(df), last_date, float(df["Close"].iat[-1])

def get_model(ticker, x_train, y_train):
    now = time.time()

    if ticker in model_cache:
        model, timestamp = model_cache[ticker]

        if now - timestamp < MODEL_CACHE_TTL_SECONDS:  # 1 hour cache
            return model

    #changed: kept model construction centralized so cached models are reused
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(x_train, y_train)

    model_cache[ticker] = (model, now)
    return model
def prepare_data(df):
    #changed: avoid full DataFrame copy - work directly on numpy arrays for speed
    close = df["Close"].to_numpy(dtype=np.float64, copy=False)
    rsi = df["RSI"].to_numpy(dtype=np.float64, copy=False)
    macd = df["MACD"].to_numpy(dtype=np.float64, copy=False)
    sma50 = df["SMA_50"].to_numpy(dtype=np.float64, copy=False)
    sma100 = df["SMA_100"].to_numpy(dtype=np.float64, copy=False)
    volatility = df["Volatility"].to_numpy(dtype=np.float64, copy=False)
    
    n = len(close)
    # Pre-allocate feature matrix
    feature_matrix = np.empty((n - 10, 10), dtype=np.float64)
    target = np.empty(n - 10, dtype=np.float64)
    
    # Vectorized feature computation
    returns_1d = np.empty(n, dtype=np.float64)
    returns_1d[1:] = np.diff(close) / close[:-1]
    returns_1d[0] = np.nan
    
    returns_5d = np.empty(n, dtype=np.float64)
    returns_5d[5:] = (close[5:] - close[:-5]) / close[:-5]
    returns_5d[:5] = np.nan
    
    momentum = np.empty(n, dtype=np.float64)
    momentum[5:] = close[5:] - close[:-5]
    momentum[:5] = np.nan
    
    volatility_5d = np.empty(n, dtype=np.float64)
    for i in range(5, n):
        volatility_5d[i] = close[i-5:i].std()
    volatility_5d[:5] = np.nan
    
    sma_ratio = np.empty(n, dtype=np.float64)
    sma_ratio = sma50 / (sma100 + 1e-10)
    
    # Build feature matrix (skip first 100 rows for warmup, last 5 for target)
    start_idx = 100
    end_idx = n - 5
    
    for i in range(start_idx, end_idx):
        idx = i - start_idx
        feature_matrix[idx, 0] = rsi[i]
        feature_matrix[idx, 1] = macd[i]
        feature_matrix[idx, 2] = sma50[i]
        feature_matrix[idx, 3] = sma100[i]
        feature_matrix[idx, 4] = volatility[i]
        feature_matrix[idx, 5] = returns_1d[i]
        feature_matrix[idx, 6] = returns_5d[i]
        feature_matrix[idx, 7] = momentum[i]
        feature_matrix[idx, 8] = volatility_5d[i]
        feature_matrix[idx, 9] = sma_ratio[i]
        target[idx] = (close[i + 5] - close[i]) / close[i] if i + 5 < n else np.nan
    
    # Remove rows with NaN
    valid_mask = np.isfinite(target) & np.isfinite(feature_matrix).all(axis=1)
    return feature_matrix[valid_mask], target[valid_mask]


def get_ml_predictions(df, ticker):
    # Check cache
    cache_key = get_prediction_cache_key(df, ticker)
    now = time.time()
    cached = prediction_cache.get(cache_key)
    if cached and now - cached[0] < PREDICTION_CACHE_TTL_SECONDS:
        return cached[1]

    x, y = prepare_data(df)

    if x.size == 0 or y.size == 0:
        raise ValueError("Not enough clean data for prediction")

    # Time-based split
    split = int(len(x) * 0.8)
    x_train = x.iloc[:split].to_numpy(dtype=np.float32, copy=False)
    y_train = y.iloc[:split].to_numpy(dtype=np.float32, copy=False)

    # Get/Train Model
    model = get_model(ticker, x_train, y_train)

    # Predict using the last 5 rows
    latest = x.iloc[-5:].to_numpy(dtype=np.float32, copy=False)
    preds = model.predict(latest)

    # Weighted average of the return predictions
    weighted_return = (preds * PREDICTION_WEIGHTS).sum() / PREDICTION_WEIGHT_SUM
    
    # --- MODIFIED SECTION ---
    # Convert decimal (0.025) to percentage (2.5)
    pct_change = float(weighted_return) * 100
    
    # Clip to +/- 15% for sanity (adjust these bounds as needed)
    pct_change = np.clip(pct_change, -15.0, 15.0)
    
    # Cache and return the percentage value
    prediction_cache[cache_key] = (now, pct_change)
    return pct_change
