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
    ml_df = df.copy()
    #changed: bind the close series once so all derived ML features reuse the same source values.
    close = ml_df["Close"]

    # --- Feature Engineering ---
    ml_df['returns_1d'] = close.pct_change()
    ml_df['returns_5d'] = close.pct_change(5)
    ml_df['momentum'] = close - close.shift(5)
    ml_df['volatility_5d'] = close.rolling(5).std()
    ml_df['sma_ratio'] = ml_df['SMA_50'] / ml_df['SMA_100']

    # --- Target (predict % return instead of raw price) ---
    ml_df['target'] = close.pct_change(5).shift(-5)

    ml_df.dropna(inplace=True)

    #changed: reuse the shared feature list instead of rebuilding it per call
    x = ml_df[FEATURE_COLUMNS]
    y = ml_df['target']

    return x, y, ml_df


def get_ml_predictions(df, ticker):
    # Check cache
    cache_key = get_prediction_cache_key(df, ticker)
    now = time.time()
    cached = prediction_cache.get(cache_key)
    if cached and now - cached[0] < PREDICTION_CACHE_TTL_SECONDS:
        return cached[1]

    x, y, ml_df = prepare_data(df)

    if x.empty or y.empty:
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