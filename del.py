# Global executor to avoid the overhead of recreating it every request
model_cache = {}
prediction_cache = {}

FEATURE_COLUMNS_COUNT = 10
PREDICTION_WEIGHTS = np.array([1, 2, 3, 4, 5], dtype=np.float64)
PREDICTION_WEIGHT_SUM = PREDICTION_WEIGHTS.sum()
MODEL_CACHE_TTL_SECONDS = 3600
PREDICTION_CACHE_TTL_SECONDS = 300

# Mapping indices for the incoming matrix
DATE_IDX = 0
CLOSE_IDX = 1
RSI_IDX = 2
MACD_IDX = 3
SMA50_IDX = 4
SMA100_IDX = 5
VOL_IDX = 6
executor = ProcessPoolExecutor(max_workers=4) 
def get_prediction_cache_key(data_matrix, ticker):
    # Fix: Use CLOSE_IDX (1) and DATE_IDX (0) instead of strings
    return (ticker, data_matrix.shape[0], data_matrix[-1, DATE_IDX], float(data_matrix[-1, CLOSE_IDX]))

def prepare_data(data_matrix):
    # Fix: Access columns by their position in the matrix
    close = data_matrix[:, CLOSE_IDX].astype(np.float32)
    rsi = data_matrix[:, RSI_IDX].astype(np.float32)
    macd = data_matrix[:, MACD_IDX].astype(np.float32)
    sma_50 = data_matrix[:, SMA50_IDX].astype(np.float32)
    sma_100 = data_matrix[:, SMA100_IDX].astype(np.float32)
    volatility = data_matrix[:, VOL_IDX].astype(np.float32)

    row_count = close.size
    if row_count < 35:
        return np.empty((0, FEATURE_COLUMNS_COUNT), dtype=np.float32), np.empty(0, dtype=np.float32)

    # 1d Returns
    returns_1d = np.zeros(row_count, dtype=np.float32)
    returns_1d[1:] = (close[1:] / close[:-1]) - 1

    # 5d Returns
    returns_5d = np.full(row_count, np.nan, dtype=np.float32)
    returns_5d[5:] = (close[5:] / close[:-5]) - 1

    # Momentum
    momentum_pct = np.full(row_count, np.nan, dtype=np.float32)
    momentum_pct[5:] = (close[5:] / close[:-5]) - 1

    # SMA Distances
    sma_50_dist = np.divide(close - sma_50, sma_50, out=np.zeros_like(close), where=sma_50!=0)
    sma_100_dist = np.divide(close - sma_100, sma_100, out=np.zeros_like(close), where=sma_100!=0)

    # Volatility and Ratio
    volatility_5d = np.asarray(pd.Series(close).rolling(5).std(), dtype=np.float32)
    sma_ratio = np.divide(sma_50, sma_100, out=np.full(row_count, np.nan, dtype=np.float32), where=sma_100 != 0)

    # Target
    target = np.full(row_count, np.nan, dtype=np.float32)
    target[:-30] = (close[30:] / close[:-30]) - 1

    # Build the matrix using the pre-calculated vectors
    feature_matrix = np.empty((row_count, FEATURE_COLUMNS_COUNT), dtype=np.float32)
    feature_matrix[:, 0] = rsi
    feature_matrix[:, 1] = macd
    feature_matrix[:, 2] = sma_50_dist
    feature_matrix[:, 3] = sma_100_dist
    feature_matrix[:, 4] = volatility
    feature_matrix[:, 5] = returns_1d
    feature_matrix[:, 6] = returns_5d
    feature_matrix[:, 7] = momentum_pct
    feature_matrix[:, 8] = volatility_5d
    feature_matrix[:, 9] = sma_ratio

    valid_mask = np.isfinite(target) & np.isfinite(feature_matrix).all(axis=1)
    return feature_matrix[valid_mask], target[valid_mask]
def heavy_compute_logic(data_matrix, ticker, target_price):
    # Ensure get_ml_predictions is defined to accept the matrix
    ml_output = get_ml_predictions(data_matrix, ticker)
    daily_drift = ml_output / 30
    
    close_prices = data_matrix[:, CLOSE_IDX].astype(np.float32)
    paths, p5, p50, p95 = sim(close_prices, daily_drift)
    
    current_price = close_prices[-1]
    ml_expected_price = current_price * (1 + ml_output)
    prob = round(np.mean(paths[-1] >= target_price) * 100, 2) if target_price else 0
    
    days = np.arange(1, len(p5) + 1)
    # Round to 2 decimals to save bytes in the JSON response
    payload = np.column_stack((days, p5, p50, p95)).round(2)

    return {
        "columns": ["Day", "p5", "p50", "p95"],
        "data": payload.tolist(), 
        "probability": f"{prob}%",
        "ml_expected_price": round(ml_expected_price, 2)
    }
