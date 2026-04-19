from sklearn.ensemble import RandomForestRegressor
import numpy as np


def prepare_data(df):
    ml_df = df.copy()

    # --- Feature Engineering ---
    ml_df['returns_1d'] = ml_df['Close'].pct_change()
    ml_df['returns_5d'] = ml_df['Close'].pct_change(5)
    ml_df['momentum'] = ml_df['Close'] - ml_df['Close'].shift(5)
    ml_df['volatility_5d'] = ml_df['Close'].rolling(5).std()
    ml_df['sma_ratio'] = ml_df['SMA_50'] / ml_df['SMA_100']

    # --- Target (predict % return instead of raw price) ---
    ml_df['target'] = ml_df['Close'].pct_change(5).shift(-5)

    ml_df.dropna(inplace=True)

    features = [
        'RSI', 'MACD', 'SMA_50', 'SMA_100', 'Volatility',
        'returns_1d', 'returns_5d', 'momentum',
        'volatility_5d', 'sma_ratio'
    ]

    x = ml_df[features]
    y = ml_df['target']

    return x, y, ml_df


def get_ml_predictions(df):
    x, y, ml_df = prepare_data(df)

    if x.empty or y.empty:
        raise ValueError("Not enough clean data for prediction")

    # --- Time-based split (prevents leakage) ---
    split = int(len(x) * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # --- Model ---
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(x_train, y_train)

    # --- Optional evaluation (you can log this) ---
    y_pred_test = model.predict(x_test)
    # Example metric (directional accuracy)
    
   
    # print(f"Directional Accuracy: {accuracy:.2f}")

    # --- Prediction (use last 5 rows, weighted) ---
    latest = ml_df[x.columns].iloc[-5:]
    preds = model.predict(latest)

    weights = np.array([1, 2, 3, 4, 5])
    weighted_pred = (preds * weights).sum() / weights.sum()

    # --- Convert return → price ---
    current_price = df['Close'].iloc[-1]
    predicted_price = current_price * (1 + weighted_pred)

    return float(predicted_price)