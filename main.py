from fastapi import FastAPI
import yfinance as yf
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"])
@app.get('/')
def root():
    return {'message': 'App is running'}


def port(tickers, num_port=3000):
    df = yf.download(tickers, period="2y", auto_adjust=True)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        prices = df['Close']
    else:
        prices = df[['Close']].rename(columns={'Close': tickers[0]})
    if prices.empty or len(prices.columns) < 2:
        return None
    returns = np.log(prices / prices.shift(1))
    mean = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    assets = len(tickers)
    risk_free = 0.0422
    gene = np.random.default_rng()
    result = []
    for _ in range(num_port):
        w = gene.random(assets)
        w = w / w.sum()
        portfolio_return = np.dot(w, mean)
        portfolio_risk = np.sqrt(w.T @ cov_matrix.values @ w)
        sharpe = (portfolio_return - risk_free) / portfolio_risk
        result.append({"returns": portfolio_return, "risk": portfolio_risk, "sharpe": sharpe, "Weight": w})
    result_df = pd.DataFrame(result)
    max_sharpe = result_df["sharpe"].idxmax()
    min_risk = result_df["risk"].idxmin()
    return result_df.iloc[max_sharpe], result_df.iloc[min_risk]

@app.get("/portfolio")
def optimize(tickers: str):
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
            "weights": {t: float(w) for t, w in zip(ticker_list, max_sharpe_df["Weight"])}
        },
        "min_vol": {
            "return": float(min_vol["returns"]),
            "risk": float(min_vol["risk"]),
            "sharpe": float(min_vol["sharpe"]),
            "weights": {t: float(w) for t, w in zip(ticker_list, min_vol["Weight"])}
        }
    }


@app.get("/stock/{ticker}")
def get_stock(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
    "ticker": ticker,
    "price": info.get("currentPrice"),
    "market_cap": info.get("marketCap"),
    "pe_ratio": info.get("trailingPE"),
    "forward_pe": info.get("forwardPE"),
    "dividend_yield": info.get("dividendYield"),
    "debt_to_equity": info.get("debtToEquity"),
    "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
    "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
}
@app.get('/stock/{ticker}/history')
def hist(ticker: str, period: str = '1y'):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    hist = hist.reset_index()
    hist["Date"] = hist["Date"].dt.strftime("%Y-%m-%d")
    return hist[["Date", "Close"]].to_dict(orient="records")


