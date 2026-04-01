from fastapi import FastAPI
import yfinance as yf
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import os
import asyncio
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"])

load_dotenv()
fred_key = os.getenv("FRED_KEY")
@app.get('/')
def root():
    return {'message': 'App is running'}


def port(tickers, num_port=3000):
    try:
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
    except Exception:
        return None, None
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
                "weights": {t: float(w) for t, w in zip(ticker_list, max_sharpe_df["Weight"])}
            },
            "min_vol": {
                "return": float(min_vol["returns"]),
                "risk": float(min_vol["risk"]),
                "sharpe": float(min_vol["sharpe"]),
                "weights": {t: float(w) for t, w in zip(ticker_list, min_vol["Weight"])}
            }
        }
    except Exception as e:
        return {'error':f'{e}'}

@app.get("/stock/{ticker}")
def get_stock(ticker: str):
    try:
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
    except:
        return None
@app.get('/stock/{ticker}/history')
def hist(ticker: str, period: str = '1y'):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        hist = hist.reset_index()
        hist["Date"] = hist["Date"].dt.strftime("%Y-%m-%d")
        return hist[["Date", "Close"]].to_dict(orient="records")
    except:
        return None


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
    
@app.get('/macro')
async def macro():
    try:
        loop = asyncio.get_event_loop()
        results = [
            loop.run_in_executor(None, get_macro, "A191RL1Q225SBEA", fred_key),
            loop.run_in_executor(None, get_macro, "CPIAUCSL", fred_key),
            loop.run_in_executor(None, get_macro, "FEDFUNDS", fred_key),
            loop.run_in_executor(None, get_macro, "UNRATE", fred_key),
            loop.run_in_executor(None, get_macro, "DGS10", fred_key),
            loop.run_in_executor(None, get_macro, "SP500", fred_key)
        ]
        results = await asyncio.gather(*results)
        data = {
            "gdp_growth": results[0],
            "inflation": results[1],
            "fed_funds": results[2],
            "unemployment": results[3],
            "treasury_yield": results[4],
            "sp500": results[5]
        }
        return data
    except Exception as e:
        return {'error':str(e)}

@app.get('/intrinsic')
def intr(ticker: str, growth_rate: float = 0.08, discount_rate: float = 0.10, terminal_growth_rate: float = 0.03, years: int = 5):
    try:
        
        stock = yf.Ticker(ticker)
        info = stock.info
        cash = stock.cashflow
        try:
            oper_cash = cash.loc["Operating Cash Flow"].iloc[0]
            capex = cash.loc["Capital Expenditure"].iloc[0]
            fcf = oper_cash + capex

        except:
            return {'error':'Could Not Find Data'}
        
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        shares_outstanding = info.get("sharesOutstanding")
        if not current_price or not shares_outstanding:
            return {'error':'Could Not Find Data'}
        proj = []
        fcf_curr = fcf

        for year in range(1, years + 1):
            fcf_proj = fcf_curr * (1 + growth_rate) ** year
            pres_val = fcf_proj / (1 + discount_rate) ** year
            proj.append(
                {
                    "Year": f"Year {year}",
                    "Projected FCF ($B)": fcf_proj / 1e9,
                    "Present Value ($B)": pres_val / 1e9,
                }
            )

        df_proj = pd.DataFrame(proj)
        final_fcf = fcf * (1 + growth_rate) ** years
        terminal_value = (
            final_fcf
            * (1 + terminal_growth_rate)
            / (discount_rate - terminal_growth_rate)
        )
        terminal_value_pv = terminal_value / (1 + discount_rate) ** years
        total_pv = df_proj["Present Value ($B)"].sum() * 1e9
        intrinsic_value_total = total_pv + terminal_value_pv
        intrinsic_value_per_share = intrinsic_value_total / shares_outstanding
        terminal_value_pv = terminal_value_pv / 1e9
        return {
    "intrinsic_value": round(float(intrinsic_value_per_share), 2),
    "current_price": current_price,
    "terminal_value": round(float(terminal_value_pv), 2),
    "projections": proj
}
    except Exception as e:
        return {'error':f'{e}'}


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


def sim(df):
    returns = df["Close"].dropna().pct_change()
    price = df["Close"].iloc[-1]

    vola = returns.std()
    ret = returns.mean()

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

@app.get('/analyze/{ticker}')
def analyse(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period='1y')
        df = df.reset_index()
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        spy = yf.Ticker('SPY').history('1y')
        spy = spy.reset_index()
        spy['Date'] = pd.to_datetime(spy["Date"]).dt.tz_localize(None)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = rsi(df)
        df = macd(df)
        df = bollinger(df)
        df = wrap(df)
        current_rsi = float(df["RSI"].iloc[-1])
        current_macd = float(df["MACD"].iloc[-1])
        signal_line = float(df["Signal_Line"].iloc[-1])
        current_price = float(df["Close"].iloc[-1])
        sma50 = (float(df["Close"].rolling(50).mean().iloc[-1]))
   
        sma100 = float(df["Close"].rolling(100).mean().iloc[-1])
        annual_vol = float(df["Close"].pct_change().std() * (252 ** 0.5) * 100)
        def cagr(df, price_col):
            start = df[price_col].iloc[0]
            end = df[price_col].iloc[-1]
            days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days
            if days == 0 or start == 0:
                return 0
            return ((end / start) ** (365 / days) - 1) * 100
        signal = current_rsi
        if signal <= 30:
            signal = 'Buy'
        elif signal > 30 and signal < 70:
            signal = 'Hold'
        else:
            signal = 'Sell'
        spy_cagr = cagr(spy, 'Close')
        stock_cagr = cagr(df, 'Close')
        info = stock.info
        risk_free = 0.0422
        sharpe = float(sharpness(df, risk_free))

        return {
            'rsi':round(float(current_rsi)),
            'macd':round(float(current_macd)),
            'signal_line':signal_line,
            'price':current_price,
            'sma50':sma50, 
            'sma100':sma100,
            'vola':round(float(annual_vol)),
            'rsi_signal': signal,
            'stock_cagr': stock_cagr,
            'spy_cagr': spy_cagr,
            'sharpe': sharpe
            }
    except Exception as e:
        return {'error':str(e)}