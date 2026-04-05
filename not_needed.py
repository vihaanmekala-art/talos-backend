@app.get("/intrinsic")
def intr(
    ticker: str,
    growth_rate: float = 0.08,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.03,
    years: int = 5,
):
    try:

        stock = yf.Ticker(ticker)
        info = stock.info
        cash = stock.cashflow
        try:
            oper_cash = cash.loc["Operating Cash Flow"].iloc[0]
            capex = cash.loc["Capital Expenditure"].iloc[0]
            fcf = oper_cash + capex

        except:
            return {"error": "Could Not Find Data"}

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        shares_outstanding = info.get("sharesOutstanding")
        if not current_price or not shares_outstanding:
            return {"error": "Could Not Find Data"}
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
            "projections": proj,
        }
    except Exception as e:
        return {"error": f"{e}"}
