# buyhold_metrics.py
# import libraries
import math
import numpy as np
import pandas as pd


# buy and hold function implementation
def buy_and_hold_metrics(
    daily_csv: str,
    start_date: str,
    end_date: str,
    total_round_trip_pips: float = 10.0,
    pip_value: float = 1e-4,
):
    df = pd.read_csv(daily_csv)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    mask = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
    d = df.loc[mask, ["Date", "Close"]].dropna().copy()
    if len(d) < 2:
        raise ValueError("Not enough rows in the requested date range.")

    # Snap to actual available trading days
    actual_start = str(d["Date"].iloc[0].date())
    actual_end = str(d["Date"].iloc[-1].date())

    p0 = float(d["Close"].iloc[0])
    pT = float(d["Close"].iloc[-1])

    cost_price = total_round_trip_pips * pip_value
    net_return = ((pT - p0) - cost_price) / p0
    net_return_pct = 100.0 * net_return

    # Equity (normalized) & Max Drawdown (%)
    equity = d["Close"].astype(float).to_numpy() / p0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    mdd_pct = 100.0 * float(dd.max())

    # Sharpe (daily, annualized), rf=0   -->   ریسک فری فرض شد
    prices = d["Close"].astype(float).to_numpy()
    daily_log_returns = np.diff(np.log(prices))
    if len(daily_log_returns) < 2 or daily_log_returns.std(ddof=1) == 0:
        sharpe_ann = float("nan")
    else:
        sharpe_daily = daily_log_returns.mean() / daily_log_returns.std(ddof=1)
        sharpe_ann = float(sharpe_daily * math.sqrt(252))

    # چاپ موارد مورد نیاز
    print("Period (actual):", actual_start, "->", actual_end)
    print("Net Return (%), with costs:", round(net_return_pct, 2))
    print("Max Drawdown (%):", round(mdd_pct, 2))
    print("Sharpe (annualized, daily, rf=0):", round(sharpe_ann, 4))

# فایل csv را میگیرد
if __name__ == "__main__":
    # Example:
    # python buyhold_metrics.py --file GBPUSD_DailyData.csv --start 2019-01-01 --end 2025-01-01
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Daily CSV with Date, Close")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--pips", type=float, default=10.0, help="Total round-trip pips (default 10)")
    args = parser.parse_args()

    buy_and_hold_metrics(args.file, args.start, args.end, total_round_trip_pips=args.pips)
