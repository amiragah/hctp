# trade_metrics.py
# import libraries
import numpy as np
import pandas as pd

# محاسبه MDD
def max_drawdown_money(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    return float(drawdown.max()) if len(drawdown) else float("nan")

# sharpe ratio (per trade) 
def sharpe_per_trade(profits: np.ndarray, rf: float = 0.0) -> float:
    if len(profits) < 2:
        return float("nan")
    excess = profits - rf
    std = excess.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(excess.mean() / std)


def compute_metrics(trades_csv: str, profit_col: str = "Profit"):
    df = pd.read_csv(trades_csv)
    profits = df[profit_col].dropna().astype(float).to_numpy()

    equity = np.cumsum(profits)  # P&L equity curve
    net_profit = float(profits.sum())
    mdd_money = max_drawdown_money(equity)

    rf = float("nan")
    if mdd_money and mdd_money > 0:
        rf = net_profit / abs(mdd_money)

    sr = sharpe_per_trade(profits, rf=0.0)

    print("Trades:", len(profits))
    print("Net Profit:", round(net_profit, 2))
    print("Max DD (money):", round(mdd_money, 2))
    print("Recovery Factor (RF):", round(rf, 4))
    print("Sharpe (per-trade, rf=0):", round(sr, 4))


if __name__ == "__main__":
    # Example:
    # python trade_metrics.py --file GBP_HCTP_trades.csv
    # فایل CSV مورد نیاز است
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="CSV file with per-trade profits")
    parser.add_argument("--profit_col", default="Profit", help="Profit column name")
    args = parser.parse_args()

    # نهایی
    compute_metrics(args.file, profit_col=args.profit_col)
