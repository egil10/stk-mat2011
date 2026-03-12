
import numpy as np

def average_slippage(executed_price, intended_price):
    return np.mean(np.abs(executed_price - intended_price))

def fill_rate(volume_executed, volume_required):
    return np.sum(volume_executed) / np.sum(volume_required)

def average_holding_time(t_open, t_close):
    return np.mean(t_close - t_open)

def win_rate(pnl):
    return np.mean(pnl > 0)

def profit_factor(pnl):
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    return gains / losses

def average_trade_pnl(pnl):
    return np.mean(pnl)

def total_pnl(pnl):
    return np.sum(pnl)

def average_trade_return(entry_price, exit_price, quantity, costs):
    trade_return = ((exit_price-entry_price)*quantity-costs)/(entry_price*quantity)
    return np.mean(trade_return)

def average_gain(pnl):
    wins = pnl[pnl > 0]
    return np.mean(wins)

def average_loss(pnl):
    losses = pnl[pnl < 0]
    return np.mean(np.abs(losses))

def expectancy(pnl):
    w = win_rate(pnl)
    avg_gain = average_gain(pnl)
    avg_loss = average_loss(pnl)
    return w * avg_gain - (1 - w) * avg_loss

def max_drawdown(equity_curve):
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    return np.max(drawdown)

def sharpe_ratio(returns, risk_free = 0.0):
    downside = returns[returns < risk_free] - risk_free
    return np.mean(returns - risk_free) / np.std(downside)

def sortino_ratio(returns, risk_free = 0.0):
    downside = returns[returns < risk_free] - risk_free
    return np.mean(returns - risk_free) / np.std(downside)

def return_volatility(returns):
    return np.std(returns, ddof=1)

def mean_return(returns):
    return np.mean(returns)

def calmar_ratio(returns, max_dd):
    annual_return = np.mean(returns)
    return annual_return / max_dd

def turnover(quantity, entry_price, initial_capital):
    traded_value = np.sum(np.abs(quantity) * entry_price)
    return traded_value / initial_capital

def exposure(t_open, t_close, sample_time):
    total_time = np.sum(t_close - t_open)
    return total_time / sample_time

