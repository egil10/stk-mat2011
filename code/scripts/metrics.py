"""Financial metrics used in our analyses."""

import numpy as np


# =============================================================================
# Execution Quality Metrics
# =============================================================================

def average_slippage(executed_price, intended_price):
    """Mean absolute difference between executed and intended prices."""
    return np.mean(np.abs(executed_price - intended_price))


def fill_rate(volume_executed, volume_required):
    """Proportion of required volume that was actually executed."""
    return np.sum(volume_executed) / np.sum(volume_required)


# =============================================================================
# Trade Duration Metrics
# =============================================================================

def average_holding_time(t_open, t_close):
    """Mean time between opening and closing trades."""
    return np.mean(t_close - t_open)


def exposure(t_open, t_close, sample_time):
    """Fraction of total sample time spent holding positions."""
    total_time = np.sum(t_close - t_open)
    return total_time / sample_time


# =============================================================================
# Profit & Loss Metrics
# =============================================================================

def total_pnl(pnl):
    """Sum of all trade profits and losses."""
    return np.sum(pnl)


def average_trade_pnl(pnl):
    """Mean profit/loss per trade."""
    return np.mean(pnl)


def average_gain(pnl):
    """Mean profit on winning trades."""
    return np.mean(pnl[pnl > 0])


def average_loss(pnl):
    """Mean absolute loss on losing trades."""
    return np.mean(np.abs(pnl[pnl < 0]))


def average_trade_return(entry_price, exit_price, quantity, costs):
    """Mean return per trade, accounting for transaction costs."""
    trade_return = ((exit_price - entry_price) * quantity - costs) / (entry_price * quantity)
    return np.mean(trade_return)


# =============================================================================
# Win/Loss Ratios
# =============================================================================

def win_rate(pnl):
    """Proportion of trades that are profitable."""
    return np.mean(pnl > 0)


def profit_factor(pnl):
    """Ratio of gross gains to gross losses."""
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    return gains / losses


def expectancy(pnl):
    """Expected value per trade: P(win)*avg_gain - P(loss)*avg_loss."""
    w = win_rate(pnl)
    return w * average_gain(pnl) - (1 - w) * average_loss(pnl)


# =============================================================================
# Risk Metrics
# =============================================================================

def max_drawdown(equity_curve):
    """Largest peak-to-trough decline as a fraction of peak value."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    return np.max(drawdown)


def return_volatility(returns):
    """Sample standard deviation of returns."""
    return np.std(returns, ddof=1)


# =============================================================================
# Risk-Adjusted Return Metrics
# =============================================================================

def mean_return(returns):
    """Arithmetic mean of returns."""
    return np.mean(returns)


def sharpe_ratio(returns, risk_free=0.0):
    """Excess return per unit of total volatility."""
    excess = returns - risk_free
    return np.mean(excess) / np.std(excess)


def sortino_ratio(returns, risk_free=0.0):
    """Excess return per unit of downside volatility."""
    excess = returns - risk_free
    downside = excess[excess < 0]
    return np.mean(excess) / np.std(downside)


def calmar_ratio(returns, max_dd):
    """Mean return divided by maximum drawdown."""
    return np.mean(returns) / max_dd


# =============================================================================
# Portfolio Activity Metrics
# =============================================================================

def turnover(quantity, entry_price, initial_capital):
    """Total traded value as a multiple of initial capital."""
    traded_value = np.sum(np.abs(quantity) * entry_price)
    return traded_value / initial_capital