# ar1.py — AR(1) process

import numpy as np
import matplotlib.pyplot as plt
# from statsmodels.tsa.ar_model import AutoReg


def load_data(path):
    # TODO: read parquet, return series
    pass


def fit_ar1(series):
    # TODO: fit AR(1), return model result
    pass


def print_summary(result):
    # TODO: print phi, sigma, AIC
    pass


def plot_fit(series, result):
    # TODO: plot actual vs fitted
    pass


def plot_diagnostics(result):
    # TODO: ACF of residuals, QQ plot
    pass


def forecast(result, n_steps=10):
    # TODO: forecast next n_steps, return array
    pass


if __name__ == "__main__":
    data = load_data("../../code/data/TODO.parquet")
    res = fit_ar1(data)
    print_summary(res)
    plot_fit(data, res)
    plot_diagnostics(res)
    forecast(res)
