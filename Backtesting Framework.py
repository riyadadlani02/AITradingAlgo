def backtest_strategy(strategy, data):
    """
    Test strategy against historical data
    """
    portfolio_value = []  # Keep track of portfolio value over time
    for timestep in data.index:
        signal = strategy(data.loc[timestep])  # Compute signal
        execute_order(signal)  # Execution logic
        portfolio_value.append(compute_portfolio_value())  # Recompute portfolio value
    return compute_strat_performance(portfolio_value)  # Compute performance

