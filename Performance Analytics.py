def calculate_performance_metrics(portfolio_value):
    """
    Calculate performance metrics like Sharpe ratio, Sortino ratio, etc.
    """
    return {
        "sharpe_ratio": calculate_sharpe_ratio(portfolio_value),
        "sortino_ratio": calculate_sortino_ratio(portfolio_value),
        "max_drawdown": calculate_max_drawdown(portfolio_value)
    }

