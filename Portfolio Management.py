def manage_risk(order, current_portfolio):
    """
    Implement risk management rules to an order based on the current portfolio
    """
    if is_risky_order(order, current_portfolio):  # Implement risk checks
        modify_order_risk_parameters(order)  # Implement function to modify order
    return order

