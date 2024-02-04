from sklearn.ensemble import RandomForestClassifier

def generate_trade_signal(model, data):
    """
    Generate trade signals based on a pre-trained machine learning model
    """
    features = create_features(data)  # You need to implement create_features
    prediction = model.predict(features)
    return 'buy' if prediction == 1 else 'sell'