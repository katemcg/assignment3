def predict_sentiment(text, model):
    """Predict sentiment of text reviews."""
    # Make predictions
    predictions = model.predict(text)
    predicted_sentiment = ['positive' if pred[1] > 0.5 else 'negative' for pred in predictions]

    return predicted_sentiment
