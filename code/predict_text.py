import numpy as np

def predict_sentiment(text, model):
    """Predict sentiment of text reviews."""
    # Make predictions
    predicted_sentiment = model.predict(text)
    # Assuming binary classification (positive/negative)
    # predicted_sentiment = 'Positive' if sentiment_prob > 0.5 else 'Negative'

    return predicted_sentiment
