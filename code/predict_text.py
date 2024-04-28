import numpy as np

def predict_sentiment(review, model):
    """Predict sentiment on a single movie review."""
    # Make predictions
    sentiment_prob = model.predict(review)
    # Assuming binary classification (positive/negative)
    predicted_sentiment = 'Positive' if sentiment_prob > 0.5 else 'Negative'

    return predicted_sentiment
