import numpy as np

def predict_sentiment(text, model):
    """Predict sentiment of text reviews."""
    # Make predictions
    prediction = model.predict(text)
    # Assuming binary classification (positive/negative)
    predicted_sentiment = 'Positive' if prediction > 0 else 'Negative'

    return predicted_sentiment
