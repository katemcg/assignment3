from tensorflow.keras.models import load_model

def load_sentiment_model(model_path):
    """Load the sentiment analysis model from a file."""
    model = load_model(model_path)
    return model
