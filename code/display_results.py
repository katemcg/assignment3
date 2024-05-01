imports pandas as pd

def display_table(text, predicted_sentiment, test_data_sentiment):
    df = pd.DataFrame({
        'Review': text,
        'Prediction': predicted_sentiment,
        'Ground Truth': test_data_sentiment
        })
    
    return df
