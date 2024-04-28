import pandas as pd
import tensorflow as tf
from preprocess_text import preprocessor
from predict_text import predict_categories
from model_loader import download_model_from_drive, load_model_from_file

# Provide a comment like this:


### Hi User! Please choose a model between ["Glove150d", "DistillBERT", "GPT2"]
model_name = "DistillBERT"  # Choice to modify model name from your given list

## Now call appropriate class/function from backend to download relevant weights, instantiate specified model and load the weights, and return the loaded model, ready to predict

# load test csv
test_data = pd.read_csv("test_sample.csv")
test_reviews=test_data.review  # reviews to predict on

# Preprocess images
preprocessed_data = preprocessor(test_data)

# Make predictions
predictions = predict_review_categories(preprocessed_data, model_name)

# Display predictions
for test_data, prediction in zip(preprocessed_data, predictions):
    print(f'Review: {test_data}, Prediction: {prediction}')
