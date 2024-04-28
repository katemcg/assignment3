import pandas as pd
import tensorflow as tf
from preprocess_text import preprocessor
from predict_text import predict_sentiment
from model_loader import download_model_from_drive, load_sentiment_model

folder_id = ' ' # our google drive folder id
model_name = input("Please choose a model between ["","",""]: ")
dest_path = '/path/to/save/model.h5'
download_model_from_drive(folder_id, model_name, dest_path)
sentiment_model = load_sentiment_model(dest_path)

# load test csv
test_data = pd.read_csv("test_sample.csv")
test_reviews=test_data.review  # reviews to predict on

# Preprocess images
preprocessed_data = preprocessor(test_data)

# Make predictions
predictions = predict_sentiment(preprocessed_data, model_name)

# Display predictions
for test_data, prediction in zip(preprocessed_data, predictions):
    print(f'Review: {test_data}, Prediction: {prediction}')
