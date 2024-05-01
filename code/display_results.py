from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def display_table(text, predicted_sentiment, test_data_sentiment):
    """
    Returns a table displaying movie reviews, their corresponding predicted sentiment and ground truth sentiment.
    Parameters:
    - text: Column with unprocessed movie reviews in text.
    - predicted_sentiment: Sentiment predicted by model.
    - test_data_sentiment: Sentiment as per test data.
    """
    df = pd.DataFrame({
        'Review': text,
        'Prediction': predicted_sentiment,
        'Ground Truth': test_data_sentiment
        })
    return df

def evaluate_model(test_data_sentiment, predicted_sentiment):
    """
    Returns a classification report as well as confusion matrix.
    Parameters:
    - predicted_sentiment: Sentiment predicted by model.
    - test_data_sentiment: Sentiment as per test data.
    """
    # Generate classification report
    report = classification_report(test_data_sentiment, predicted_sentiment)
    print("Classification Report:")
    print(report)

    # Generate confusion matrix
    cm = confusion_matrix(test_data_sentiment, predicted_sentiment)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
