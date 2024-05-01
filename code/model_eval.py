from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(test_data_sentiment, predicted_sentiment):
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
