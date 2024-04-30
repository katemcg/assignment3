from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from tensorflow.keras.models import load_model

def download_model_from_drive(folder_id, model_name, dest_path):
    """Download model weights from Google Drive."""
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Authenticate using local web server
    drive = GoogleDrive(gauth)

    # Search for the model file in the specified folder
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    for file in file_list:
        if file['title'] == model_name:
            file.GetContentFile(dest_path)
            print(f"Downloaded {model_name} to {dest_path}")
            return

    print(f"Model file {model_name} not found in the specified folder.")

def load_sentiment_model(model_path):
    """Load the sentiment analysis model from a file."""
    model = load_model(model_path)
    return model

# Example usage
if __name__ == "__main__":
    folder_id = 'your_google_drive_folder_id'
    model_name = input("Enter the name of the model file: ")
    dest_path = '/path/to/save/model.h5'
    download_model_from_drive(folder_id, model_name, dest_path)
    sentiment_model = load_sentiment_model(dest_path)
