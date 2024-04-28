from google_drive_downloader import GoogleDriveDownloader as gdd
from tensorflow.keras.models import load_model

def download_model_from_drive(file_id, dest_path):
    """Download model weights from Google Drive."""
    gdd.download_file_from_google_drive(file_id=file_id, dest_path=dest_path, unzip=False)

def load_model_from_file(model_path):
    """Load model from a file."""
    model = load_model(model_path)
    return model

# Example usage
if __name__ == "__main__":
    file_id = 'your_google_drive_file_id'
    dest_path = '/path/to/save/model.h5'
    download_model_from_drive(file_id, dest_path)
    loaded_model = load_model_from_file(dest_path)
