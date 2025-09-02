import os
import requests
from pathlib import Path

def download_model():
    """Download model from Google Drive"""
    file_id = os.getenv('GOOGLE_DRIVE_FILE_ID', '1Mt2Xvx--d04qxP-rrrfZcjAsj8RN_IPN')
    destination = os.getenv('SKIN_CANCER_MODEL_PATH', 'skin_cancer_model.h5')
    
    # Check if model already exists
    if os.path.exists(destination):
        file_size = os.path.getsize(destination) / (1024*1024)
        print(f"✅ Model already exists: {destination} ({file_size:.1f} MB)")
        return True
    
    print(f"📥 Downloading model from Google Drive...")
    print(f"   File ID: {file_id}")
    print(f"   Destination: {destination}")
    
    # Google Drive direct download URL
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Save the file
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Verify download
        if os.path.exists(destination):
            file_size = os.path.getsize(destination) / (1024*1024)
            print(f"✅ Model downloaded successfully: {file_size:.1f} MB")
            return True
        else:
            print("❌ Model download failed")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

if __name__ == "__main__":
    download_model()
