import os
import requests
from pathlib import Path

MODELS = {
    "yolov8n-face.pt": {
        "url": "https://github.com/akanametov/yolov8-face/releases/download/v0.0.1/yolov8n-face.pt",
        "md5": "9b37d0135e12340c4c9d9d1573a8d7e5"
    },
    "arcface_r100_v1.onnx": {
        "url": "https://github.com/foamliu/InsightFace-v2/releases/download/v1.0/arcface_r100_v1.onnx",
        "md5": "f27d8b2199e5a150d3eab1c5d325d1e1"
    }
}

def download_models():
    # Create models directory
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    for model_name, model_info in MODELS.items():
        model_path = model_dir / model_name
        
        if not model_path.exists():
            print(f"Downloading {model_name}...")
            response = requests.get(model_info["url"], stream=True)
            
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"✅ {model_name} downloaded successfully")
        else:
            print(f"✓ {model_name} already exists")

if __name__ == "__main__":
    download_models()
    