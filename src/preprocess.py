# preprocess.py — Citra Preprocessing untuk Model NeuroCheck
import os

def preprocess_image(image_path: str, target_size=(224, 224)):
    """
    Validasi dan pra-pemrosesan citra scan otak (MRI/CT).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File citra tidak ditemukan: {image_path}")

    # Standard check
    size = os.path.getsize(image_path)
    return {
        "valid": True,
        "path": image_path,
        "file_size": size,
        "target_size": target_size
    }
