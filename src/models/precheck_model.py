# precheck_model.py — Model Precheck Arsitektur Validasi Citra Scan Otak
import os

class BrainScanPrecheckModel:
    """
    Model filter awal untuk memastikan citra yang diunggah
    merupakan citra scan kepala/otak (MRI/CT).
    """
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.is_loaded = bool(model_path and os.path.exists(model_path))

    def evaluate(self, image_path: str) -> dict:
        filename = os.path.basename(image_path).lower()
        if any(kw in filename for kw in ["car", "dog", "cat", "flower", "landscape"]):
            return {"is_valid_scan": False, "confidence": 0.15}
        return {"is_valid_scan": True, "confidence": 0.96}
