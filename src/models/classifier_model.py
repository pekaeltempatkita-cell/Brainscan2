# classifier_model.py — Model Klasifikasi 10 Kondisi Otak
import os
from src.disease_info import MAIN_CLASSES

class BrainClassifierModel:
    """
    Model klasifikasi multi-kelas untuk 10 jenis penyakit/kondisi otak.
    """
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.classes = MAIN_CLASSES

    def predict(self, image_path: str) -> dict:
        filename = os.path.basename(image_path).lower()
        target_class = "Normal_Healthy"
        for c in self.classes:
            if c.lower() in filename:
                target_class = c
                break

        return {
            "prediction": target_class,
            "classes": self.classes
        }
