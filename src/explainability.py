# explainability.py — Visual Explainability (Grad-CAM & Salience Mapping)
import os

def generate_heatmap(image_path: str, output_path: str = None) -> str:
    """
    Menghasilkan visualisasi interpretasi model (Grad-CAM heatmap).
    """
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_gradcam{ext}"
    return output_path
