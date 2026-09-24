"""
preprocess.py — Siapkan gambar upload user supaya persis sama seperti
format yang dipakai saat training (CLAHE + resize + normalize).
JANGAN pakai augmentasi random di sini -- ini bukan training, ini inferensi.

CATATAN VERSI ONNX: file ini SENGAJA tidak pakai `torch` sama sekali (cuma
numpy) -- biar backend produksi gak perlu install torch/torchvision yang
berat. ToTensorV2 (bagian dari albumentations.pytorch) juga sengaja gak
dipakai, transpose HWC->CHW dilakukan manual pakai numpy.
"""
import cv2
import numpy as np
from PIL import Image
import io

from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

import albumentations as A


def get_inference_transform():
    """Transform SAMA PERSIS seperti get_val_transforms() saat training
    (minus ToTensorV2 -- konversi ke array CHW dilakukan manual di bawah)."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_CUBIC),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def validate_image_bytes(file_bytes: bytes) -> bool:
    """Cek apakah file yang diupload beneran gambar valid (bukan file rusak/bukan gambar)."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        return True
    except Exception:
        return False


def bytes_to_array(file_bytes: bytes) -> np.ndarray:
    """
    Konversi bytes gambar (dari upload FastAPI) -> numpy array siap masuk model ONNX.
    Return shape: [1, 3, IMG_SIZE, IMG_SIZE], dtype float32 (sudah ada batch dimension,
    sudah CHW, sudah dinormalisasi) -- persis format input yang diharapkan
    onnxruntime.InferenceSession.run().
    """
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_array = np.array(pil_img)   # RGB, shape (H, W, 3)

    transform = get_inference_transform()
    transformed = transform(image=img_array)
    hwc = transformed["image"]                    # shape (IMG_SIZE, IMG_SIZE, 3), sudah dinormalisasi
    chw = np.transpose(hwc, (2, 0, 1))             # HWC -> CHW
    batched = np.expand_dims(chw, axis=0)          # tambah batch dim -> (1, 3, IMG_SIZE, IMG_SIZE)
    return batched.astype(np.float32)


def tensor_to_display_image(file_bytes: bytes) -> np.ndarray:
    """
    Untuk keperluan TAMPILAN (overlay, dll) -- gambar di-resize
    ke IMG_SIZE tapi TIDAK dinormalisasi, biar masih enak dilihat mata manusia.
    """
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_array = np.array(pil_img)
    resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    return resized
