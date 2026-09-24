"""
inference.py — VERSI ONNX RUNTIME (tanpa torch sama sekali).

Model .onnx di-download SEKALI dari Hugging Face Hub saat startup (di-cache
otomatis oleh huggingface_hub di disk lokal container, jadi restart berikutnya
gak download ulang selama cache-nya persist). Kalau file .onnx sudah ada di
outputs/checkpoints/ (misal buat testing lokal), itu dipakai duluan tanpa
perlu internet.

CATATAN GRAD-CAM: fitur heatmap "area perhatian model" di versi torch lama
butuh gradient (backward pass), yang TIDAK tersedia di onnxruntime inference
session biasa. Makanya di versi ini gradcam_path SELALU None. Kalau nanti mau
dihidupkan lagi, alternatifnya pakai Score-CAM (gradient-free, cuma butuh
banyak forward pass) -- kasih tau saya kalau mau itu diimplementasikan.
"""
import numpy as np
import onnxruntime as ort

from config import (
    MAIN_MODEL_PATH, PRECHECK_MODEL_PATH,
    HF_REPO_ID, HF_PRECHECK_FILENAME, HF_MAIN_MODEL_FILENAME,
    MAIN_CLASSES, PRECHECK_CLASSES, PRECHECK_THRESHOLD,
)
from preprocess import bytes_to_array

precheck_session = None
main_session = None
MODELS_READY = False
MODEL_LOAD_ERROR = None


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _resolve_model_path(local_path, hf_filename):
    """Prioritas: file lokal (outputs/checkpoints/) kalau ada -> kalau tidak,
    download dari Hugging Face Hub (otomatis ke-cache oleh huggingface_hub)."""
    if local_path.exists():
        print(f"Pakai file lokal: {local_path}")
        return str(local_path)

    from huggingface_hub import hf_hub_download
    print(f"File lokal {local_path.name} tidak ada -- download dari Hugging Face Hub "
          f"({HF_REPO_ID}/{hf_filename}) ...")
    return hf_hub_download(repo_id=HF_REPO_ID, filename=hf_filename)


def load_models():
    """Coba load kedua model ONNX. Aman dipanggil ulang."""
    global precheck_session, main_session, MODELS_READY, MODEL_LOAD_ERROR
    try:
        precheck_path = _resolve_model_path(PRECHECK_MODEL_PATH, HF_PRECHECK_FILENAME)
        print("Memuat model precheck (ONNX) ...")
        precheck_session = ort.InferenceSession(precheck_path, providers=["CPUExecutionProvider"])

        main_path = _resolve_model_path(MAIN_MODEL_PATH, HF_MAIN_MODEL_FILENAME)
        print("Memuat model utama (ONNX) ...")
        main_session = ort.InferenceSession(main_path, providers=["CPUExecutionProvider"])

        MODELS_READY = True
        MODEL_LOAD_ERROR = None
        print("Semua model ONNX siap dipakai.")
    except Exception as e:
        MODELS_READY = False
        MODEL_LOAD_ERROR = (
            f"Gagal memuat model ONNX: {e}. Pastikan file .onnx ada di "
            f"outputs/checkpoints/ ATAU sudah diupload ke Hugging Face Hub "
            f"repo '{HF_REPO_ID}' (lihat scripts/upload_to_hf.py), dan HF_REPO_ID "
            f"di .env sudah benar."
        )
        print(f"[PERINGATAN] {MODEL_LOAD_ERROR}")


# Coba load pas modul ini di-import pertama kali (saat startup app).
load_models()


def run_pipeline(file_bytes: bytes, generate_heatmap: bool = True) -> dict:
    if not MODELS_READY:
        return {
            "status": "model_not_ready",
            "error": MODEL_LOAD_ERROR or "Model belum dimuat.",
            "precheck_status": None, "precheck_confidence": None,
            "prediction_label": None, "prediction_confidence": None,
            "all_probabilities": None, "gradcam_path": None,
        }

    x = bytes_to_array(file_bytes)   # numpy float32, shape [1,3,IMG_SIZE,IMG_SIZE]
    input_name = precheck_session.get_inputs()[0].name

    precheck_logits = precheck_session.run(None, {input_name: x})[0][0]
    brain_prob = float(_softmax(precheck_logits)[PRECHECK_CLASSES.index("Brain")])

    if brain_prob < PRECHECK_THRESHOLD:
        return {
            "status": "ok",
            "precheck_status": "invalid", "precheck_confidence": round(brain_prob, 4),
            "prediction_label": None, "prediction_confidence": None,
            "all_probabilities": None, "gradcam_path": None,
        }

    main_input_name = main_session.get_inputs()[0].name
    main_logits = main_session.run(None, {main_input_name: x})[0][0]
    probs = _softmax(main_logits)
    idx = int(probs.argmax())

    # Grad-CAM tidak tersedia di versi ONNX (butuh gradient). Lihat catatan di atas.
    gradcam_relative_path = None

    return {
        "status": "ok",
        "precheck_status": "valid", "precheck_confidence": round(brain_prob, 4),
        "prediction_label": MAIN_CLASSES[idx], "prediction_confidence": round(float(probs[idx]), 4),
        "all_probabilities": {cls: round(float(p), 4) for cls, p in zip(MAIN_CLASSES, probs)},
        "gradcam_path": gradcam_relative_path,
    }
