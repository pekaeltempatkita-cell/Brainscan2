"""
api.py — Endpoint JSON tipis di atas inference.py, buat kebutuhan integrasi
programatik (mis. dipanggil dari aplikasi mobile/eksternal), terpisah dari
alur HTML utama di app.py.

Di-mount di app.py lewat: app.include_router(api.router, prefix="/api")
"""
from fastapi import APIRouter, UploadFile, File, HTTPException

import inference
from config import MAIN_CLASSES

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "app_status": "ok",
        "models_ready": inference.MODELS_READY,
        "model_error": inference.MODEL_LOAD_ERROR,
    }


@router.get("/classes")
def list_classes():
    return {"classes": MAIN_CLASSES}


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Prediksi langsung dari 1 file gambar, TANPA menyimpan ke database pasien.
    Cocok buat testing cepat model tanpa perlu alur registrasi pasien penuh."""
    if not inference.MODELS_READY:
        raise HTTPException(status_code=503, detail=inference.MODEL_LOAD_ERROR)

    file_bytes = await file.read()
    result = inference.run_pipeline(file_bytes, generate_heatmap=False)
    return result
