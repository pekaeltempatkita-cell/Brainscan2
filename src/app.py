"""
app.py — Entry point aplikasi NeuroCheck, VERSI API MURNI.

Beda besar dari versi sebelumnya:
- TIDAK ADA lagi Jinja2/HTML rendering di sini -- semua endpoint balikin JSON
  (atau file PDF buat 1 endpoint laporan). Tampilan sekarang jadi tanggung
  jawab frontend statis terpisah (folder frontend/, di-deploy ke GitHub Pages).
- Login pakai JWT (lihat auth.py), BUKAN cookie session -- karena frontend &
  backend beda domain.
- CORS diaktifkan supaya domain GitHub Pages boleh manggil API ini.

Cara jalanin (dari dalam folder src/):
    uvicorn app:app --reload
"""
import os

from fastapi import FastAPI, Form, UploadFile, File, Depends, HTTPException, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import inference
import gemini_client
import api
from report_generator import generate_report_pdf
from disease_info import get_disease_info, DISEASE_INFO
from auth import verify_google_id_token, create_jwt, get_current_user, require_admin
from database import (
    init_db, create_patient, get_patient_by_nik,
    get_all_patients, get_medical_records_by_patient,
    save_prediction, add_medical_record, get_all_medical_records,
    get_prediction_by_id,
)

load_dotenv()

app = FastAPI(title="NeuroCheck API")

# --- CORS: izinkan frontend (GitHub Pages / localhost saat dev) manggil API ini ---
# Isi FRONTEND_ORIGIN di .env dengan URL GitHub Pages kamu, contoh:
# https://username.github.io  (TANPA trailing slash)
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_credentials=False,   # pakai Bearer token, bukan cookie -> gak butuh credentials
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api.router, prefix="/api")

init_db()

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")


def _model_warning():
    if inference.MODELS_READY:
        return None
    return inference.MODEL_LOAD_ERROR


@app.get("/api/status")
def status():
    """Dicek frontend di halaman Beranda -- info umum + peringatan kalau model belum siap."""
    return {"app": "NeuroCheck", "model_warning": _model_warning()}


@app.get("/api/disease-info")
def disease_info_all():
    """Dipakai halaman Informasi (frontend) -- daftar 10 kelas + deskripsi klinisnya."""
    return {label: get_disease_info(label) for label in DISEASE_INFO}


# ============================== AUTH ==============================

@app.post("/api/auth/google")
def auth_google(body: dict = Body(...)):
    """Frontend kirim {"credential": "<ID token dari Google Identity Services>"}."""
    credential = body.get("credential")
    if not credential:
        raise HTTPException(status_code=400, detail="Field 'credential' wajib diisi.")
    user_info = verify_google_id_token(credential)
    token = create_jwt({"email": user_info["email"], "name": user_info["name"], "role": "user"})
    return {"token": token, "user": user_info}


@app.post("/api/auth/admin/login")
def auth_admin_login(body: dict = Body(...)):
    username = body.get("username")
    password = body.get("password")
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        token = create_jwt({"username": username, "role": "admin"})
        return {"token": token}
    raise HTTPException(status_code=401, detail="Username atau password salah.")


# ============================== PATIENTS ==============================

@app.get("/api/patients")
def list_patients(user: dict = Depends(get_current_user)):
    return {"patients": get_all_patients()}


@app.post("/api/patients")
def register_patient(body: dict = Body(...), user: dict = Depends(get_current_user)):
    required = ["nik", "nama", "tanggal_lahir", "jenis_kelamin"]
    missing = [f for f in required if not body.get(f)]
    if missing:
        raise HTTPException(status_code=400, detail=f"Field wajib belum diisi: {', '.join(missing)}")

    new_id = create_patient(
        body["nik"], body["nama"], body["tanggal_lahir"], body["jenis_kelamin"],
        body.get("alamat", ""), body.get("no_telepon", ""), user.get("email", ""),
    )
    if new_id is None:
        raise HTTPException(status_code=409, detail=f"NIK {body['nik']} sudah terdaftar.")
    return {"id": new_id, "nik": body["nik"]}


@app.get("/api/patients/{nik}")
def patient_detail(nik: str, user: dict = Depends(get_current_user)):
    patient = get_patient_by_nik(nik)
    if not patient:
        raise HTTPException(status_code=404, detail="Pasien tidak ditemukan.")
    records = get_medical_records_by_patient(patient["id"])
    return {"patient": patient, "records": records, "model_warning": _model_warning()}


@app.post("/api/patients/{nik}/upload")
async def upload_scan(
    nik: str,
    file: UploadFile = File(...),
    jenis_scan: str = Form(""),
    umur_saat_scan: str = Form(""),
    gejala: str = Form(""),
    user: dict = Depends(get_current_user),
):
    patient = get_patient_by_nik(nik)
    if not patient:
        raise HTTPException(status_code=404, detail="Pasien tidak ditemukan.")

    file_bytes = await file.read()
    result = inference.run_pipeline(file_bytes)

    if result["status"] == "model_not_ready":
        raise HTTPException(status_code=503, detail=result["error"])

    if result["precheck_status"] != "valid":
        save_prediction(
            filename=file.filename, precheck_status="invalid",
            precheck_confidence=result["precheck_confidence"],
            jenis_scan=jenis_scan, umur_saat_scan=umur_saat_scan, gejala=gejala,
        )
        return {
            "precheck_status": "invalid",
            "precheck_confidence": result["precheck_confidence"],
            "message": "Gambar terdeteksi BUKAN citra MRI/CT otak.",
        }

    explanation = gemini_client.get_explanation(result["prediction_label"], result["prediction_confidence"])

    prediction_id = save_prediction(
        filename=file.filename, precheck_status="valid",
        precheck_confidence=result["precheck_confidence"],
        prediction_label=result["prediction_label"],
        prediction_confidence=result["prediction_confidence"],
        all_probabilities=result["all_probabilities"],
        gradcam_path=result.get("gradcam_path"),
        gemini_explanation=explanation,
        jenis_scan=jenis_scan, umur_saat_scan=umur_saat_scan, gejala=gejala,
    )
    add_medical_record(patient_id=patient["id"], prediction_id=prediction_id)

    return {
        "precheck_status": "valid",
        "prediction_id": prediction_id,
        "prediction_label": result["prediction_label"],
        "prediction_confidence": result["prediction_confidence"],
        "all_probabilities": result["all_probabilities"],
        "gemini_explanation": explanation,
        "info": get_disease_info(result["prediction_label"]),
    }


@app.get("/api/patients/{nik}/report/{prediction_id}")
def download_report(nik: str, prediction_id: int, user: dict = Depends(get_current_user)):
    """Auth di sini terima token lewat header ATAU ?token=... di URL (lihat auth.py),
    karena link unduh PDF bakal diklik langsung/dibuka tab baru oleh browser."""
    patient = get_patient_by_nik(nik)
    if not patient:
        raise HTTPException(status_code=404, detail="Pasien tidak ditemukan.")

    prediction = get_prediction_by_id(prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Data pemeriksaan tidak ditemukan.")

    pdf_path = generate_report_pdf(patient, prediction)
    filename = f"Laporan_{patient['nama'].replace(' ', '_')}_{prediction_id}.pdf"
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=filename)


# ============================== ADMIN ==============================

@app.get("/api/admin/overview")
def admin_overview(admin: dict = Depends(require_admin)):
    return {
        "patients": get_all_patients(),
        "records": get_all_medical_records(),
        "model_warning": _model_warning(),
    }
