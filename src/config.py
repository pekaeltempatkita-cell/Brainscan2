"""
config.py — Pusat konfigurasi aplikasi.
Semua "angka ajaib" dan path ditaruh di sini, BUKAN di-hardcode di file lain.

CATATAN VERSI ONNX: file ini SENGAJA tidak import `torch` -- backend produksi
pakai onnxruntime, bukan torch. Kalau butuh training/export ulang model,
jalankan scripts/export_to_onnx.py pakai venv terpisah yang ada torch-nya
(lihat requirements-export.txt).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# --- Load file .env (isinya API key, dll -- gak ikut ke-push ke GitHub) ---
load_dotenv()

# --- Path dasar proyek ---
BASE_DIR = Path(__file__).resolve().parent.parent   # folder root project (di atas src/)
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = SRC_DIR / "models"

OUTPUTS_DIR = BASE_DIR / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"        # FIX: nested di dalam outputs/, sesuai struktur asli
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "brainscan.db"            # BARU: riwayat prediksi user

TEMP_UPLOADS_DIR = BASE_DIR / "temp_uploads"

# Pastikan folder-folder ini ada (auto-buat kalau belum ada)
for folder in [CHECKPOINTS_DIR, TEMP_UPLOADS_DIR, FIGURES_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# --- Path checkpoint model LOKAL, dipakai backend (opsional -- kalau tidak
#     ada, inference.py otomatis download .onnx dari Hugging Face Hub lewat
#     HF_REPO_ID di bawah). Ini format ONNX, dipakai onnxruntime. ---
MAIN_MODEL_PATH = CHECKPOINTS_DIR / "hybrid_vit_efficientnet_brain.onnx"
PRECHECK_MODEL_PATH = CHECKPOINTS_DIR / "precheck_brain_gate.onnx"

# --- Path checkpoint ASLI (.pth), dipakai HANYA oleh scripts/export_to_onnx.py
#     di Colab -- BUKAN dipakai backend. Sesuaikan nama file di sini kalau nama
#     hasil training kamu beda. ---
MAIN_MODEL_PTH_PATH = CHECKPOINTS_DIR / "hybrid_vit_efficientnet_brain_best.pth"
PRECHECK_MODEL_PTH_PATH = CHECKPOINTS_DIR / "precheck_brain_gate_best.pth"

# --- Hugging Face Hub (sumber utama file .onnx di produksi) ---
# Isi repo ini dengan hasil `scripts/upload_to_hf.py`. Bisa dioverride lewat .env.
HF_REPO_ID = os.getenv("HF_REPO_ID", "namamu/neurocheck-onnx")
HF_PRECHECK_FILENAME = os.getenv("HF_PRECHECK_FILENAME", "precheck_brain_gate.onnx")
HF_MAIN_MODEL_FILENAME = os.getenv("HF_MAIN_MODEL_FILENAME", "hybrid_vit_efficientnet_brain.onnx")

# --- Kelas-kelas model (URUTAN HARUS SAMA PERSIS kayak pas training!) ---
MAIN_CLASSES = [
    "Alzheimer_Mild", "Alzheimer_Moderate", "Alzheimer_Very_Mild",
    "Intracranial_Hemorrhage", "Multiple_Sclerosis", "Normal_Healthy",
    "Stroke_Iskemik", "Tumor_Glioma", "Tumor_Meningioma", "Tumor_Pituitary",
]
PRECHECK_CLASSES = ["Non_Brain", "Brain"]

# --- Konfigurasi gambar (HARUS SAMA kayak training) ---
# TERKONFIRMASI: 300 sudah dicocokkan langsung dengan shape pos_embed checkpoint asli
# (EfficientNet-B3 pada input 300x300 -> feature map 10x10 -> 101 token, cocok persis).
IMG_SIZE = 300
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Threshold precheck (di bawah ini dianggap "bukan gambar otak") ---
PRECHECK_THRESHOLD = 0.5

# --- API Keys (dari .env, JANGAN ditulis langsung di sini) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- Info aplikasi ---
APP_NAME = "NeuroCheck"
APP_VERSION = "1.0.0"