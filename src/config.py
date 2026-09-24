# config.py — NeuroCheck configuration settings
import os
from dotenv import load_dotenv

load_dotenv()

PORT = int(os.getenv("PORT", 3000))
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "dev-secret-neurocheck-key")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
HF_REPO_ID = os.getenv("HF_REPO_ID", "pekaeltempatkita/neurocheck-models")
HF_MAIN_MODEL_FILENAME = os.getenv("HF_MAIN_MODEL_FILENAME", "model_main.onnx")
HF_PRECHECK_FILENAME = os.getenv("HF_PRECHECK_FILENAME", "model_precheck.onnx")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
