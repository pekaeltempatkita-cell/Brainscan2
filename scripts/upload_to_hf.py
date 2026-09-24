"""
upload_to_hf.py — Upload file .onnx hasil export ke Hugging Face Hub.
Jalankan SEKALI setelah scripts/export_to_onnx.py selesai.

Sebelum jalan:
1. pip install huggingface_hub
2. huggingface-cli login   (atau set env HF_TOKEN)
3. Bikin model repo baru di https://huggingface.co/new (misal: namamu/neurocheck-onnx)
4. Edit HF_REPO_ID di bawah sesuai repo kamu.
"""
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_REPO_ID = "namamu/neurocheck-onnx"   # <-- GANTI sesuai repo Hugging Face kamu
ONNX_DIR = Path(__file__).resolve().parent.parent / "onnx_models"

FILES = ["precheck_brain_gate.onnx", "hybrid_vit_efficientnet_brain.onnx"]


def main():
    api = HfApi()
    create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)

    for fname in FILES:
        fpath = ONNX_DIR / fname
        if not fpath.exists():
            print(f"[SKIP] {fname} tidak ditemukan di {ONNX_DIR} -- jalankan export_to_onnx.py dulu.")
            continue
        print(f"Uploading {fname} ...")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=HF_REPO_ID,
            repo_type="model",
        )
        print(f"[OK] {fname} ter-upload ke https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
