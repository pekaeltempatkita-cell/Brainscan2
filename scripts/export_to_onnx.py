"""
export_to_onnx.py — Convert checkpoint .pth (hasil training) -> .onnx.

JALANKAN INI LOKAL/DI COLAB (bukan di server produksi), sekali saja tiap kali
model baru selesai training. Butuh torch + torchvision terpasang (lihat
requirements-export.txt). Hasil .onnx-nya nanti diupload ke Hugging Face Hub,
lalu backend ringan (onnxruntime, TANPA torch) yang download & pakai file itu.

Cara jalanin (dari root folder project, venv yang ADA torch):
    python scripts/export_to_onnx.py
"""
import sys
from pathlib import Path

import torch
import onnx
import onnxruntime
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import (
    MAIN_MODEL_PTH_PATH, PRECHECK_MODEL_PTH_PATH, MAIN_CLASSES, PRECHECK_CLASSES, IMG_SIZE,
)
from models.precheck_model import PrecheckModel
from models.classifier_model import HybridViTEfficientNet

OUT_DIR = Path(__file__).resolve().parent.parent / "onnx_models"
OUT_DIR.mkdir(exist_ok=True)


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_one(model, dummy_input, out_path, input_names, output_names):
    print(f"\n=== Export -> {out_path.name} ===")

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        dynamo=False,
    )

    # Validasi struktur ONNX
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)

    print(f"[OK] Struktur ONNX valid: {out_path}")

    # Bandingkan PyTorch vs ONNX Runtime
    with torch.no_grad():
        torch_out = model(dummy_input).cpu().numpy()

    session = onnxruntime.InferenceSession(
        str(out_path),
        providers=["CPUExecutionProvider"],
    )

    onnx_out = session.run(
        None,
        {input_names[0]: dummy_input.cpu().numpy()},
    )[0]

    max_diff = np.abs(torch_out - onnx_out).max()

    print(
        f"[CEK] Selisih maksimum PyTorch vs ONNX: "
        f"{max_diff:.6f}"
    )

    if max_diff > 1e-3:
        print(
            "[PERINGATAN] Selisih agak besar. "
            "Cek arsitektur/preprocessing."
        )
    else:
        print("[OK] Output PyTorch dan ONNX cocok.")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[INFO] Ukuran file: {size_mb:.1f} MB")


def main():
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    # --- Model 1: Precheck (Brain vs Non-Brain) ---
    precheck = load_checkpoint(PrecheckModel(num_classes=len(PRECHECK_CLASSES)), PRECHECK_MODEL_PTH_PATH)
    export_one(
        precheck, dummy, OUT_DIR / "precheck_brain_gate.onnx",
        input_names=["image"], output_names=["logits"],
    )

    # --- Model 2: Model utama (10 kelas penyakit) ---
    main_model = load_checkpoint(HybridViTEfficientNet(num_classes=len(MAIN_CLASSES)), MAIN_MODEL_PTH_PATH)
    export_one(
        main_model, dummy, OUT_DIR / "hybrid_vit_efficientnet_brain.onnx",
        input_names=["image"], output_names=["logits"],
    )

    print(f"\nSelesai. File .onnx ada di: {OUT_DIR}")
    print("Langkah selanjutnya: upload kedua file .onnx ini ke Hugging Face Hub (lihat README_ONNX.md).")


if __name__ == "__main__":
    main()
