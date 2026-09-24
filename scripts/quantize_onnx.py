"""
quantize_onnx.py — Kompres file .onnx pakai INT8 dynamic quantization.

Kenapa perlu: file .onnx hasil export biasanya masih FP32 (4 byte per angka).
Dynamic quantization ubah bobot Linear/Conv jadi INT8 (1 byte per angka) --
otomatis, TANPA butuh data kalibrasi tambahan, TANPA training ulang.
Efeknya: ukuran file turun ~2-4x, RAM saat inferensi juga ikut turun (walau
gak sebesar penurunan ukuran file, karena aktivasi tetap FP32).

Trade-off: akurasi bisa turun sedikit (biasanya <1% buat model klasifikasi
gambar), tapi worth banget kalau syaratnya app harus muat di RAM 512MB
host gratis. Selalu cek "[CEK] Selisih" di bawah sebelum dipakai produksi --
kalau melenceng jauh, jangan dipakai versi quantized-nya.

Cara jalanin (SETELAH scripts/export_to_onnx.py selesai, folder onnx_models/
sudah ada isinya):
    python scripts/quantize_onnx.py
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import IMG_SIZE

ONNX_DIR = Path(__file__).resolve().parent.parent / "onnx_models"
FILES = ["precheck_brain_gate.onnx", "hybrid_vit_efficientnet_brain.onnx"]


def quantize_one(fname: str):
    src_path = ONNX_DIR / fname
    dst_path = ONNX_DIR / fname.replace(".onnx", "_int8.onnx")
    preprocessed_path = ONNX_DIR / fname.replace(".onnx", "_preprocessed.onnx")

    if not src_path.exists():
        print(f"[SKIP] {fname} tidak ditemukan -- jalankan export_to_onnx.py dulu.")
        return

    # WAJIB: model hasil torch.onnx dynamo export butuh shape inference tambahan
    # ini dulu -- tanpa ini, quantize_dynamic() bisa error "ShapeInferenceError"
    # karena graph-nya punya shape simbolik yang belum "diselesaikan".
    print(f"\n=== Pre-process (shape inference) -> {fname} ===")
    quant_pre_process(str(src_path), str(preprocessed_path), skip_symbolic_shape=False)

    print(f"=== Quantize -> {dst_path.name} ===")
    quantize_dynamic(
        model_input=str(preprocessed_path),
        model_output=str(dst_path),
        weight_type=QuantType.QUInt8,
    )
    preprocessed_path.unlink()  # file antara, gak perlu disimpan

    size_before = src_path.stat().st_size / (1024 * 1024)
    size_after = dst_path.stat().st_size / (1024 * 1024)
    print(f"[INFO] Ukuran: {size_before:.1f} MB -> {size_after:.1f} MB "
          f"({size_before / size_after:.1f}x lebih kecil)")

    # Bandingkan output model asli (FP32) vs quantized (INT8) pakai input acak yang SAMA
    dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

    sess_fp32 = onnxruntime.InferenceSession(str(src_path), providers=["CPUExecutionProvider"])
    sess_int8 = onnxruntime.InferenceSession(str(dst_path), providers=["CPUExecutionProvider"])
    input_name = sess_fp32.get_inputs()[0].name

    out_fp32 = sess_fp32.run(None, {input_name: dummy})[0]
    out_int8 = sess_int8.run(None, {input_name: dummy})[0]

    max_diff = np.abs(out_fp32 - out_int8).max()
    # softmax kedua output, bandingkan prediksi kelas (top-1) -- ini yang paling penting
    pred_fp32 = out_fp32.argmax()
    pred_int8 = out_int8.argmax()
    print(f"[CEK] Selisih logit maksimum: {max_diff:.4f}")
    print(f"[CEK] Prediksi kelas (top-1) sama? {'YA' if pred_fp32 == pred_int8 else 'TIDAK -- HATI-HATI'}")
    if max_diff > 1.0:
        print("[PERINGATAN] Selisih logit lumayan besar. Cek beberapa contoh gambar asli "
              "(bukan cuma random noise) sebelum pakai versi quantized ini di produksi.")


def main():
    for fname in FILES:
        quantize_one(fname)

    print("\nSelesai. File *_int8.onnx ada di folder onnx_models/, di samping versi FP32 aslinya.")
    print("Kalau hasilnya OK, pakai file _int8.onnx buat diupload ke Hugging Face Hub "
          "(ganti nama filenya di HF_PRECHECK_FILENAME/HF_MAIN_MODEL_FILENAME di .env "
          "sesuai nama file _int8.onnx, atau rename dulu sebelum upload).")


if __name__ == "__main__":
    main()