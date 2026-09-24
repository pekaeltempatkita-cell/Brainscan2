"""
explainability.py — Grad-CAM (versi TORCH, LAMA).

CATATAN PENTING: sejak backend pindah ke onnxruntime (lihat inference.py),
file ini SUDAH TIDAK DIPAKAI, karena Grad-CAM butuh gradient/backward pass
yang tidak tersedia di onnxruntime inference session biasa. Dibiarkan di
sini cuma buat referensi/kalau suatu saat balik pakai torch untuk serving.
Kalau mau heatmap tetap ada di versi ONNX, opsinya pakai Score-CAM
(gradient-free, forward-pass berkali-kali) -- minta bantuan kalau mau itu
diimplementasikan.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM generik: cocok untuk model apapun asal ada 1 conv feature map target."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Return (heatmap 0..1 shape [H,W], class_idx yang dipakai).
        input_tensor: shape [1, 3, H, W], sudah di device yang sama dengan model.
        """
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)   # [1, num_classes]

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        score = output[0, class_idx]
        score.backward()

        # Global-average-pool gradien per channel -> bobot pentingnya tiap channel
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)          # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)      # [1, 1, h, w]
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx


def overlay_heatmap_on_image(heatmap: np.ndarray, original_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    heatmap: array 2D nilai 0..1 (ukuran boleh beda dari original_rgb, akan di-resize).
    original_rgb: array HxWx3 uint8 (RGB, BUKAN dinormalisasi).
    Return: array HxWx3 uint8 (RGB) hasil overlay.
    """
    h, w = original_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)   # BGR
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (original_rgb.astype(np.float32) * (1 - alpha)
               + heatmap_color_rgb.astype(np.float32) * alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def generate_gradcam_overlay(model, input_tensor, original_rgb: np.ndarray,
                              class_idx: int, target_layer, save_path) -> bool:
    """
    Hitung Grad-CAM lalu simpan hasil overlay PNG ke `save_path`.
    Return True kalau berhasil, False kalau gagal (mis. arsitektur berubah).
    Note: butuh gradient, jadi jangan panggil di dalam blok `torch.no_grad()`.
    """
    try:
        was_training = model.training
        model.eval()
        cam_tool = GradCAM(model, target_layer)
        # requires_grad harus aktif di input buat backward jalan
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        heatmap, _ = cam_tool.generate(input_tensor, class_idx=class_idx)
        overlay = overlay_heatmap_on_image(heatmap, original_rgb)
        cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if was_training:
            model.train()
        return True
    except Exception as e:
        print(f"[GradCAM error] Gagal generate heatmap: {e}")
        return False
