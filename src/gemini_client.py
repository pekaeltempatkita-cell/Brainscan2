"""
gemini_client.py — Minta penjelasan medis singkat dari Gemini untuk hasil klasifikasi.
Kalau GEMINI_API_KEY kosong atau API error, otomatis fallback ke template
statis dari disease_info.py supaya aplikasi tetap jalan tanpa API key.
"""
from config import GEMINI_API_KEY
from prompts import build_explanation_prompt
from disease_info import get_disease_info

_model = None
_GEMINI_READY = False

if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _model = genai.GenerativeModel("gemini-2.0-flash")
        _GEMINI_READY = True
    except Exception as e:
        print(f"[Gemini init error] Gagal setup Gemini, pakai fallback offline. Detail: {e}")


def _offline_explanation(label: str, confidence: float) -> str:
    """Fallback penjelasan tanpa panggil API eksternal, dari template disease_info.py.
    Strukturnya sengaja dibikin mirip output Gemini (penjelasan -> langkah selanjutnya ->
    arah solusi -> spesialis -> disclaimer) biar konsisten dari sisi pengguna."""
    info = get_disease_info(label)
    kalimat_temuan = info["analisis"][0] if info["analisis"] else ""
    langkah_selanjutnya = " ".join(info["rekomendasi"][:2]) if info["rekomendasi"] else "-"
    arah_solusi = info["rekomendasi"][-1] if info["rekomendasi"] else "-"

    return (
        f"Apa itu kondisi ini? Hasil skrining AI mengindikasikan '{info['nama_tampilan']}' dengan "
        f"tingkat keyakinan {confidence * 100:.1f}%. {kalimat_temuan}\n\n"
        f"Langkah selanjutnya: {langkah_selanjutnya}\n\n"
        f"Arah solusi/penanganan: {arah_solusi}\n\n"
        f"Spesialis yang perlu dikonsultasikan: {info['spesialis']}.\n\n"
        f"Catatan: ini hanya alat bantu skrining awal, BUKAN diagnosis final -- hasil wajib "
        f"dikonfirmasi oleh dokter/radiolog yang berwenang sebelum dipakai sebagai dasar "
        f"keputusan medis apapun."
    )


def get_explanation(label: str, confidence: float) -> str:
    """Selalu return string penjelasan -- pakai Gemini kalau siap, kalau tidak pakai fallback."""
    if _GEMINI_READY:
        try:
            prompt = build_explanation_prompt(label, confidence)
            response = _model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"[Gemini error] {e} -- pakai fallback offline.")
    return _offline_explanation(label, confidence)
