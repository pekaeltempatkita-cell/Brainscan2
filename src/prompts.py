"""
prompts.py — Template prompt buat Gemini. Dipisah dari gemini_client.py
biar gampang di-tweak tanpa ubah logic pemanggilan API.
"""

EXPLANATION_PROMPT_TEMPLATE = """Kamu adalah asisten edukasi medis yang membantu menjelaskan hasil
skrining AI ke tenaga medis/pasien awam dengan bahasa Indonesia yang mudah dipahami dan empatik.

Hasil klasifikasi citra MRI/CT otak dari sistem AI:
- Prediksi: {label}
- Tingkat keyakinan (confidence): {confidence:.1f}%

Tulis penjelasan yang mencakup 4 bagian berikut (pakai heading singkat buat tiap bagian):

1. **Apa itu kondisi ini?** — Jelaskan kondisi "{label}" secara umum dengan bahasa awam (bukan
   jargon medis berat), 2-3 kalimat. Termasuk gejala umum yang biasa menyertai kondisi ini.

2. **Langkah selanjutnya** — Apa yang sebaiknya segera dilakukan pasien/keluarga setelah menerima
   hasil ini (mis. ke IGD, jadwalkan konsultasi, pemeriksaan penunjang apa yang biasanya
   diperlukan). Sesuaikan urgensinya dengan jenis kondisi (kondisi gawat darurat vs kondisi yang
   bisa dijadwalkan).

3. **Arah solusi / penanganan** — Gambaran umum arah penanganan/pengobatan yang BIASANYA dilakukan
   untuk kondisi ini (mis. observasi, obat-obatan, tindakan bedah, terapi), tanpa merekomendasikan
   dosis, obat spesifik, atau rencana pengobatan pasti untuk pasien ini.

4. **Spesialis yang perlu dikonsultasikan** — Sebutkan dokter spesialis yang paling relevan.

Tutup dengan satu kalimat penegasan bahwa ini HANYA alat bantu skrining awal (decision support),
BUKAN diagnosis final, dan hasil WAJIB dikonfirmasi oleh dokter/radiolog yang berwenang sebelum
dipakai sebagai dasar keputusan medis apapun.

Jangan menyebutkan angka statistik lain selain confidence yang sudah diberikan. Jangan membuat
klaim kepastian diagnosis, dan jangan memberi rekomendasi dosis obat atau resep spesifik.
"""


def build_explanation_prompt(label: str, confidence: float) -> str:
    return EXPLANATION_PROMPT_TEMPLATE.format(label=label, confidence=confidence * 100)
