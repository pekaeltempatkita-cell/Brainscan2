"""
report_generator.py — Bikin laporan hasil pemeriksaan dalam bentuk PDF,
formatnya niru struktur laporan radiologi (Identitas -> Ringkasan Temuan ->
Analisis Klinis -> Tingkat Keyakinan -> Rekomendasi -> disclaimer).

Dipanggil dari app.py pas user klik "Unduh Laporan (PDF)".
"""
import json
import re
import xml.sax.saxutils as saxutils
from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable,
)

from config import REPORTS_DIR, FIGURES_DIR, APP_NAME
from disease_info import get_disease_info

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", fontSize=13, leading=16, alignment=TA_CENTER,
                           spaceAfter=2, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="ReportSubtitle", fontSize=9, leading=12, alignment=TA_CENTER,
                           textColor=colors.HexColor("#5b6b85")))
styles.add(ParagraphStyle(name="SectionHeader", fontSize=11, leading=14, spaceBefore=12,
                           spaceAfter=6, fontName="Helvetica-Bold", textColor=colors.HexColor("#0b1c3f")))
styles.add(ParagraphStyle(name="BulletItem", fontSize=9.5, leading=14, leftIndent=10,
                           spaceAfter=4))
styles.add(ParagraphStyle(name="BodyMuted", fontSize=9, leading=13,
                           textColor=colors.HexColor("#5b6b85")))
styles.add(ParagraphStyle(name="DisclaimerStyle", fontSize=8, leading=11,
                           textColor=colors.HexColor("#5b6b85")))


def _bullets(items):
    return [Paragraph(f"&bull;&nbsp;&nbsp;{text}", styles["BulletItem"]) for text in items]


def generate_report_pdf(patient: dict, prediction: dict) -> Path:
    """
    patient: dict dari database.get_patient_by_nik/get_patient_by_id
    prediction: dict dari database.get_prediction_by_id
    Return: Path file PDF yang sudah dibuat di REPORTS_DIR.
    """
    filename = f"laporan_{patient['nik']}_{prediction['id']}.pdf"
    out_path = REPORTS_DIR / filename

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        topMargin=18 * mm, bottomMargin=16 * mm, leftMargin=18 * mm, rightMargin=18 * mm,
    )
    story = []

    is_valid = prediction.get("precheck_status") == "valid" and prediction.get("prediction_label")
    label = prediction.get("prediction_label")
    info = get_disease_info(label) if is_valid else None

    # --- Header ---
    story.append(Paragraph(APP_NAME, styles["ReportTitle"]))
    story.append(Paragraph("Laporan Hasil Pemeriksaan Skrining Otak Berbasis AI (Opini Sekunder)",
                            styles["ReportSubtitle"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#e2e8f5")))
    story.append(Spacer(1, 10))

    # --- I. Identitas Pasien & Pemeriksaan ---
    story.append(Paragraph("I. IDENTITAS PASIEN &amp; PEMERIKSAAN", styles["SectionHeader"]))
    tanggal_analisis = prediction.get("upload_time", "")
    try:
        tanggal_analisis = datetime.fromisoformat(tanggal_analisis).strftime("%A, %d %B %Y %H:%M")
    except Exception:
        pass

    cell = lambda text: Paragraph(str(text), styles["BodyMuted"])
    identity_rows = [
        ["Nama Pasien", cell(patient.get("nama", "-")), "Jenis Kelamin", cell(patient.get("jenis_kelamin", "-"))],
        ["NIK", cell(patient.get("nik", "-")), "Tanggal Lahir", cell(patient.get("tanggal_lahir", "-"))],
        ["Umur saat Scan", cell(prediction.get("umur_saat_scan") or "-"), "Jenis Scan", cell(prediction.get("jenis_scan") or "-")],
        ["Tanggal Analisis", cell(tanggal_analisis), "No. Telepon", cell(patient.get("no_telepon") or "-")],
        ["Gejala (dilaporkan)", cell(prediction.get("gejala") or "-"), "", ""],
    ]
    id_table = Table(identity_rows, colWidths=[80, 145, 75, 125])
    id_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#5b6b85")),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#5b6b85")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("SPAN", (1, 4), (3, 4)),
    ]))
    story.append(id_table)

    story.append(Spacer(1, 4))

    if not is_valid:
        # --- Kasus gambar bukan citra otak (precheck gagal) ---
        story.append(Paragraph("II. HASIL PEMERIKSAAN", styles["SectionHeader"]))
        story.append(Paragraph(
            f"Sistem precheck AI mendeteksi bahwa file yang diunggah kemungkinan BUKAN citra "
            f"MRI/CT otak (confidence deteksi otak: "
            f"{(prediction.get('precheck_confidence') or 0) * 100:.1f}%). Tidak ada klasifikasi "
            f"penyakit yang dihasilkan. Silakan unggah ulang dengan citra MRI/CT otak yang jelas.",
            styles["BodyMuted"],
        ))
    else:
        # --- II. Ringkasan Temuan ---
        story.append(Paragraph("II. RINGKASAN TEMUAN (FINDINGS SUMMARY)", styles["SectionHeader"]))
        story.extend(_bullets(info["temuan"]))

        # --- III. Analisis Klinis ---
        story.append(Paragraph("III. ANALISIS KLINIS (CLINICAL ANALYSIS)", styles["SectionHeader"]))
        story.extend(_bullets(info["analisis"]))

        # --- IV. Tingkat Keyakinan ---
        story.append(Paragraph("IV. TINGKAT KEYAKINAN (CONFIDENCE LEVEL)", styles["SectionHeader"]))
        conf = (prediction.get("prediction_confidence") or 0) * 100
        story.extend(_bullets([
            f"Prediksi utama sistem AI: <b>{info['nama_tampilan']}</b> dengan tingkat keyakinan "
            f"<b>{conf:.2f}%</b>.",
            f"Tingkat urgensi (referensi awal): <b>{info['urgensi']}</b>.",
            "Tingkat keyakinan AI tetap memerlukan penilaian kualitatif oleh tenaga medis "
            "profesional/radiolog yang berwenang.",
        ]))

        # Tabel probabilitas semua kelas (kalau tersedia)
        all_probs = prediction.get("all_probabilities")
        if all_probs:
            try:
                probs_dict = json.loads(all_probs) if isinstance(all_probs, str) else all_probs
                sorted_probs = sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True)
                rows = [["Kelas", "Probabilitas"]] + [
                    [get_disease_info(k)["nama_tampilan"], f"{v * 100:.2f}%"] for k, v in sorted_probs
                ]
                prob_table = Table(rows, colWidths=[300, 100])
                prob_table.setStyle(TableStyle([
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3fb")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f5")),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(Spacer(1, 6))
                story.append(prob_table)
            except Exception:
                pass

        # Grad-CAM image kalau ada
        gradcam_path = prediction.get("gradcam_path")
        if gradcam_path:
            full_path = FIGURES_DIR / gradcam_path
            if full_path.exists():
                story.append(Spacer(1, 10))
                story.append(Paragraph(
                    "Visualisasi area perhatian model (Grad-CAM / explainability):",
                    styles["BodyMuted"],
                ))
                story.append(Spacer(1, 4))
                story.append(Image(str(full_path), width=200, height=200))

        # --- V. Rekomendasi ---
        story.append(Paragraph("V. REKOMENDASI PEMERIKSAAN LANJUTAN", styles["SectionHeader"]))
        story.extend(_bullets(info["rekomendasi"]))

        # Penjelasan tambahan (Gemini / fallback)
        if prediction.get("gemini_explanation"):
            story.append(Paragraph("VI. PENJELASAN &amp; LANGKAH SELANJUTNYA", styles["SectionHeader"]))
            raw = prediction["gemini_explanation"]
            # escape dulu biar aman, baru ubah **bold** -> <b> dan newline -> <br/> (reportlab
            # Paragraph cuma support subset tag HTML ini)
            safe = saxutils.escape(raw)
            safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
            safe = safe.replace("\n", "<br/>")
            story.append(Paragraph(safe, styles["BodyMuted"]))

    # --- Disclaimer & footer ---
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#e2e8f5")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Catatan: Hasil analisis berbasis sistem kecerdasan buatan (AI) ini bersifat sebagai "
        "opini sekunder dan alat bantu keputusan klinis (decision support tool). Laporan ini "
        "BUKAN diagnosis final dan tetap memerlukan evaluasi, korelasi klinis penuh, dan "
        "validasi resmi dari Dokter Spesialis yang berwenang sebelum digunakan sebagai dasar "
        "tindakan medis.",
        styles["DisclaimerStyle"],
    ))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Dicetak otomatis oleh sistem {APP_NAME} pada "
                            f"{datetime.now().strftime('%A, %d %B %Y %H:%M')}.", styles["DisclaimerStyle"]))

    doc.build(story)
    return out_path
