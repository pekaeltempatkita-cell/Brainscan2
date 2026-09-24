"""
database.py — Simpan & ambil riwayat hasil prediksi user.

Mendukung DUA mode, otomatis dipilih berdasarkan .env:
- Kalau TURSO_DATABASE_URL diisi -> connect ke Turso (libSQL) lewat paket `libsql`.
- Kalau kosong -> fallback ke file SQLite lokal (data/brainscan.db), berguna
  buat development di komputer sendiri tanpa perlu akun Turso.

Query SQL-nya sama persis di kedua mode (libSQL kompatibel dengan SQLite),
jadi gak perlu ada percabangan logic SELECT/INSERT di bawah.
"""
import os
import json
from datetime import datetime
from contextlib import contextmanager

from config import DATABASE_PATH

TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "")
USE_TURSO = bool(TURSO_DATABASE_URL)


def _raw_connect():
    if USE_TURSO:
        import libsql
        return libsql.connect(database=TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN)
    else:
        import sqlite3
        return sqlite3.connect(DATABASE_PATH)


@contextmanager
def get_connection():
    """Context manager biar koneksi DB selalu ditutup rapi, walau ada error."""
    conn = _raw_connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _rows_to_dicts(cursor):
    """Ubah hasil cursor.fetchall() jadi list of dict pakai cursor.description
    (bukan sqlite3.Row) -- ini yang bikin kode ini jalan sama di sqlite3 MAUPUN
    libsql tanpa perlu percabangan kode."""
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _row_to_dict(cursor, row):
    if row is None:
        return None
    columns = [col[0] for col in cursor.description]
    return dict(zip(columns, row))


def _add_column_if_missing(conn, table, column, coltype):
    """SQLite/libSQL gak punya 'ADD COLUMN IF NOT EXISTS', jadi dicek manual dulu."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cursor.fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db():
    """Bikin tabel kalau belum ada + migrasi kolom baru. Panggil ini SEKALI pas aplikasi start."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                upload_time TEXT NOT NULL,
                precheck_status TEXT NOT NULL,
                precheck_confidence REAL NOT NULL,
                prediction_label TEXT,
                prediction_confidence REAL,
                all_probabilities TEXT,
                gradcam_path TEXT,
                gemini_explanation TEXT
            )
        """)
        _add_column_if_missing(conn, "predictions", "jenis_scan", "TEXT")
        _add_column_if_missing(conn, "predictions", "umur_saat_scan", "TEXT")
        _add_column_if_missing(conn, "predictions", "gejala", "TEXT")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nik TEXT UNIQUE NOT NULL,
                nama TEXT NOT NULL,
                tanggal_lahir TEXT,
                jenis_kelamin TEXT,
                alamat TEXT,
                no_telepon TEXT,
                created_at TEXT NOT NULL,
                created_by_email TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS medical_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                prediction_id INTEGER,
                catatan_dokter TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(id),
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        """)
    mode = "Turso (libSQL)" if USE_TURSO else f"SQLite lokal ({DATABASE_PATH})"
    print(f"Database siap, mode: {mode}")


def save_prediction(filename, precheck_status, precheck_confidence,
                     prediction_label=None, prediction_confidence=None,
                     all_probabilities=None, gradcam_path=None, gemini_explanation=None,
                     jenis_scan=None, umur_saat_scan=None, gejala=None):
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO predictions (
                filename, upload_time, precheck_status, precheck_confidence,
                prediction_label, prediction_confidence, all_probabilities,
                gradcam_path, gemini_explanation, jenis_scan, umur_saat_scan, gejala
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            filename, datetime.now().isoformat(), precheck_status, precheck_confidence,
            prediction_label, prediction_confidence,
            json.dumps(all_probabilities) if all_probabilities else None,
            gradcam_path, gemini_explanation, jenis_scan, umur_saat_scan, gejala,
        ))
        return cursor.lastrowid


def get_history(limit=50):
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM predictions ORDER BY upload_time DESC LIMIT ?", (limit,))
        return _rows_to_dicts(cursor)


def get_prediction_by_id(prediction_id):
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
        return _row_to_dict(cursor, cursor.fetchone())


def create_patient(nik, nama, tanggal_lahir, jenis_kelamin, alamat, no_telepon, created_by_email):
    try:
        with get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO patients (nik, nama, tanggal_lahir, jenis_kelamin, alamat, no_telepon, created_at, created_by_email)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (nik, nama, tanggal_lahir, jenis_kelamin, alamat, no_telepon, datetime.now().isoformat(), created_by_email))
            return cursor.lastrowid
    except Exception as e:
        # Sqlite3: sqlite3.IntegrityError. libsql: exception constraint UNIQUE juga,
        # tapi kelasnya beda -- dicek lewat pesan errornya biar aman di kedua mode.
        if "UNIQUE" in str(e).upper():
            return None  # NIK sudah terdaftar
        raise


def get_patient_by_nik(nik):
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM patients WHERE nik = ?", (nik,))
        return _row_to_dict(cursor, cursor.fetchone())


def get_patient_by_id(patient_id):
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
        return _row_to_dict(cursor, cursor.fetchone())


def get_all_patients():
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM patients ORDER BY created_at DESC")
        return _rows_to_dicts(cursor)


def add_medical_record(patient_id, prediction_id, catatan_dokter=None):
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO medical_records (patient_id, prediction_id, catatan_dokter, created_at)
            VALUES (?, ?, ?, ?)
        """, (patient_id, prediction_id, catatan_dokter, datetime.now().isoformat()))
        return cursor.lastrowid


def get_medical_records_by_patient(patient_id):
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT mr.id, mr.prediction_id, mr.catatan_dokter, mr.created_at,
                   p.filename, p.prediction_label, p.prediction_confidence,
                   p.gemini_explanation, p.gradcam_path, p.precheck_status,
                   p.jenis_scan, p.umur_saat_scan, p.gejala
            FROM medical_records mr
            JOIN predictions p ON mr.prediction_id = p.id
            WHERE mr.patient_id = ?
            ORDER BY mr.created_at DESC
        """, (patient_id,))
        return _rows_to_dicts(cursor)


def get_all_medical_records():
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT mr.id, mr.prediction_id, mr.created_at, mr.catatan_dokter,
                   p.nik, p.nama AS nama_pasien,
                   pr.prediction_label, pr.prediction_confidence, pr.gemini_explanation,
                   pr.gradcam_path
            FROM medical_records mr
            JOIN patients p ON mr.patient_id = p.id
            JOIN predictions pr ON mr.prediction_id = pr.id
            ORDER BY mr.created_at DESC
        """)
        return _rows_to_dicts(cursor)
