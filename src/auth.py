"""
auth.py — Auth berbasis JWT (bukan cookie session lagi).

KENAPA GANTI: dulu login pakai session cookie (server-rendered HTML dari
Jinja2 di domain yang sama). Sekarang frontend (GitHub Pages) dan backend
(Google AI Studio) ada di DOMAIN BEDA -- cookie session lintas-domain
bermasalah (SameSite, third-party cookie diblokir browser modern). Solusinya
token JWT yang dikirim manual lewat header, bukan otomatis lewat cookie.

ALUR LOGIN GOOGLE (beda dari sebelumnya!):
- Backend TIDAK lagi redirect ke accounts.google.com (authlib OAuth flow lama).
- Frontend pakai Google Identity Services (GIS, script JS dari Google) buat
  nampilin tombol "Sign in with Google" dan dapetin `credential` (ID token)
  LANGSUNG dari Google, di browser.
- Frontend kirim `credential` itu ke POST /api/auth/google.
- Backend verifikasi keasliannya (tanda tangan Google + `aud` cocok dengan
  GOOGLE_CLIENT_ID kita), lalu terbitkan JWT sendiri buat dipakai selanjutnya.
- Konsekuensinya: GOOGLE_CLIENT_SECRET & GOOGLE_REDIRECT_URI SUDAH TIDAK
  DIPAKAI LAGI -- cukup GOOGLE_CLIENT_ID (dipakai di backend UNTUK VERIFIKASI,
  dan di frontend buat nampilin tombol login).

ALUR LOGIN ADMIN: tetap username/password (POST /api/auth/admin/login),
bedanya sekarang balikin JWT (role="admin") bukan set cookie session.
"""
import os
import time
from typing import Optional

import jwt
from fastapi import Header, HTTPException, Query

SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "dev-secret-ganti-ini")
JWT_ALGORITHM = "HS256"
JWT_EXP_SECONDS = 60 * 60 * 24 * 7  # token berlaku 7 hari

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")


def verify_google_id_token(credential: str) -> dict:
    """Verifikasi ID token yang dikirim frontend (hasil Google Identity Services).
    Return {"email":..., "name":...}. Raise HTTPException kalau tidak valid."""
    try:
        # Import di dalam fungsi biar startup server gak gagal kalau paketnya
        # belum ke-install pas lagi develop fitur lain (fail lambat, bukan cepat).
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests

        info = google_id_token.verify_oauth2_token(
            credential, google_requests.Request(), GOOGLE_CLIENT_ID,
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token Google tidak valid: {e}")

    return {"email": info["email"], "name": info.get("name", info["email"])}


def create_jwt(payload: dict) -> str:
    to_encode = {**payload, "exp": int(time.time()) + JWT_EXP_SECONDS}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Sesi habis, silakan login ulang.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token tidak valid.")


def _extract_token(authorization: Optional[str], token_query: Optional[str]) -> str:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1]
    if token_query:
        return token_query
    raise HTTPException(status_code=401, detail="Belum login.")


def get_current_user(
    authorization: Optional[str] = Header(default=None),
    token: Optional[str] = Query(default=None),
) -> dict:
    """Dependency FastAPI: WAJIB login (user ATAU admin).
    Terima token lewat header 'Authorization: Bearer <jwt>' (cara normal),
    ATAU lewat query string '?token=<jwt>' (dipakai KHUSUS buat link unduh
    PDF, karena kalau orang klik link langsung di browser gak bisa nyisipin
    header custom)."""
    jwt_token = _extract_token(authorization, token)
    return decode_jwt(jwt_token)


def require_admin(authorization: Optional[str] = Header(default=None),
                   token: Optional[str] = Query(default=None)) -> dict:
    """Dependency FastAPI: WAJIB login sebagai admin."""
    payload = get_current_user(authorization, token)
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Khusus admin.")
    return payload
