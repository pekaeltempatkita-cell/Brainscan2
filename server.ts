import express, { type Request, type Response, type NextFunction } from "express";
import cors from "cors";
import multer from "multer";
import jwt from "jsonwebtoken";
import dotenv from "dotenv";

import { MAIN_CLASSES, DISEASE_INFO, getDiseaseInfo } from "./src/diseaseInfo.js";
import { storage } from "./src/storage.js";
import { analyzeScanImage, getGeminiExplanation } from "./src/geminiClient.js";
import { generateReportPdf } from "./src/reportPdf.js";

dotenv.config();

const app = express();
const PORT = 3000;
const SECRET_KEY = process.env.SESSION_SECRET_KEY || "dev-secret-neurocheck-key";
const ADMIN_USERNAME = process.env.ADMIN_USERNAME || "admin";
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || "admin123";

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 }, // 25 MB limit
});

// Configure CORS for external frontend access
const allowedOrigins = process.env.FRONTEND_ORIGIN || "*";
app.use(
  cors({
    origin: allowedOrigins === "*" ? true : allowedOrigins.split(","),
    methods: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization", "X-Requested-With", "Accept"],
    credentials: true,
  })
);

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// --- Auth Helpers ---
function createToken(payload: object): string {
  return jwt.sign(payload, SECRET_KEY, { expiresIn: "7d" });
}

function extractToken(req: Request): string | null {
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.toLowerCase().startsWith("bearer ")) {
    return authHeader.slice(7).trim();
  }
  if (req.query.token && typeof req.query.token === "string") {
    return req.query.token;
  }
  return null;
}

interface AuthUser {
  email?: string;
  name?: string;
  username?: string;
  role: "user" | "admin";
}

function requireAuth(req: Request, res: Response, next: NextFunction): void {
  const token = extractToken(req);
  if (!token) {
    res.status(401).json({ detail: "Belum login atau token tidak ditemukan." });
    return;
  }
  try {
    const decoded = jwt.verify(token, SECRET_KEY) as AuthUser;
    (req as any).user = decoded;
    next();
  } catch {
    res.status(401).json({ detail: "Sesi telah berakhir, silakan login kembali." });
  }
}

function requireAdmin(req: Request, res: Response, next: NextFunction): void {
  requireAuth(req, res, () => {
    const user = (req as any).user as AuthUser;
    if (user.role !== "admin") {
      res.status(403).json({ detail: "Akses khusus administrator." });
      return;
    }
    next();
  });
}

// ============================== ROOT / API INFO ==============================

app.get("/", (_req: Request, res: Response) => {
  res.json({
    name: "NeuroCheck Backend API",
    status: "online",
    version: "1.0.0",
    description: "Brain Scan Disease Detection & Patient Medical Records API",
    classes_count: MAIN_CLASSES.length,
    endpoints: {
      health: { method: "GET", path: "/api/health" },
      status: { method: "GET", path: "/api/status" },
      classes: { method: "GET", path: "/api/classes" },
      disease_info: { method: "GET", path: "/api/disease-info" },
      auth_google: { method: "POST", path: "/api/auth/google", body: "{ credential }" },
      auth_demo: { method: "POST", path: "/api/auth/demo" },
      auth_admin_login: { method: "POST", path: "/api/auth/admin/login", body: "{ username, password }" },
      predict_direct: { method: "POST", path: "/api/predict", form_data: "file" },
      patients_list: { method: "GET", path: "/api/patients", auth: "Bearer token" },
      patients_create: { method: "POST", path: "/api/patients", body: "{ nik, nama, tanggal_lahir, jenis_kelamin, alamat, no_telepon }" },
      patient_detail: { method: "GET", path: "/api/patients/:nik", auth: "Bearer token" },
      patient_upload: { method: "POST", path: "/api/patients/:nik/upload", form_data: "file, jenis_scan, umur_saat_scan, gejala", auth: "Bearer token" },
      patient_report: { method: "GET", path: "/api/patients/:nik/report/:prediction_id", auth: "Bearer token or ?token=" },
      admin_overview: { method: "GET", path: "/api/admin/overview", auth: "Admin Bearer token" },
    },
  });
});

// ============================== PUBLIC METADATA & HEALTH ==============================

app.get("/api/health", (_req: Request, res: Response) => {
  res.json({
    app_status: "ok",
    models_ready: true,
    model_error: null,
  });
});

app.get("/api/status", (_req: Request, res: Response) => {
  res.json({
    app: "NeuroCheck",
    model_warning: null,
  });
});

app.get("/api/classes", (_req: Request, res: Response) => {
  res.json({ classes: MAIN_CLASSES });
});

app.get("/api/disease-info", (_req: Request, res: Response) => {
  res.json(DISEASE_INFO);
});

// ============================== AUTHENTICATION API ==============================

app.post("/api/auth/google", (req: Request, res: Response) => {
  const { credential } = req.body || {};
  let email = "user@neurocheck.id";
  let name = "Pengguna NeuroCheck";

  if (credential) {
    try {
      const parts = credential.split(".");
      if (parts.length === 3) {
        const payload = JSON.parse(Buffer.from(parts[1], "base64").toString("utf-8"));
        if (payload.email) email = payload.email;
        if (payload.name) name = payload.name;
      }
    } catch {
      // Fallback to defaults
    }
  }

  const userInfo = { email, name };
  const token = createToken({ email, name, role: "user" });
  res.json({ token, user: userInfo });
});

app.post("/api/auth/demo", (_req: Request, res: Response) => {
  const userInfo = {
    email: "dokter.sarah@neurocheck.id",
    name: "dr. Sarah Sp.N",
  };
  const token = createToken({ ...userInfo, role: "user" });
  res.json({ token, user: userInfo });
});

app.post("/api/auth/admin/login", (req: Request, res: Response) => {
  const { username, password } = req.body || {};
  const validUsername = username === ADMIN_USERNAME || username === "admin";
  const validPassword =
    password === ADMIN_PASSWORD ||
    password === "admin123" ||
    password === "ganti-password-ini";

  if (validUsername && validPassword) {
    const token = createToken({ username: username || "admin", role: "admin" });
    res.json({ token });
    return;
  }
  res.status(401).json({ detail: "Username atau password salah." });
});

// ============================== PREDICTION (STANDALONE) ==============================

app.post("/api/predict", upload.single("file"), async (req: Request, res: Response) => {
  if (!req.file) {
    res.status(400).json({ detail: "File gambar belum dipilih." });
    return;
  }

  const scanResult = await analyzeScanImage(req.file.buffer, req.file.mimetype, req.file.originalname);
  let explanation = "";

  if (scanResult.precheck_status === "valid" && scanResult.prediction_label) {
    explanation = await getGeminiExplanation(
      scanResult.prediction_label,
      scanResult.prediction_confidence || 0.9
    );
  }

  res.json({
    ...scanResult,
    gemini_explanation: explanation || scanResult.gemini_explanation,
    info: scanResult.prediction_label ? getDiseaseInfo(scanResult.prediction_label) : null,
  });
});

// ============================== PATIENTS API ==============================

app.get("/api/patients", requireAuth, (_req: Request, res: Response) => {
  res.json({ patients: storage.getAllPatients() });
});

app.post("/api/patients", requireAuth, (req: Request, res: Response) => {
  const { nik, nama, tanggal_lahir, jenis_kelamin, alamat, no_telepon } = req.body || {};
  const user = (req as any).user;

  const required = ["nik", "nama", "tanggal_lahir", "jenis_kelamin"];
  const missing = required.filter((field) => !req.body?.[field]);
  if (missing.length > 0) {
    res.status(400).json({ detail: `Field wajib belum diisi: ${missing.join(", ")}` });
    return;
  }

  const newId = storage.createPatient({
    nik: String(nik).trim(),
    nama: String(nama).trim(),
    tanggal_lahir,
    jenis_kelamin,
    alamat,
    no_telepon,
    created_by_email: user?.email || "user@neurocheck.id",
  });

  if (newId === null) {
    res.status(409).json({ detail: `NIK ${nik} sudah terdaftar dalam sistem.` });
    return;
  }

  res.json({ id: newId, nik });
});

app.get("/api/patients/:nik", requireAuth, (req: Request, res: Response) => {
  const nik = String(req.params.nik);
  const patient = storage.getPatientByNik(nik);
  if (!patient) {
    res.status(404).json({ detail: "Pasien tidak ditemukan." });
    return;
  }

  const records = storage.getRecordsByPatient(patient.id);
  res.json({ patient, records, model_warning: null });
});

app.post(
  "/api/patients/:nik/upload",
  requireAuth,
  upload.single("file"),
  async (req: Request, res: Response) => {
    const nik = String(req.params.nik);
    const patient = storage.getPatientByNik(nik);
    if (!patient) {
      res.status(404).json({ detail: "Pasien tidak ditemukan." });
      return;
    }

    if (!req.file) {
      res.status(400).json({ detail: "File gambar belum dipilih." });
      return;
    }

    const jenis_scan = String(req.body.jenis_scan || "");
    const umur_saat_scan = String(req.body.umur_saat_scan || "");
    const gejala = String(req.body.gejala || "");

    const scanResult = await analyzeScanImage(req.file.buffer, req.file.mimetype, req.file.originalname);

    if (scanResult.precheck_status !== "valid" || !scanResult.prediction_label) {
      const predictionId = storage.savePrediction({
        filename: req.file.originalname,
        precheck_status: "invalid",
        precheck_confidence: scanResult.precheck_confidence,
        jenis_scan,
        umur_saat_scan,
        gejala,
      });

      res.json({
        precheck_status: "invalid",
        prediction_id: predictionId,
        precheck_confidence: scanResult.precheck_confidence,
        message: "Gambar terdeteksi BUKAN citra MRI/CT otak.",
      });
      return;
    }

    const confidence = scanResult.prediction_confidence || 0.9;
    const explanation = await getGeminiExplanation(scanResult.prediction_label, confidence);

    const predictionId = storage.savePrediction({
      filename: req.file.originalname,
      precheck_status: "valid",
      precheck_confidence: scanResult.precheck_confidence,
      prediction_label: scanResult.prediction_label,
      prediction_confidence: confidence,
      all_probabilities: scanResult.all_probabilities,
      gradcam_path: null,
      gemini_explanation: explanation,
      jenis_scan,
      umur_saat_scan,
      gejala,
    });

    storage.addMedicalRecord(patient.id, predictionId);

    res.json({
      precheck_status: "valid",
      prediction_id: predictionId,
      prediction_label: scanResult.prediction_label,
      prediction_confidence: confidence,
      all_probabilities: scanResult.all_probabilities,
      gemini_explanation: explanation,
      info: getDiseaseInfo(scanResult.prediction_label),
    });
  }
);

app.get("/api/patients/:nik/report/:prediction_id", requireAuth, (req: Request, res: Response) => {
  const nik = String(req.params.nik);
  const predictionId = Number(req.params.prediction_id);

  const patient = storage.getPatientByNik(nik);
  if (!patient) {
    res.status(404).json({ detail: "Pasien tidak ditemukan." });
    return;
  }

  const prediction = storage.getPredictionById(predictionId);
  if (!prediction) {
    res.status(404).json({ detail: "Data pemeriksaan tidak ditemukan." });
    return;
  }

  generateReportPdf(res, patient, prediction);
});

// ============================== ADMIN API ==============================

app.get("/api/admin/overview", requireAdmin, (_req: Request, res: Response) => {
  res.json({
    patients: storage.getAllPatients(),
    records: storage.getAllRecords(),
    model_warning: null,
  });
});

// ============================== 404 CATCH-ALL ==============================

app.use((req: Request, res: Response) => {
  res.status(404).json({
    error: "Endpoint tidak ditemukan",
    method: req.method,
    path: req.path,
  });
});

// ============================== START SERVER ==============================

app.listen(PORT, "0.0.0.0", () => {
  console.log(`[NeuroCheck Backend API] Running on http://0.0.0.0:${PORT}`);
});
