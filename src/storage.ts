import fs from "fs";
import path from "path";

export interface Patient {
  id: number;
  nik: string;
  nama: string;
  tanggal_lahir: string;
  jenis_kelamin: string;
  alamat?: string;
  no_telepon?: string;
  created_at: string;
  created_by_email?: string;
}

export interface Prediction {
  id: number;
  filename: string;
  upload_time: string;
  precheck_status: "valid" | "invalid";
  precheck_confidence: number;
  prediction_label?: string | null;
  prediction_confidence?: number | null;
  all_probabilities?: Record<string, number> | null;
  gradcam_path?: string | null;
  gemini_explanation?: string | null;
  jenis_scan?: string;
  umur_saat_scan?: string;
  gejala?: string;
}

export interface MedicalRecord {
  id: number;
  patient_id: number;
  prediction_id: number;
  catatan_dokter?: string | null;
  created_at: string;
}

export interface MedicalRecordDetail extends MedicalRecord {
  filename?: string;
  prediction_label?: string | null;
  prediction_confidence?: number | null;
  gemini_explanation?: string | null;
  gradcam_path?: string | null;
  precheck_status?: string;
  jenis_scan?: string;
  umur_saat_scan?: string;
  gejala?: string;
  nik?: string;
  nama_pasien?: string;
}

class StorageService {
  private patients: Patient[] = [];
  private predictions: Prediction[] = [];
  private records: MedicalRecord[] = [];
  private nextPatientId = 1;
  private nextPredictionId = 1;
  private nextRecordId = 1;
  private dataDir: string;
  private dataFile: string;

  constructor() {
    this.dataDir = path.resolve(process.cwd(), "data");
    this.dataFile = path.join(this.dataDir, "store.json");
    this.init();
  }

  private init() {
    try {
      if (!fs.existsSync(this.dataDir)) {
        fs.mkdirSync(this.dataDir, { recursive: true });
      }

      if (fs.existsSync(this.dataFile)) {
        const raw = fs.readFileSync(this.dataFile, "utf-8");
        const parsed = JSON.parse(raw);
        this.patients = parsed.patients || [];
        this.predictions = parsed.predictions || [];
        this.records = parsed.records || [];
        this.nextPatientId = parsed.nextPatientId || this.patients.length + 1;
        this.nextPredictionId = parsed.nextPredictionId || this.predictions.length + 1;
        this.nextRecordId = parsed.nextRecordId || this.records.length + 1;
        return;
      }
    } catch (e) {
      console.warn("[Storage] Error loading persisted file, re-initializing:", e);
    }

    // Seed default demo data if empty
    this.seedDefaults();
    this.persist();
  }

  private seedDefaults() {
    const p1: Patient = {
      id: 1,
      nik: "3271012304850001",
      nama: "Budi Santoso",
      tanggal_lahir: "1985-04-23",
      jenis_kelamin: "Laki-Laki",
      alamat: "Jl. Merdeka No. 45, Jakarta",
      no_telepon: "081234567890",
      created_at: new Date(Date.now() - 86400000 * 3).toISOString().slice(0, 19).replace("T", " "),
      created_by_email: "dokter.sarah@neurocheck.id",
    };

    const p2: Patient = {
      id: 2,
      nik: "3271025508920002",
      nama: "Siti Rahmawati",
      tanggal_lahir: "1992-08-15",
      jenis_kelamin: "Perempuan",
      alamat: "Jl. Sudirman No. 12, Bandung",
      no_telepon: "085678901234",
      created_at: new Date(Date.now() - 86400000 * 1).toISOString().slice(0, 19).replace("T", " "),
      created_by_email: "dokter.sarah@neurocheck.id",
    };

    const pred1: Prediction = {
      id: 1,
      filename: "brain_mri_budi_axial_t2.jpg",
      upload_time: p1.created_at,
      precheck_status: "valid",
      precheck_confidence: 0.98,
      prediction_label: "Normal_Healthy",
      prediction_confidence: 0.942,
      all_probabilities: {
        Normal_Healthy: 0.942,
        Stroke_Iskemik: 0.021,
        Alzheimer_Mild: 0.015,
        Tumor_Glioma: 0.008,
        Tumor_Meningioma: 0.006,
        Tumor_Pituitary: 0.003,
        Intracranial_Hemorrhage: 0.002,
        Multiple_Sclerosis: 0.001,
        Alzheimer_Moderate: 0.001,
        Alzheimer_Very_Mild: 0.001,
      },
      gradcam_path: null,
      gemini_explanation:
        "Apa itu kondisi ini? Hasil skrining AI mengindikasikan 'Normal / Tidak Ditemukan Kelainan Signifikan' dengan tingkat keyakinan 94.2%. Struktur parenkim otak dan ventrikel tampak simetris dalam batas normal.\n\n" +
        "Langkah selanjutnya: Tidak diperlukan tindak lanjut darurat jika tidak ada gejala neurologis klinis yang dirasakan.\n\n" +
        "Arah solusi/penanganan: Pertahankan gaya hidup sehat, tidur cukup, dan kontrol tekanan darah berkala.\n\n" +
        "Spesialis yang perlu dikonsultasikan: Dokter Umum / Spesialis Neurologi (Sp.N) bila ada keluhan spesifik.\n\n" +
        "Catatan: ini hanya alat bantu skrining awal, BUKAN diagnosis final -- hasil wajib dikonfirmasi oleh dokter/radiolog yang berwenang.",
      jenis_scan: "MRI",
      umur_saat_scan: "41",
      gejala: "Pemeriksaan kesehatan rutin",
    };

    const rec1: MedicalRecord = {
      id: 1,
      patient_id: 1,
      prediction_id: 1,
      catatan_dokter: "Hasil pemeriksaan rutin tidak menunjukkan kelainan akut.",
      created_at: p1.created_at,
    };

    this.patients = [p1, p2];
    this.predictions = [pred1];
    this.records = [rec1];
    this.nextPatientId = 3;
    this.nextPredictionId = 2;
    this.nextRecordId = 2;
  }

  private persist() {
    try {
      fs.writeFileSync(
        this.dataFile,
        JSON.stringify(
          {
            patients: this.patients,
            predictions: this.predictions,
            records: this.records,
            nextPatientId: this.nextPatientId,
            nextPredictionId: this.nextPredictionId,
            nextRecordId: this.nextRecordId,
          },
          null,
          2
        ),
        "utf-8"
      );
    } catch (e) {
      console.warn("[Storage] Failed to persist data to file:", e);
    }
  }

  getAllPatients(): Patient[] {
    return [...this.patients].reverse();
  }

  getPatientByNik(nik: string): Patient | undefined {
    return this.patients.find((p) => p.nik === nik);
  }

  getPatientById(id: number): Patient | undefined {
    return this.patients.find((p) => p.id === id);
  }

  createPatient(data: {
    nik: string;
    nama: string;
    tanggal_lahir: string;
    jenis_kelamin: string;
    alamat?: string;
    no_telepon?: string;
    created_by_email?: string;
  }): number | null {
    if (this.patients.some((p) => p.nik === data.nik)) {
      return null; // duplicate NIK
    }

    const newId = this.nextPatientId++;
    const now = new Date().toISOString().slice(0, 19).replace("T", " ");
    const patient: Patient = {
      id: newId,
      nik: data.nik,
      nama: data.nama,
      tanggal_lahir: data.tanggal_lahir,
      jenis_kelamin: data.jenis_kelamin,
      alamat: data.alamat || "",
      no_telepon: data.no_telepon || "",
      created_at: now,
      created_by_email: data.created_by_email || "",
    };

    this.patients.push(patient);
    this.persist();
    return newId;
  }

  savePrediction(data: Omit<Prediction, "id" | "upload_time">): number {
    const id = this.nextPredictionId++;
    const now = new Date().toISOString().slice(0, 19).replace("T", " ");
    const prediction: Prediction = {
      id,
      upload_time: now,
      ...data,
    };
    this.predictions.push(prediction);
    this.persist();
    return id;
  }

  getPredictionById(id: number): Prediction | undefined {
    return this.predictions.find((p) => p.id === id);
  }

  addMedicalRecord(patientId: number, predictionId: number, catatan?: string): number {
    const id = this.nextRecordId++;
    const now = new Date().toISOString().slice(0, 19).replace("T", " ");
    const record: MedicalRecord = {
      id,
      patient_id: patientId,
      prediction_id: predictionId,
      catatan_dokter: catatan || null,
      created_at: now,
    };
    this.records.push(record);
    this.persist();
    return id;
  }

  getRecordsByPatient(patientId: number): MedicalRecordDetail[] {
    return this.records
      .filter((r) => r.patient_id === patientId)
      .map((r) => {
        const pred = this.predictions.find((p) => p.id === r.prediction_id);
        return {
          ...r,
          filename: pred?.filename,
          prediction_label: pred?.prediction_label,
          prediction_confidence: pred?.prediction_confidence,
          gemini_explanation: pred?.gemini_explanation,
          gradcam_path: pred?.gradcam_path,
          precheck_status: pred?.precheck_status,
          jenis_scan: pred?.jenis_scan,
          umur_saat_scan: pred?.umur_saat_scan,
          gejala: pred?.gejala,
        };
      })
      .reverse();
  }

  getAllRecords(): MedicalRecordDetail[] {
    return this.records
      .map((r) => {
        const patient = this.patients.find((p) => p.id === r.patient_id);
        const pred = this.predictions.find((p) => p.id === r.prediction_id);
        return {
          ...r,
          nik: patient?.nik,
          nama_pasien: patient?.nama,
          filename: pred?.filename,
          prediction_label: pred?.prediction_label,
          prediction_confidence: pred?.prediction_confidence,
          gemini_explanation: pred?.gemini_explanation,
          gradcam_path: pred?.gradcam_path,
          precheck_status: pred?.precheck_status,
        };
      })
      .reverse();
  }
}

export const storage = new StorageService();
