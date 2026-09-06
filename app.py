"""FastAPI web application for the dual-stage fraud-call detector."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from anti_spoofing import analyze_audio_spoofing, get_anti_spoofing_info
from fusion import fuse_scores
from intent_recognition import IntentRecognizer, analyze_audio_intent

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "temp_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}

app = FastAPI(
    title="Dual-Stage Fraud Call Detection API",
    version="2.0.0",
)

# Restrict this in production through ALLOWED_ORIGINS.
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Load expensive NLP models once per application process.
intent_recognizer = None


def get_intent_recognizer() -> IntentRecognizer:
    global intent_recognizer
    if intent_recognizer is None:
        intent_recognizer = IntentRecognizer()
    return intent_recognizer


@app.get("/", response_class=HTMLResponse)
def read_index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "service": "fraud-call-detector"}


@app.post("/api/analyze")
def analyze_audio(audio: UploadFile = File(...)):
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    extension = Path(audio.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{extension}"

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        size_mb = temp_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file is too large. Maximum size is {MAX_UPLOAD_MB} MB.",
            )

        spoofing_score = analyze_audio_spoofing(str(temp_path))
        intent_score, transcript, label_scores = analyze_audio_intent(
            str(temp_path),
            recognizer=get_intent_recognizer(),
        )
        final_score, verdict = fuse_scores(spoofing_score, intent_score)

        return {
            "status": "success",
            "results": {
                "spoofing_score": round(spoofing_score, 4),
                "intent_score": round(intent_score, 4),
                "final_score": round(final_score, 4),
                "verdict": verdict,
                "transcript": transcript,
                "intent_labels": {
                    key: round(value, 4) for key, value in label_scores.items()
                },
                "anti_spoofing": get_anti_spoofing_info(),
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        try:
            audio.file.close()
        except Exception:
            pass
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
