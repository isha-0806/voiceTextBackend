
import os
import tempfile
from fastapi import Request
from pathlib import Path

import whisper
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

app = FastAPI(title="VoiceText Whisper API")

# Multilingual model: supports English, Hindi, and other languages.
# The weights download the first time this model is loaded.
model = whisper.load_model("base")

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
ALLOWED_LANGUAGES = {"en": "english", "hi": "hindi"}


@app.get("/health")
def health():
    return {"status": "ok", "model": "base"}


@app.post("/transcribe")
async def transcribe(
    request: Request,
    language: str = "en",
):
    if language not in ALLOWED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail="Language must be 'en' or 'hi'.",
        )

    content_type = request.headers.get("content-type", "")
    suffix = ".m4a" if "m4a" in content_type else ".wav"
    temp_path = None

    try:
        audio_bytes = await request.body()

        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file.")

        if len(audio_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Audio exceeds 25 MB.")

        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as temp:
            temp.write(audio_bytes)
            temp_path = temp.name

        result = model.transcribe(
            temp_path,
            language=ALLOWED_LANGUAGES[language],
            task="transcribe",
            fp16=False,
        )

        return {
            "text": result["text"].strip(),
            "language": result.get("language", language),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
