# speech_io.py
import sounddevice as sd
import soundfile as sf
import tempfile
import os
from pathlib import Path
import numpy as np
import subprocess
import sys

# Try to import whisper; if not available we'll fallback
try:
    import whisper
    _HAS_WHISPER = True
except Exception:
    _HAS_WHISPER = False

# SpeechRecognition fallback
try:
    import speech_recognition as sr
    _HAS_SR = True
except Exception:
    _HAS_SR = False

def record_audio(duration=3, fs=16000):
    """
    Record audio from default microphone for `duration` seconds and return the file path (wav).
    """
    fname = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    filename = fname.name
    fname.close()

    print(f"Recording for {duration}s to {filename} ...")
    data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    sf.write(filename, data, fs)
    return filename

def transcribe_audio(filepath, prefer_whisper=True):
    """
    Try Whisper local transcription first (if prefer_whisper and available),
    else use SpeechRecognition (Google Web Speech) as fallback.
    Returns transcription string or empty string on failure.
    """
    if prefer_whisper and _HAS_WHISPER:
        try:
            model = whisper.load_model("base")  # 'small' or 'base' recommended depending on resources
            # Whisper expects 16-bit wav; we saved as such
            res = model.transcribe(filepath)
            return res.get("text", "").strip()
        except Exception as e:
            print("Whisper transcription failed:", e)

    # fallback to SpeechRecognition
    if _HAS_SR:
        try:
            r = sr.Recognizer()
            with sr.AudioFile(filepath) as source:
                audio = r.record(source)
            text = r.recognize_google(audio)
            return text
        except Exception as e:
            print("SpeechRecognition fallback failed:", e)

    # last resort: return empty
    return ""
