import whisper
from transformers import pipeline
import torchaudio
import sounddevice as sd
import torch
import tempfile
import os

class Voice_Analysis:
    def __init__(self, emotion_model="prithivMLmods/Speech-Emotion-Classification", whisper_size="base"):
        # HF pipeline for speech emotion
        self.classifier = pipeline(
            "audio-classification",
            model=emotion_model,
            feature_extractor=emotion_model
        )
        # Whisper for ASR
        self.modelwa = whisper.load_model(whisper_size)

    def record_audio(self, filename="temp.wav", duration=3, sampling=16000):
        """Record audio from microphone to a WAV file."""
        audio_data = sd.rec(int(duration * sampling), samplerate=sampling, channels=1, dtype="float32")
        sd.wait()
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        torchaudio.save(filename, torch.tensor(audio_data.T), samplerate=sampling)
        return filename

    def detect(self, path):
        """Run emotion classification on an audio file. Returns list of dicts with label/score."""
        return self.classifier(path)

    def subtitles(self, path):
        """Transcribe audio to text using Whisper."""
        result = self.modelwa.transcribe(path)
        return result.get("text", "").strip()
