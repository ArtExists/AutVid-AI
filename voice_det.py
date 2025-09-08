import whisper
from transformers import pipeline

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

    def detect(self, path):
        """Run emotion classification on an audio file. Returns list of dicts with label/score."""
        return self.classifier(path)

    def subtitles(self, path):
        """Transcribe audio to text using Whisper."""
        result = self.modelwa.transcribe(path)
        return result.get("text", "").strip()
