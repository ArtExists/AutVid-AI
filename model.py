# model.py -- Realtime Video + Audio + Subtitles + Emotion Fusion
import os
import time
import threading
import wave
from pathlib import Path
from typing import List, Tuple, Dict

import av
import cv2
import numpy as np
import streamlit as st
import torch

from transformers import BertTokenizer, BertForSequenceClassification

# Custom modules (ensure they exist)
from face_model import FacialEmotionDetector
from voice_det import Voice_Analysis

from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase, AudioProcessorBase

# ------------------------- Config -------------------------
FRAME_DETECT_EVERY_N = 4   # run YOLO every Nth frame (adjust for CPU)
AUDIO_SAMPLE_RATE = 48000
TEMP_AUDIO_PATH = "temp_recordings/live.wav"
BEST_PT = Path(__file__).parent / "best.pt"

st.set_page_config(page_title="AutVid AI — Realtime", layout="wide")
st.title("🧠 AutVid AI — Real-time Video + Audio Emotion")

# ------------------------- Cached model loaders -------------------------
@st.cache_resource
def load_face_model_main():
    if not BEST_PT.exists():
        st.warning(f"YOLO weights not found at {BEST_PT.resolve()}. Video detection will show placeholder.")
        return None
    try:
        det = FacialEmotionDetector(model_path=str(BEST_PT))
        st.info("FacialEmotionDetector loaded.")
        return det
    except Exception as e:
        st.error(f"Failed to load FacialEmotionDetector: {e}")
        return None

@st.cache_resource
def load_voice_model():
    try:
        vm = Voice_Analysis()
        st.info("Voice_Analysis loaded.")
        return vm
    except Exception as e:
        st.error(f"Failed to load Voice_Analysis: {e}")
        return None

@st.cache_resource
def load_text_model():
    try:
        model_name = "bhadresh-savani/bert-base-go-emotion"
        tok = BertTokenizer.from_pretrained(model_name)
        mdl = BertForSequenceClassification.from_pretrained(model_name)
        mdl.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mdl.to(device)
        id2label = mdl.config.id2label if hasattr(mdl.config, "id2label") else {i: str(i) for i in range(mdl.config.num_labels)}
        label_list = [id2label[i] for i in range(len(id2label))]
        st.info("Text model loaded.")
        return tok, mdl, device, label_list
    except Exception as e:
        st.error(f"Failed to load text model: {e}")
        return None, None, None, []

face_model_main = load_face_model_main()
voice_model = load_voice_model()
tokenizer, text_model, device, label_list = load_text_model()

# ------------------------- Text analysis -------------------------
def analyze_text_multilabel(text: str, threshold: float = 0.3) -> Tuple[List[str], Dict[str, float]]:
    if not text.strip() or text_model is None:
        return [], {}
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        logits = text_model(**enc).logits
        probs = torch.sigmoid(logits)[0].cpu().numpy()
    scores = {label_list[i]: float(probs[i]) for i in range(len(label_list))}
    chosen = [lbl for lbl, p in scores.items() if p >= threshold]
    if not chosen:
        chosen = [max(scores, key=scores.get)]
    return chosen, scores

# ------------------------- WebRTC processors -------------------------
class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.frames = []
        self.lock = threading.Lock()
        self.sample_rate = AUDIO_SAMPLE_RATE

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        arr = frame.to_ndarray()
        mono = np.mean(arr, axis=0).astype(np.int16) if arr.ndim == 2 else arr.astype(np.int16)
        with self.lock:
            self.frames.append(mono)
        return frame

    def save_wav(self, filename: str = TEMP_AUDIO_PATH) -> str:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with self.lock:
            if not self.frames:
                raise ValueError("No audio captured")
            audio = np.concatenate(self.frames, axis=0).astype(np.int16)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())
        return filename

    def clear(self):
        with self.lock:
            self.frames = []

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        try:
            self.detector = FacialEmotionDetector(model_path=str(BEST_PT)) if BEST_PT.exists() else None
        except:
            self.detector = None
        self.lock = threading.Lock()
        self.counter = 0
        self.last_annotated = None
        self.last_emotion = None

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        annotated = img.copy()
        emo = None
        self.counter += 1
        try:
            if self.counter % FRAME_DETECT_EVERY_N == 0 and self.detector:
                ann, emo = self.detector.detect_emotion(img)
                if ann is not None:
                    annotated = ann
        except Exception as e:
            print("Frame detection error:", e)

        # Overlay transcript
        transcript = st.session_state.get("transcript_overlay", "")
        y0 = 30
        for i, line in enumerate(transcript.split("\n")[-3:]):
            y = y0 + i*25
            cv2.putText(annotated, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Overlay last emotion
        if emo:
            cv2.putText(annotated, f"Emotion: {emo}", (10, y0 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        with self.lock:
            self.last_annotated = annotated.copy()
            self.last_emotion = emo

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def get_last(self):
        with self.lock:
            return self.last_annotated, self.last_emotion

# ------------------------- Session state -------------------------
for k, v in {
    "video_emotion": None,
    "voice_emotion": None,
    "transcript": "",
    "transcript_overlay": "",
    "text_emotions": []
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------------- UI / Streamer -------------------------
st.sidebar.markdown("## Controls")
FRAME_DETECT_EVERY_N = st.sidebar.slider("Run YOLO every N frames", 1, 12, FRAME_DETECT_EVERY_N, 1)
auto_analyze = st.sidebar.checkbox("Auto analyze audio every interval", value=False)
auto_interval = st.sidebar.slider("Auto analyze interval (s)", 5, 30, 12, 1)

col_main, col_side = st.columns([2, 1])

with col_main:
    st.subheader("Live camera (annotated)")
    ctx = webrtc_streamer(
        key="live-av",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=VideoProcessor,
        audio_processor_factory=AudioRecorder,
        media_stream_constraints={"video": True, "audio": True},
        async_processing=True,
    )

    st.markdown("---")
    st.write("Live preview from worker:")
    if ctx and ctx.video_transformer:
        annotated_frame, last_emo = ctx.video_transformer.get_last()
        if annotated_frame is not None:
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption=f"Emotion: {last_emo}")

with col_side:
    st.subheader("Live outputs")
    st.metric("Video emotion", st.session_state.get("video_emotion") or "N/A")
    st.metric("Voice emotion", st.session_state.get("voice_emotion") or "N/A")
    st.text_area("Transcript", value=st.session_state.get("transcript", ""), height=160)

    if st.button("Clear audio buffer") and ctx and ctx.audio_receiver:
        try:
            ctx.audio_receiver._processor.clear()
            st.success("Cleared audio buffer.")
        except Exception as e:
            st.error(f"Clear failed: {e}")

    if st.button("Save & Analyze now") and ctx and ctx.audio_receiver:
        proc = ctx.audio_receiver._processor
        try:
            wav = proc.save_wav(TEMP_AUDIO_PATH)
            proc.clear()
            st.audio(wav)
            if voice_model:
                res = voice_model.detect(wav)
                if res:
                    st.session_state.voice_emotion = max(res, key=lambda r: r["score"])["label"]
                st.session_state.transcript = voice_model.subtitles(wav)
                st.session_state.transcript_overlay = st.session_state.transcript
            st.success("Saved and analyzed audio.")
        except Exception as e:
            st.error(f"Save/analyze failed: {e}")

# Update video emotion from worker
if ctx and ctx.video_transformer:
    _, last_vid_emo = ctx.video_transformer.get_last()
    if last_vid_emo:
        st.session_state.video_emotion = last_vid_emo

# Auto audio analyze loop
def auto_audio_loop():
    while True:
        if auto_analyze and ctx and ctx.audio_receiver:
            try:
                proc = ctx.audio_receiver._processor
                wav = proc.save_wav(TEMP_AUDIO_PATH.replace(".wav","_auto.wav"))
                proc.clear()
                if voice_model:
                    res = voice_model.detect(wav)
                    if res:
                        st.session_state.voice_emotion = max(res, key=lambda r: r["score"])["label"]
                    txt = voice_model.subtitles(wav)
                    st.session_state.transcript = txt
                    st.session_state.transcript_overlay = txt
            except Exception:
                pass
        time.sleep(auto_interval)

threading.Thread(target=auto_audio_loop, daemon=True).start()

# ---- Text analysis UI ----
st.markdown("---")
st.subheader("Text Emotion (BERT multi-label)")
text_in = st.text_area("Enter text to analyze", value=st.session_state.get("transcript", ""), height=140)
thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.3, 0.05)
if st.button("Analyze text"):
    chosen, scores = analyze_text_multilabel(text_in, threshold=thresh)
    st.session_state.text_emotions = chosen
    if scores:
        st.json({k: round(v,4) for k,v in sorted(scores.items(), key=lambda x: x[1], reverse=True)})
    if chosen:
        st.success(f"Predicted (≥{thresh:.2f}): {', '.join(chosen)}")

# ---- Multimodal Fusion ----
st.markdown("---")
st.subheader("Multimodal Fusion")
st.write("Video 0.5, Voice 0.3, Text 0.2")

def fuse(video_emotion, voice_emotion, text_emotions):
    w = {"video":0.5, "voice":0.3, "text":0.2}
    s = {}
    if video_emotion:
        s[video_emotion] = s.get(video_emotion,0)+w["video"]
    if voice_emotion:
        s[voice_emotion] = s.get(voice_emotion,0)+w["voice"]
    if text_emotions:
        share = w["text"]/max(1,len(text_emotions))
        for t in text_emotions:
            s[t] = s.get(t,0)+share
    return s

if st.button("Fuse now"):
    breakdown = fuse(st.session_state.get("video_emotion"), st.session_state.get("voice_emotion"), st.session_state.get("text_emotions", []))
    if breakdown:
        dom = max(breakdown, key=breakdown.get)
        st.success(f"Dominant emotion: {dom}")
        st.json({k: round(v,3) for k,v in breakdown.items()})
    else:
        st.warning("No modalities available yet.")
