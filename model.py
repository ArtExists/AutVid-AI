import os
from pathlib import Path
import streamlit as st
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple

# Video (YOLO) detector
from face_model import FacialEmotionDetector
# Voice analyzer (emotion + ASR)
from voice_det import Voice_Analysis

# Text model (BERT multi-label by default if your fine-tuned model exists)
from transformers import BertTokenizer, BertForSequenceClassification

st.set_page_config(page_title="AutVid AI - Multimodal Emotion Analysis", page_icon="🧠", layout="wide")
st.markdown("<h1 style='text-align:center;'>🧠 AutVid AI - Multimodal Emotion Analysis</h1>", unsafe_allow_html=True)

# ------------------ CACHED LOADERS ------------------ #
@st.cache_resource
def load_face_model():
    # Adjust path to your YOLO weights
    model_path = "best.pt"
    return FacialEmotionDetector(model_path=model_path)

@st.cache_resource
def load_voice_model():
    return Voice_Analysis()

@st.cache_resource
def load_text_model():
    """
    Try to load local fine-tuned model at ./bert_emotion (from train_bert.py).
    Fallback to a public BERT fine-tuned on GoEmotions.
    """
    if Path("./bert_emotion").exists():
        model_name = "./bert_emotion"
    else:
        # BERT model fine-tuned on GoEmotions (multi-label)
        model_name = "bhadresh-savani/bert-base-go-emotion"

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # label mapping (GoEmotions - 28 incl neutral in some configs; tokenizer/model has id2label)
    id2label = model.config.id2label if hasattr(model.config, "id2label") else {i: str(i) for i in range(model.config.num_labels)}
    label_list = [id2label[i] for i in range(len(id2label))]

    return tokenizer, model, device, label_list

face_model = load_face_model()
voice_model = load_voice_model()
tokenizer, text_model, device, label_list = load_text_model()

# ------------------ HELPERS ------------------ #
def analyze_text_multilabel(text: str, threshold: float = 0.3) -> Tuple[List[str], Dict[str, float]]:
    """Return list of emotions above threshold + full prob dict."""
    if not text.strip():
        return [], {}

    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        logits = text_model(**enc).logits
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()  # multi-label

    scores = {label_list[i]: float(probs[i]) for i in range(len(label_list))}
    chosen = [lbl for lbl, p in scores.items() if p >= threshold]
    # if nothing crossed threshold, take top-1
    if not chosen:
        top = max(scores, key=scores.get)
        chosen = [top]
    return chosen, scores

def fuse_emotions(video_emotion: str, voice_emotion: str, text_emotions: List[str]) -> Tuple[str, Dict[str, float]]:
    """Weighted fusion: video 0.5, voice 0.3, text 0.2 (distributed among predicted text labels)."""
    weights = {"video": 0.5, "voice": 0.3, "text": 0.2}
    scores: Dict[str, float] = {}

    if video_emotion:
        scores[video_emotion] = scores.get(video_emotion, 0.0) + weights["video"]

    if voice_emotion:
        scores[voice_emotion] = scores.get(voice_emotion, 0.0) + weights["voice"]

    if text_emotions:
        share = weights["text"] / len(text_emotions)
        for t in text_emotions:
            scores[t] = scores.get(t, 0.0) + share

    dominant = max(scores, key=scores.get) if scores else "unknown"
    return dominant, scores

def cv2_to_st(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

# ------------------ SESSION STATE ------------------ #
if "video_emotion" in st.session_state is False:
    st.session_state["video_emotion"] = None
if "voice_emotion" in st.session_state is False:
    st.session_state["voice_emotion"] = None
if "text_emotions" in st.session_state is False:
    st.session_state["text_emotions"] = []
if "transcript" in st.session_state is False:
    st.session_state["transcript"] = ""

# ------------------ UI TABS ------------------ #
tab1, tab2, tab3, tab4 = st.tabs(["🎥 Video", "🎵 Voice", "📝 Text", "⚡ Combined"])

# ---- VIDEO TAB ---- #
with tab1:
    st.subheader("Video Emotion Detection (YOLO)")
    st.caption("Upload an image or use your camera to capture a frame.")

    colA, colB = st.columns(2)
    with colA:
        img_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
        if img_file is not None:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            annotated, emo = face_model.detect_emotion(frame)
            st.image(cv2_to_st(annotated), caption=f"Detected: {emo}")
            st.session_state["video_emotion"] = emo

    with colB:
        snap = st.camera_input("Or capture from your webcam")
        if snap is not None:
            snap_bytes = np.asarray(bytearray(snap.read()), dtype=np.uint8)
            frame = cv2.imdecode(snap_bytes, cv2.IMREAD_COLOR)
            annotated, emo = face_model.detect_emotion(frame)
            st.image(cv2_to_st(annotated), caption=f"Detected: {emo}")
            st.session_state["video_emotion"] = emo

    if st.session_state["video_emotion"]:
        st.success(f"Video Emotion: {st.session_state['video_emotion']}")

# ---- VOICE TAB ---- #
with tab2:
    st.subheader("Voice Emotion & Transcription")
    st.caption("Record 3 seconds or upload a .wav file.")

    colL, colR = st.columns(2)

    with colL:
        if st.button("🎙️ Record 3s"):
            wav_path = voice_model.record_audio("temp_recordings/record.wav", duration=3)
            st.audio(wav_path)
            results = voice_model.detect(wav_path)
            if results:
                voice_label = max(results, key=lambda r: r["score"])["label"]
                st.session_state["voice_emotion"] = voice_label
                # Also transcribe
                st.session_state["transcript"] = voice_model.subtitles(wav_path)
            st.success(f"Voice Emotion: {st.session_state.get('voice_emotion', 'N/A')}")
            if st.session_state["transcript"]:
                st.info(f"Transcript: {st.session_state['transcript']}")

    with colR:
        up = st.file_uploader("Upload WAV", type=["wav"])
        if up is not None:
            tmp_path = "temp_uploads/upload.wav"
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(up.read())
            st.audio(tmp_path)
            results = voice_model.detect(tmp_path)
            if results:
                voice_label = max(results, key=lambda r: r["score"])["label"]
                st.session_state["voice_emotion"] = voice_label
                st.session_state["transcript"] = voice_model.subtitles(tmp_path)
            st.success(f"Voice Emotion: {st.session_state.get('voice_emotion', 'N/A')}")
            if st.session_state["transcript"]:
                st.info(f"Transcript: {st.session_state['transcript']}")

# ---- TEXT TAB ---- #
with tab3:
    st.subheader("Text Emotion (BERT multi-label)")
    default_text = st.session_state.get("transcript", "")
    text_in = st.text_area("Enter text to analyze", value=default_text, height=120, placeholder="Type or paste transcript...")
    thresh = st.slider("Confidence threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.05)
    if st.button("🚀 Analyze Text"):
        chosen, scores = analyze_text_multilabel(text_in, threshold=thresh)
        st.session_state["text_emotions"] = chosen
        if scores:
            st.json({k: round(v, 4) for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)})
        if chosen:
            st.success(f"Predicted (≥{thresh:.2f}): {', '.join(chosen)}")

# ---- COMBINED TAB ---- #
with tab4:
    st.subheader("Multimodal Fusion")
    st.write("Weights: Video **0.5**, Voice **0.3**, Text **0.2**")
    ve = st.session_state.get("video_emotion")
    vo = st.session_state.get("voice_emotion")
    te = st.session_state.get("text_emotions", [])
    if st.button("⚡ Fuse Now"):
        dominant, scores = fuse_emotions(ve, vo, te)
        if scores:
            st.success(f"🎭 Dominant Emotion: **{dominant}**")
            st.write("Breakdown (higher = stronger influence):")
            st.json({k: round(v, 3) for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)})
        else:
            st.warning("Please run at least one modality first (Video / Voice / Text).")
