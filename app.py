import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import psutil
import streamlit as st
import torch
from PIL import Image

from src.model import EmotionResNet
from src.utils import FaceDetector, get_emotion_label, preprocess_face

try:
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
    import av

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
MODEL_PATH = "models/emotion_model.pth"


if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.now()

st.set_page_config(
    page_title="Sentira - AI Emotion Intelligence",
    page_icon=":brain:",
    layout="wide",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: 0;
    }

    .main-title {
        font-size: 3.3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #ffffff, #b7c9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #a7adba;
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 2.5rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    .stButton>button {
        background-color: #2e66ff;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        background-color: #1a52eb;
        box-shadow: 0 4px 12px rgba(46, 102, 255, 0.3);
    }

    .badge {
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }

    .badge-active {
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    .badge-demo {
        background: rgba(245, 158, 11, 0.1);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        return None, False, f"Model file not found at {MODEL_PATH}."

    try:
        model = EmotionResNet(pretrained=False)
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model, True, "Model loaded successfully."
    except Exception as exc:
        return None, False, f"Error loading model: {exc}"


@st.cache_resource
def get_detector():
    return FaceDetector()


def predict_emotion(face_roi, color_space="BGR"):
    if not is_trained or emotion_model is None:
        return None, None

    input_tensor = preprocess_face(face_roi, color_space=color_space)
    with torch.no_grad():
        output = emotion_model(input_tensor)
        probs = torch.nn.functional.softmax(output / 1.2, dim=1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    return get_emotion_label(top_idx), probs


def annotate_frame(frame, detector, color_space="BGR"):
    faces = detector.detect_faces(frame, color_space=color_space)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (46, 102, 255), 2)
        face_roi = frame[y : y + h, x : x + w]

        try:
            emotion, probs = predict_emotion(face_roi, color_space=color_space)
        except ValueError:
            emotion, probs = None, None

        if emotion and probs is not None:
            confidence = probs[EMOTIONS.index(emotion)] * 100
            label = f"{emotion} {confidence:.1f}%"
            text_color = (255, 255, 255)
        else:
            label = "Face detected"
            text_color = (16, 185, 129)

        cv2.putText(frame, label, (x, max(y - 15, 20)), cv2.FONT_HERSHEY_DUPLEX, 0.7, text_color, 1)

    return frame, faces


emotion_model, is_trained, model_status = load_emotion_model()

st.markdown("<h1 class='main-title'>Sentira Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time facial emotion recognition</p>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/brain.png", width=60)
    st.title("Dashboard")
    badge_class = "badge-active" if is_trained else "badge-demo"
    badge_text = "Model: ResNet18 Active" if is_trained else "Model: Demo Mode"
    st.markdown(f"<div class='badge {badge_class}'>{badge_text}</div>", unsafe_allow_html=True)
    if not is_trained:
        st.caption(model_status)
    st.markdown("---")
    mode = st.radio("Navigation", ["Live Webcam", "Image Analysis", "System Info"])

if mode == "Live Webcam":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### Live Emotion Detection")
    st.write("Detect emotions in real time using your webcam.")

    cam_mode = st.radio(
        "Select stream type",
        ["Cloud WebRTC (for deployed app)", "Local OpenCV (for local development)"],
        horizontal=True,
    )

    if cam_mode == "Cloud WebRTC (for deployed app)":
        if WEBRTC_AVAILABLE:
            class EmotionProcessor(VideoProcessorBase):
                def __init__(self):
                    self.detector = get_detector()

                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.flip(img, 1)
                    img, _ = annotate_frame(img, self.detector, color_space="BGR")
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            rtc_config = RTCConfiguration(
                {
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]},
                    ]
                }
            )
            webrtc_streamer(
                key="emotion-recognition",
                rtc_configuration=rtc_config,
                video_processor_factory=EmotionProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        else:
            st.error("streamlit-webrtc is not installed. Run `pip install streamlit-webrtc av` to use cloud mode.")

    else:
        if "run_camera" not in st.session_state:
            st.session_state.run_camera = False

        c1, c2, _ = st.columns([1, 1, 2])
        if c1.button("Start Camera", use_container_width=True):
            st.session_state.run_camera = True
        if c2.button("Stop Camera", use_container_width=True):
            st.session_state.run_camera = False
            st.rerun()

        frame_placeholder = st.empty()

        if st.session_state.run_camera:
            detector = get_detector()
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.session_state.run_camera = False
                st.error("Could not open the local webcam.")
            else:
                try:
                    while st.session_state.run_camera:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Camera frame could not be read.")
                            break

                        frame = cv2.flip(frame, 1)
                        frame, _ = annotate_frame(frame, detector, color_space="BGR")
                        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        time.sleep(0.03)
                finally:
                    cap.release()
    st.markdown("</div>", unsafe_allow_html=True)

elif mode == "Image Analysis":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### Static Image Analysis")
    st.write("Upload an image to analyze facial expressions.")

    file = st.file_uploader("Select an image file", type=["jpg", "png", "jpeg"])

    if file:
        img = Image.open(file).convert("RGB")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, use_container_width=True, caption="Uploaded image")

        with col2:
            st.markdown("#### Analysis Report")
            if not is_trained:
                st.info("A trained model is required for emotion predictions. Face detection is still available.")

            if st.button("Run Analysis", use_container_width=True):
                with st.spinner("Analyzing..."):
                    img_np = np.array(img)
                    faces = get_detector().detect_faces(img_np, color_space="RGB")

                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face_roi = img_np[y : y + h, x : x + w]
                        st.success(f"Detected {len(faces)} face(s).")
                    else:
                        st.warning("No face bounds were detected. Running analysis on the full image.")
                        face_roi = img_np

                    if is_trained:
                        emotion, probs = predict_emotion(face_roi, color_space="RGB")
                        st.success(f"Detected Emotion: **{emotion}**")

                        df_probs = pd.DataFrame({"Confidence (%)": probs * 100}, index=EMOTIONS)
                        st.bar_chart(df_probs)
                    else:
                        st.warning("Demo mode: model predictions are disabled because the model could not be loaded.")
    st.markdown("</div>", unsafe_allow_html=True)

elif mode == "System Info":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### System Diagnostics")

    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    uptime = datetime.now() - st.session_state.start_time

    c1, c2, c3 = st.columns(3)
    c1.metric("Memory Usage", f"{mem_mb:.1f} MB")
    c2.metric("Session Uptime", str(uptime).split(".")[0])
    c3.metric("Compute Backend", "CUDA (GPU)" if torch.cuda.is_available() else "CPU")
    st.caption(model_status)
    st.markdown("</div>", unsafe_allow_html=True)
