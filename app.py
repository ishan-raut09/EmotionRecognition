import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from src.model import VGG
from src.utils import FaceDetector, preprocess_face, get_emotion_label
import time
import os
import psutil
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# --- Page Config ---
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()

st.set_page_config(
    page_title="Sentira Cloud - AI Emotion Intelligence",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Configurations ---
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# --- Production CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    * { transition: all 0.2s ease-in-out; }
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; color: #E0E0E0; }
    .stApp { background: radial-gradient(circle at top right, #1a1a2e, #16213e, #0f3460); }
    .glass-card {
        background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px;
        padding: 2rem; margin-bottom: 1.5rem; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .neural-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; letter-spacing: -1px;
    }
    .badge { padding: 4px 12px; border-radius: 50px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
    .badge-active { background: rgba(0, 255, 136, 0.1); color: #00ff88; border: 1px solid #00ff88; }
    .badge-offline { background: rgba(255, 60, 60, 0.1); color: #ff3c3c; border: 1px solid #ff3c3c; }
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_vgg_engine():
    model_path = "models/emotion_model.pth"
    if os.path.exists(model_path):
        try:
            model = VGG('VGG19')
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            return model, True
        except: return None, False
    return None, False

vgg_model, is_trained = load_vgg_engine()

@st.cache_resource
def get_detector():
    return FaceDetector()

# --- WebRTC Video Processor ---
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = get_detector()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror
        
        faces = self.detector.detect_faces(img)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (79, 172, 254), 2)
            
            if is_trained:
                face_roi = img[y:y+h, x:x+w]
                input_tensor = preprocess_face(face_roi)
                with torch.no_grad():
                    output = vgg_model(input_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
                    top_idx = np.argmax(probs)
                    emotion = get_emotion_label(top_idx)
                    conf = probs[top_idx] * 100
                
                cv2.putText(img, f"{emotion.upper()} {conf:.1f}%", (x, y-15), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 242, 254), 1)
            else:
                cv2.putText(img, "FACE DETECTED", (x, y-15), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI Layout ---
st.markdown("<h1 class='neural-header' style='font-size: 3.5rem; margin-bottom: 0;'>SENTIRA CLOUD</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1rem; opacity: 0.6; margin-top: -10px;'>PRODUCTION-GRADE COGNITIVE ANALYTICS</p>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/brain.png", width=60)
    st.title("Neural Engine")
    st.markdown(f"<span class='badge {'badge-active' if is_trained else 'badge-offline'}'>{'VGG19 ACTIVE' if is_trained else 'SIMULATION MODE'}</span>", unsafe_allow_html=True)
    st.markdown("---")
    mode = st.selectbox("Intelligence Stream", ["Visual Data Feed", "Static Image Analytics", "Diagnostics"])

if mode == "Visual Data Feed":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📡 Secure Web Stream")
    webrtc_streamer(
        key="emotion-recognition",
        mode=av.VideoProcessorBase,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.caption("🔒 SSL Encrypted • Direct Browser Access")
    st.markdown("</div>", unsafe_allow_html=True)

elif mode == "Static Image Analytics":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🧬 Static Pattern Ingestion")
    file = st.file_uploader("Select high-resolution profile", type=['jpg','png','jpeg'])
    if file:
        img = Image.open(file)
        st.image(img, use_container_width=True)
        if st.button("TRIGGER DEEP ANALYSIS"):
            if is_trained:
                with st.status("Decoding Neural Patterns...", expanded=True) as status:
                    img_np = np.array(img.convert('RGB'))
                    faces = get_detector().detect_faces(img_np)
                    if len(faces) > 0:
                        face_roi = img_np[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]
                        input_tensor = preprocess_face(face_roi)
                        with torch.no_grad():
                            output = vgg_model(input_tensor)
                            probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
                            top_idx = np.argmax(probs)
                            emotion = get_emotion_label(top_idx)
                        st.subheader(f"Result: {emotion.upper()} - {probs[top_idx]*100:.1f}%")
                        st.bar_chart(probs)
                        status.update(label="Analysis Complete", state="complete")
                    else: status.update(label="No Face Detected", state="error")
            else: st.warning("Please train the model first.")
    st.markdown("</div>", unsafe_allow_html=True)

elif mode == "Diagnostics":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 System Health")
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    layer_count = len(list(vgg_model.modules())) if is_trained else 0
    param_count = sum(p.numel() for p in vgg_model.parameters()) / 1e6 if is_trained else 0
    uptime = datetime.now() - st.session_state.start_time
    
    col1, col2, col3 = st.columns(3)
    col1.metric("System RAM", f"{mem_mb:.1f} MB")
    col2.metric("Neural Weights", f"{param_count:.1f} M")
    col3.metric("Neural Depth", f"{layer_count} Units")
    st.markdown("</div>", unsafe_allow_html=True)
