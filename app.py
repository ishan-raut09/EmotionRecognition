import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from src.model import EmotionResNet
from src.utils import FaceDetector, preprocess_face, get_emotion_label
import time
import os
import psutil
from datetime import datetime

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


# --- Page Config ---
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()

st.set_page_config(
    page_title="Sentira - AI Emotion Intelligence",
    page_icon="🎭",
    layout="wide"
)

# --- Production CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean Minimalist Background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Headings */
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Main Title Styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #ffffff, #a5a5a5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 2.5rem;
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    /* Buttons */
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
    
    /* Badges */
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
    
    /* Sidebar styling tweaks */
    .css-1d391kg {
        background-color: #161a22;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_vgg_engine():
    model_path = "models/emotion_model.pth"
    if os.path.exists(model_path):
        try:
            model = EmotionResNet(pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            return model, True
        except Exception as e: 
            st.error(f"Error loading model: {e}")
            return None, False
    return None, False

vgg_model, is_trained = load_vgg_engine()

@st.cache_resource
def get_detector():
    return FaceDetector()

# --- UI Header ---
st.markdown("<h1 class='main-title'>Sentira Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time Facial Emotion Recognition</p>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/brain.png", width=60)
    st.title("Dashboard")
    st.markdown(f"<div class='badge badge-active'>{'Model: ResNet18 Active' if is_trained else 'Model: Demo Mode'}</div>", unsafe_allow_html=True)
    st.markdown("---")
    mode = st.radio("Navigation", ["📷 Live Webcam", "🖼️ Image Analysis", "📊 System Info"])

if mode == "📷 Live Webcam":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📷 Live Emotion Detection")
    st.write("Detect emotions in real-time using your webcam.")
    
    cam_mode = st.radio("Select Stream Type", ["Cloud WebRTC (For Deployed App)", "Local OpenCV (For Local Dev)"], horizontal=True)
    
    if cam_mode == "Cloud WebRTC (For Deployed App)":
        if WEBRTC_AVAILABLE:
            class EmotionProcessor(VideoProcessorBase):
                def __init__(self):
                    self.detector = get_detector()
                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.flip(img, 1)
                    faces = self.detector.detect_faces(img)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (46, 102, 255), 2)
                        if is_trained:
                            face_roi = img[y:y+h, x:x+w]
                            input_tensor = preprocess_face(face_roi)
                            with torch.no_grad():
                                output = vgg_model(input_tensor)
                                probs = torch.nn.functional.softmax(output / 1.2, dim=1).cpu().numpy()[0]
                                top_idx = np.argmax(probs)
                                emotion = get_emotion_label(top_idx)
                            cv2.putText(img, f"{emotion} {probs[top_idx]*100:.1f}%", (x, y-15), 
                                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
                        else:
                            cv2.putText(img, "Face Detected", (x, y-15), cv2.FONT_HERSHEY_DUPLEX, 0.7, (16, 185, 129), 1)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            st.markdown("#### 📡 Secure Web Stream")
            RTC_CONFIG = RTCConfiguration({
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                    {"urls": ["stun:stun3.l.google.com:19302"]},
                    {"urls": ["stun:stun4.l.google.com:19302"]},
                ]
            })
            webrtc_streamer(
                key="emotion-recognition",
                rtc_configuration=RTC_CONFIG,
                video_processor_factory=EmotionProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        else:
            st.error("⚠️ `streamlit-webrtc` is not installed! Run `pip install streamlit-webrtc av` to use Cloud mode.")
    
    else:
        if 'run_camera' not in st.session_state:
            st.session_state.run_camera = False

        c1, c2, c3 = st.columns([1, 1, 2])
        if c1.button("Start Camera", use_container_width=True): st.session_state.run_camera = True
        if c2.button("Stop Camera", use_container_width=True): 
            st.session_state.run_camera = False
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        frame_placeholder = st.empty()

        if st.session_state.run_camera:
            detector = get_detector()
            cap = cv2.VideoCapture(0)
            
            while st.session_state.run_camera:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                
                faces = detector.detect_faces(frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (46, 102, 255), 2)
                    
                    if is_trained:
                        face_roi = frame[y:y+h, x:x+w]
                        input_tensor = preprocess_face(face_roi)
                        with torch.no_grad():
                            output = vgg_model(input_tensor)
                            probs = torch.nn.functional.softmax(output / 1.2, dim=1).cpu().numpy()[0]
                            top_idx = np.argmax(probs)
                            emotion = get_emotion_label(top_idx)
                            cv2.putText(frame, f"{emotion} {probs[top_idx]*100:.1f}%", (x, y-15), 
                                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
                    else:
                        cv2.putText(frame, "Face Detected", (x, y-15), 
                                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (16, 185, 129), 1)

                try:
                    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                except:
                    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            cap.release()
    st.markdown("</div>", unsafe_allow_html=True)

elif mode == "🖼️ Image Analysis":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🖼️ Static Image Analysis")
    st.write("Upload an image to analyze facial expressions.")
    
    file = st.file_uploader("Select an image file", type=['jpg','png','jpeg'])
    
    if file:
        img = Image.open(file)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, use_container_width=True, caption="Uploaded Profile")
            
        with col2:
            st.markdown("#### Analysis Report")
            if st.button("Run Neural Analysis", use_container_width=True):
                with st.spinner("Analyzing..."):
                    img_np = np.array(img.convert('RGB'))
                    faces = get_detector().detect_faces(img_np)
                    
                    if len(faces) > 0:
                        (x,y,w,h) = faces[0]
                        face_roi = img_np[y:y+h, x:x+w]
                    else:
                        st.warning("⚠️ Auto-Focus failed to detect face bounds. Running neural analysis on the full image...")
                        face_roi = img_np
                        
                    input_tensor = preprocess_face(face_roi)
                    with torch.no_grad():
                        output = vgg_model(input_tensor)
                        probs = torch.nn.functional.softmax(output / 1.2, dim=1).cpu().numpy()[0]
                        res = get_emotion_label(np.argmax(probs))
                    
                    st.success(f"Detected Emotion: **{res}**")
                    
                    import pandas as pd
                    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
                    df_probs = pd.DataFrame({"Confidence (%)": probs * 100}, index=emotions)
                    st.bar_chart(df_probs)
    st.markdown("</div>", unsafe_allow_html=True)

elif mode == "📊 System Info":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 System Diagnostics")
    
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    uptime = datetime.now() - st.session_state.start_time
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Memory Usage", f"{mem_mb:.1f} MB")
    c2.metric("Session Uptime", str(uptime).split('.')[0])
    c3.metric("Compute Backend", "CUDA (GPU)" if torch.cuda.is_available() else "CPU")
    st.markdown("</div>", unsafe_allow_html=True)
