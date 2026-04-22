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
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
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
        font-weight: 800;
    }
    .badge { padding: 4px 12px; border-radius: 50px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
    .badge-active { background: rgba(0, 255, 136, 0.1); color: #00ff88; border: 1px solid #00ff88; }
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_vgg_engine():
    model_path = "models/emotion_model.pth"
    if os.path.exists(model_path):
        try:
            model = VGG('VGG19')
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            return model, True
        except: return None, False
    return None, False

vgg_model, is_trained = load_vgg_engine()

@st.cache_resource
def get_detector():
    return FaceDetector()

# --- UI Header ---
st.markdown("<h1 class='neural-header' style='font-size: 3.5rem; margin-bottom: 0;'>SENTIRA PRO</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/brain.png", width=60)
    st.title("Neural Engine")
    st.markdown(f"<span class='badge'>{'VGG19 ACTIVE' if is_trained else 'MODE: DEMO'}</span>", unsafe_allow_html=True)
    st.markdown("---")
    mode = st.selectbox("Intelligence Stream", ["📷 Multi-Modal Webcam", "🧬 Static Image Analytics", "📊 Diagnostics"])

if mode == "📷 Multi-Modal Webcam":
    if 'run_camera' not in st.session_state:
        st.session_state.run_camera = False

    c1, c2 = st.columns(2)
    if c1.button("Initalize Sensors"): st.session_state.run_camera = True
    if c2.button("Terminate Feed"): 
        st.session_state.run_camera = False
        st.rerun()

    frame_placeholder = st.empty()
    chart_placeholder = st.empty()

    if st.session_state.run_camera:
        detector = get_detector()
        cap = cv2.VideoCapture(0)
        
        while st.session_state.run_camera:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            faces = detector.detect_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (79, 172, 254), 2)
                
                if is_trained:
                    face_roi = frame[y:y+h, x:x+w]
                    input_tensor = preprocess_face(face_roi)
                    with torch.no_grad():
                        output = vgg_model(input_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
                        top_idx = np.argmax(probs)
                        emotion = get_emotion_label(top_idx)
                        cv2.putText(frame, f"{emotion} {probs[top_idx]*100:.1f}%", (x, y-15), 
                                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 242, 254), 1)
                else:
                    cv2.putText(frame, "DETECTION MODE", (x, y-15), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1)

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        cap.release()

elif mode == "🧬 Static Image Analytics":
    file = st.file_uploader("Upload Profile", type=['jpg','png','jpeg'])
    if file:
        img = Image.open(file)
        st.image(img, use_container_width=True)
        if st.button("ANALYSIS"):
            img_np = np.array(img.convert('RGB'))
            faces = get_detector().detect_faces(img_np)
            if len(faces) > 0:
                (x,y,w,h) = faces[0]
                face_roi = img_np[y:y+h, x:x+w]
                input_tensor = preprocess_face(face_roi)
                with torch.no_grad():
                    output = vgg_model(input_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
                    res = get_emotion_label(np.argmax(probs))
                st.subheader(f"Detection: {res}")
                st.bar_chart(probs)

elif mode == "📊 Diagnostics":
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    uptime = datetime.now() - st.session_state.start_time
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("System RAM", f"{mem_mb:.1f} MB")
    c2.metric("Uptime", str(uptime).split('.')[0])
    st.metric("Compute Engine", "RTX 3050 (CUDA)" if torch.cuda.is_available() else "CPU")
    st.markdown("</div>", unsafe_allow_html=True)
