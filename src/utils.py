import cv2
import numpy as np
import torch
from torchvision import transforms

# --- Image Constants ---
IMG_SIZE = 48

# --- Define Preprocessing Transformation ---
# FER2013 is grayscale (1-channel) and 48x48
emotion_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.CenterCrop(44),          # Matching GitHub strategy
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_face(face_roi):
    """
    Preprocess a localized face region for model input.
    """
    # Convert to grayscale if needed
    if len(face_roi.shape) == 3:
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Apply transformation
    tensor = emotion_transform(face_roi)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor

def get_emotion_label(prediction_index):
    """
    Map model output index to string label.
    """
    labels = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral"
    }
    return labels.get(prediction_index, "Unknown")

class FaceDetector:
    """
    Wrapper for OpenCV Haar Cascade Face Detection.
    """
    def __init__(self):
        # Load pre-trained Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, image):
        """
        Detect faces with tuned parameters.
        - scaleFactor: 1.1 (closer to 1.0 is more accurate but slower)
        - minNeighbors: 6 (higher reduces false positives)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=6, 
            minSize=(30, 30)
        )
        return faces
