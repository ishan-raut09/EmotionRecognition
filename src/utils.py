import cv2
import numpy as np
import torch
from torchvision import transforms

IMG_SIZE = 48

# --- Define Preprocessing Transformation ---
# ImageNet Standard Normalization (Required for Pretrained ResNet)
emotion_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_face(face_image):
    # Ensure 3-channels (RGB)
    if len(face_image.shape) == 2:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
    elif face_image.shape[2] == 3:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
    face_image = cv2.resize(face_image, (48, 48))
    img_tensor = emotion_transform(face_image)
    return img_tensor.unsqueeze(0)

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_faces(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # minNeighbors tweaked for balance between detection rate and false positives
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        return faces

def get_emotion_label(idx):
    emotions = {
        0: 'Angry', 1: 'Disgust', 2: 'Fear', 
        3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'
    }
    return emotions.get(idx, 'Unknown')
