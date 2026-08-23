import os
import urllib.request
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Facial Emotion Detector", page_icon="😊", layout="centered")

st.title("Facial Emotion Recognition")
st.write("Real-time facial expression detector powered by OpenCV Deep Neural Network.")

# Download pre-trained ONNX emotion classification model if not present
MODEL_PATH = "emotion-ferplus-8.onnx"
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"

@st.cache_resource
def load_emotion_net():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading emotion model weights..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return cv2.dnn.readNetFromONNX(MODEL_PATH)

net = load_emotion_net()

# Emotion classes from FERPlus dataset
EMOTIONS = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']

# Load standard Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Crop and resize detected face ROI to 64x64 for the ONNX model
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (64, 64))
            blob = cv2.dnn.blobFromImage(face_resized, scalefactor=1.0, size=(64, 64), mean=(0,), swapRB=False, crop=False)

            # Perform DNN forward pass
            net.setInput(blob)
            preds = net.forward()[0]
            
            # Compute Softmax probabilities
            exp_preds = np.exp(preds - np.max(preds))
            probs = exp_preds / exp_preds.sum()
            idx = np.argmax(probs)
            
            label = f"{EMOTIONS[idx]} ({probs[idx]*100:.1f}%)"

            # Draw bounding box and prediction
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img

# WebRTC Streamer for real-time webcam feed
webrtc_streamer(
    key="emotion-detection",
    video_transformer_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
