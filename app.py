import os
import urllib.request
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av

st.set_page_config(page_title="Live Facial Emotion Recognition", page_icon="😊", layout="centered")

st.title("Live Facial Emotion Recognition")
st.write("Real-time facial expression tracking with enhanced smile and feature sensitivity.")

# Emotion map
EMOTION_MAP = {
    0: ("Neutral", "😐", (200, 200, 200)),
    1: ("Happy", "😊", (0, 255, 0)),
    2: ("Surprised", "😲", (255, 255, 0)),
    3: ("Sad", "😢", (255, 100, 0)),
    4: ("Angry", "😠", (0, 0, 255)),
    5: ("Disgusted", "🤢", (0, 140, 255)),
    6: ("Fearful", "😨", (200, 0, 200)),
    7: ("Contempt", "😏", (128, 128, 128))
}

MODEL_PATH = "emotion-ferplus-8.onnx"
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"

@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading ONNX model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return net, cascade

net, face_cascade = load_models()

def process_frame(img):
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Improved multi-scale face detector
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=6, 
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        # Add 15% margin around the face to prevent cutting off smiles/mouth
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_img, x + w + margin_x)
        y2 = min(h_img, y + h + margin_y)

        face_roi = gray[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        # Resize to 64x64 and apply histogram equalization to balance shadows/lighting
        face_roi = cv2.equalizeHist(face_roi)
        face_resized = cv2.resize(face_roi, (64, 64))

        # Standard FERPlus blob normalization
        blob = cv2.dnn.blobFromImage(
            face_resized, 
            scalefactor=1.0 / 255.0, 
            size=(64, 64), 
            mean=(0.5,), 
            swapRB=False, 
            crop=False
        )

        net.setInput(blob)
        preds = net.forward()[0]

        # Softmax probability distribution
        exp_preds = np.exp(preds - np.max(preds))
        probs = exp_preds / exp_preds.sum()

        # Class balance correction: Reduce Neutral default bias slightly
        probs[0] *= 0.75 
        probs = probs / probs.sum()

        top_idx = int(np.argmax(probs))
        name, emoji, color = EMOTION_MAP[top_idx]
        confidence = probs[top_idx] * 100

        # Draw box and emotion badge
        tag = f"{name} {emoji} ({confidence:.1f}%)"
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, tag, (x, max(y - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

    return img

class VideoProcessor:
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        annotated_img = process_frame(img)
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# Live real-time webcam streamer
webrtc_streamer(
    key="live-emotion-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:global.stun.twilio.com:3478"]}
        ]
    }),
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
    async_processing=True
)
