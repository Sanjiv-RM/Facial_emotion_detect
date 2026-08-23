import os
import urllib.request
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av

st.set_page_config(page_title="Facial Emotion Detector", page_icon="😊", layout="centered")

st.title("Facial Emotion Recognition")
st.write("Detect facial expressions in real-time from webcam or via photo upload.")

# Emotion categories
EMOTIONS = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']

# Download pre-trained ONNX emotion model if not cached
MODEL_PATH = "emotion-ferplus-8.onnx"
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"

@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return net, cascade

net, face_cascade = load_models()

def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
        face_resized = cv2.resize(face_roi, (64, 64))
        blob = cv2.dnn.blobFromImage(face_resized, scalefactor=1.0, size=(64, 64), mean=(0,), swapRB=False, crop=False)

        net.setInput(blob)
        preds = net.forward()[0]
        exp_preds = np.exp(preds - np.max(preds))
        probs = exp_preds / exp_preds.sum()
        idx = np.argmax(probs)
        
        label = f"{EMOTIONS[idx]} ({probs[idx]*100:.1f}%)"

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return img

tab1, tab2 = st.tabs(["🎥 Live Webcam", "📸 Upload Photo"])

with tab1:
    st.write("Click **START** to begin real-time emotion detection:")
    
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        processed_img = process_image(img)
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

    webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]}
            ]
        }),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
        async_processing=True,
    )

with tab2:
    uploaded_file = st.file_uploader("Upload an image containing faces", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        processed = process_image(img)
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption="Analyzed Image", use_container_width=True)
