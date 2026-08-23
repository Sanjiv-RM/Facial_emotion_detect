import os
import urllib.request
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Facial Emotion Detector", page_icon="😊", layout="centered")

st.title("Facial Emotion Recognition")
st.write("Take a snapshot with your camera or upload an image to detect facial emotions instantly.")

# Emotion categories
EMOTIONS = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']

# Download pre-trained ONNX emotion model if not present
MODEL_PATH = "emotion-ferplus-8.onnx"
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"

@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Loading AI model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return net, cascade

net, face_cascade = load_models()

def detect_and_draw(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))

    if len(faces) == 0:
        st.warning("No face detected. Please ensure your face is well-lit and facing the camera.")
        return img_bgr

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
            
        face_resized = cv2.resize(face_roi, (64, 64))
        blob = cv2.dnn.blobFromImage(face_resized, scalefactor=1.0, size=(64, 64), mean=(0,), swapRB=False, crop=False)

        net.setInput(blob)
        preds = net.forward()[0]
        
        # Softmax
        exp_preds = np.exp(preds - np.max(preds))
        probs = exp_preds / exp_preds.sum()
        idx = np.argmax(probs)
        
        label = f"{EMOTIONS[idx]} ({probs[idx]*100:.1f}%)"

        # Draw box and emotion tag
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img_bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return img_bgr

tab1, tab2 = st.tabs(["📸 Live Camera Snapshot", "📁 Upload Photo"])

with tab1:
    camera_photo = st.camera_input("Take a photo")
    if camera_photo is not None:
        img_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        result_bgr = detect_and_draw(img_bgr)
        st.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), caption="Emotion Detected", use_container_width=True)

with tab2:
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        result_bgr = detect_and_draw(img_bgr)
        st.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), caption="Emotion Detected", use_container_width=True)
