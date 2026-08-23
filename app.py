import os
import urllib.request
import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Facial Emotion Detector", page_icon="😊", layout="centered")

st.title("Facial Emotion Recognition")
st.write("Take a snapshot or upload a photo to detect distinct emotions like **Happy**, **Sad**, **Angry**, and more.")

# Emotion map with clean labels and emojis
EMOTION_MAP = {
    0: ("Neutral", "😐", (180, 180, 180)),
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
        with st.spinner("Downloading AI model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return net, cascade

net, face_cascade = load_models()

def analyze_emotions(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    if len(faces) == 0:
        return img_bgr, None

    all_emotions_detected = []

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
            
        face_resized = cv2.resize(face_roi, (64, 64))
        blob = cv2.dnn.blobFromImage(face_resized, scalefactor=1.0, size=(64, 64), mean=(0,), swapRB=False, crop=False)

        net.setInput(blob)
        preds = net.forward()[0]
        
        # Calculate Softmax distribution
        exp_preds = np.exp(preds - np.max(preds))
        probs = exp_preds / exp_preds.sum()
        top_idx = int(np.argmax(probs))
        
        name, emoji, color = EMOTION_MAP[top_idx]
        confidence = probs[top_idx] * 100
        
        # Annotate face with emotion tag
        tag = f"{name} {emoji} ({confidence:.1f}%)"
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 3)
        cv2.putText(img_bgr, tag, (x, max(y - 12, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # Store probability breakdown
        emotion_breakdown = {EMOTION_MAP[i][0] + " " + EMOTION_MAP[i][1]: float(probs[i]) for i in range(8)}
        all_emotions_detected.append((name, emoji, confidence, emotion_breakdown))

    return img_bgr, all_emotions_detected

tab1, tab2 = st.tabs(["📸 Take Snapshot", "📁 Upload Image"])

def display_results(result_img, face_data):
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    if not face_data:
        st.warning("No face detected! Please ensure proper lighting and face the camera directly.")
        return

    st.subheader("Emotion Analysis Result")
    for i, (name, emoji, conf, breakdown) in enumerate(face_data):
        st.success(f"**Face #{i+1} Dominant Emotion:** {name} {emoji} (**{conf:.1f}%**)")
        
        with st.expander(f"View Full Confidence Breakdown for Face #{i+1}"):
            for emotion_label, prob in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
                col1, col2 = st.columns([3, 7])
                col1.write(f"**{emotion_label}**")
                col2.progress(min(prob, 1.0), text=f"{prob * 100:.1f}%")

with tab1:
    photo = st.camera_input("Click a photo:")
    if photo is not None:
        img_bytes = np.asarray(bytearray(photo.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        annotated_img, face_data = analyze_emotions(img_bgr)
        display_results(annotated_img, face_data)

with tab2:
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG):", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated_img, face_data = analyze_emotions(img_bgr)
        display_results(annotated_img, face_data)
