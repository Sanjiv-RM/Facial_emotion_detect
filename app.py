import os
import requests
import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="😊", layout="centered")

st.title("Facial Emotion Recognition")
st.write("Detect distinct emotions (**Happy**, **Sad**, **Angry**, **Surprise**, etc.) with calibrated sensitivity.")

EMOTION_INFO = {
    'Angry': ('😠', (0, 0, 255)),
    'Disgust': ('🤢', (0, 140, 255)),
    'Fear': ('😨', (200, 0, 200)),
    'Happy': ('😊', (0, 255, 0)),
    'Sad': ('😢', (255, 120, 0)),
    'Surprise': ('😲', (255, 255, 0)),
    'Neutral': ('😐', (180, 180, 180))
}

MODEL_FILE = "emotion_ferplus.onnx"
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"

@st.cache_resource
def load_model():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if not os.path.exists(MODEL_FILE):
        with st.spinner("Downloading emotion classifier..."):
            r = requests.get(MODEL_URL, timeout=30)
            with open(MODEL_FILE, "wb") as f:
                f.write(r.content)
    net = cv2.dnn.readNetFromONNX(MODEL_FILE)
    return net, cascade

net, face_cascade = load_model()

def classify_face(img_bgr):
    h_img, w_img = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Detect face
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.12, 
        minNeighbors=5, 
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return img_bgr, []

    results = []

    for (x, y, w, h) in faces:
        # Include brow and chin margins
        margin_x = int(w * 0.12)
        margin_y = int(h * 0.15)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_img, x + w + margin_x)
        y2 = min(h_img, y + h + margin_y)

        face_roi = gray[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        # Contrast normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_enhanced = clahe.apply(face_roi)
        face_resized = cv2.resize(face_enhanced, (64, 64))

        # Prepare blob
        blob = cv2.dnn.blobFromImage(
            face_resized, 
            scalefactor=1.0, 
            size=(64, 64), 
            mean=(0,), 
            swapRB=False, 
            crop=False
        )

        net.setInput(blob)
        raw_logits = net.forward()[0]
        
        # Raw classes from FERPlus ONNX:
        # [0: neutral, 1: happiness, 2: surprise, 3: sadness, 4: anger, 5: disgust, 6: fear, 7: contempt]
        
        # Calibrate logits: Penalize neutral bias and boost subtle micro-expression logits
        calibrated_logits = np.copy(raw_logits)
        calibrated_logits[0] -= 1.4   # Neutral penalty
        calibrated_logits[3] += 0.8   # Sad boost
        calibrated_logits[4] += 0.7   # Anger boost
        calibrated_logits[1] += 0.3   # Happy adjustment

        # Softmax probability conversion
        exp_vals = np.exp(calibrated_logits - np.max(calibrated_logits))
        probs = exp_vals / exp_vals.sum()

        class_names = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear', 'Neutral']
        
        emotion_probs = {}
        for name, p in zip(class_names, probs):
            emotion_probs[name] = emotion_probs.get(name, 0.0) + float(p)

        dominant_emotion = max(emotion_probs, key=emotion_probs.get)
        confidence = emotion_probs[dominant_emotion] * 100
        emoji, color = EMOTION_INFO.get(dominant_emotion, ('😐', (0, 255, 0)))

        # Draw bounding box and label
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 3)
        tag = f"{dominant_emotion} {emoji} ({confidence:.0f}%)"
        cv2.putText(img_bgr, tag, (x, max(y - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        results.append((dominant_emotion, emoji, confidence, emotion_probs))

    return img_bgr, results

tab1, tab2 = st.tabs(["📸 Take Snapshot", "📁 Upload Image"])

def display_dashboard(img, results):
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    if not results:
        st.warning("No face detected. Please ensure your face is well-lit and facing the camera directly.")
        return

    st.subheader("Detected Facial Expressions")
    for i, (name, emoji, conf, scores) in enumerate(results):
        st.success(f"**Face #{i+1}:** {name} {emoji} (**{conf:.1f}%**)")
        with st.expander(f"Full Probability Distribution (Face #{i+1})"):
            for emo_name, prob in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                emoji_icon, _ = EMOTION_INFO.get(emo_name, ('😐', ()))
                col1, col2 = st.columns([3, 7])
                col1.write(f"**{emo_name} {emoji_icon}**")
                col2.progress(min(prob, 1.0), text=f"{prob*100:.1f}%")

with tab1:
    cam_shot = st.camera_input("Take a snapshot:")
    if cam_shot is not None:
        file_bytes = np.asarray(bytearray(cam_shot.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated_img, results = classify_face(img_bgr)
        display_dashboard(annotated_img, results)

with tab2:
    uploaded = st.file_uploader("Upload an image (JPG, PNG):", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated_img, results = classify_face(img_bgr)
        display_dashboard(annotated_img, results)
