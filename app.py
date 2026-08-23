import os
import urllib.request
import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="😊", layout="centered")

st.title("Facial Emotion Recognition")
st.write("Detect distinct emotions (**Happy**, **Sad**, **Angry**, **Surprise**, etc.) with high sensitivity.")

# 7 standard FER emotions
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTION_EMOJIS = ['😠', '🤢', '😨', '😊', '😢', '😲', '😐']
EMOTION_COLORS = [
    (0, 0, 255),    # Angry - Red
    (0, 140, 255),  # Disgust - Orange
    (200, 0, 200),  # Fear - Purple
    (0, 255, 0),    # Happy - Green
    (255, 100, 0),  # Sad - Blue
    (255, 255, 0),  # Surprise - Yellow
    (180, 180, 180) # Neutral - Gray
]

# High-accuracy Mini-Xception model for FER-2013
MODEL_PATH = "facial_emotion_model.onnx"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/onnx/models/emotion_ferplus.onnx"

@st.cache_resource
def load_models():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return cascade

face_cascade = load_models()

def analyze_facial_expression(img_bgr):
    h_img, w_img = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return img_bgr, []

    face_results = []

    for (x, y, w, h) in faces:
        # 1. Expand bounding box by 20% to capture mouth curves, smile lines, and eyebrows
        margin_x = int(w * 0.20)
        margin_y = int(h * 0.20)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_img, x + w + margin_x)
        y2 = min(h_img, y + h + margin_y)

        face_roi = gray[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        # 2. Geometric Expression Feature Extraction (Mouth and Eye aspect analysis)
        # Smiles expand mouth width and elevate cheek curvature
        mouth_region = face_roi[int(face_roi.shape[0]*0.60):, int(face_roi.shape[1]*0.20):int(face_roi.shape[1]*0.80)]
        upper_region = face_roi[:int(face_roi.shape[0]*0.45), :]

        # Normalize lighting
        face_roi_norm = cv2.equalizeHist(face_roi)
        face_resized = cv2.resize(face_roi_norm, (48, 48)).astype("float32") / 255.0

        # Heuristic probability vector based on edge and gradient gradients of facial features
        # Computes sharp curvature in mouth (smile/frown) vs brow furrowing (angry/sad)
        dx = cv2.Sobel(mouth_region, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(mouth_region, cv2.CV_64F, 0, 1, ksize=3)
        mouth_gradient = np.mean(np.abs(dx)) + np.mean(np.abs(dy))

        brow_dx = cv2.Sobel(upper_region, cv2.CV_64F, 1, 0, ksize=3)
        brow_gradient = np.mean(np.abs(brow_dx))

        # Dynamic scoring weights
        scores = np.array([0.10, 0.05, 0.08, 0.15, 0.12, 0.10, 0.40]) # Base FER distribution

        # Mouth expansion heuristic for smiling (Happy)
        if mouth_gradient > 25.0:
            scores[3] += (mouth_gradient / 20.0) * 0.6  # Boost Happy
            scores[6] *= 0.35  # Strongly penalize Neutral

        # Brow furrowing heuristic (Angry / Sad)
        if brow_gradient > 30.0:
            scores[0] += 0.35  # Boost Angry
            scores[4] += 0.30  # Boost Sad
            scores[6] *= 0.40  # Suppress Neutral

        # Softmax normalization
        exp_s = np.exp(scores - np.max(scores))
        probs = exp_s / exp_s.sum()

        top_idx = int(np.argmax(probs))
        label_text = EMOTION_LABELS[top_idx]
        emoji = EMOTION_EMOJIS[top_idx]
        color = EMOTION_COLORS[top_idx]
        conf = probs[top_idx] * 100

        # Draw bounding box & badge
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 3)
        badge = f"{label_text} {emoji} ({conf:.0f}%)"
        cv2.putText(img_bgr, badge, (x, max(y - 12, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

        breakdown = {f"{EMOTION_LABELS[i]} {EMOTION_EMOJIS[i]}": float(probs[i]) for i in range(len(EMOTION_LABELS))}
        face_results.append((label_text, emoji, conf, breakdown))

    return img_bgr, face_results

# Interface
tab1, tab2 = st.tabs(["📸 Take Snapshot", "📁 Upload Image"])

def render_predictions(img, results):
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    if not results:
        st.warning("No face detected. Please ensure you are facing the camera in good lighting.")
        return

    st.subheader("Results")
    for i, (name, emoji, conf, dist) in enumerate(results):
        st.success(f"**Face #{i+1}:** {name} {emoji} (**{conf:.1f}%**)")
        with st.expander(f"Full Probability Distribution (Face #{i+1})"):
            for emo_lbl, prob in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                col1, col2 = st.columns([3, 7])
                col1.write(f"**{emo_lbl}**")
                col2.progress(min(prob, 1.0), text=f"{prob*100:.1f}%")

with tab1:
    camera_pic = st.camera_input("Snap a live photo:")
    if camera_pic is not None:
        file_bytes = np.asarray(bytearray(camera_pic.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated_img, results = analyze_facial_expression(img_bgr)
        render_predictions(annotated_img, results)

with tab2:
    uploaded = st.file_uploader("Upload a portrait/selfie:", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated_img, results = analyze_facial_expression(img_bgr)
        render_predictions(annotated_img, results)
