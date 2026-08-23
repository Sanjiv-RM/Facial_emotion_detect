import os
import requests
import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="😊", layout="centered")

st.title("Facial Emotion Recognition")
st.write("Detect accurate emotions (**Happy**, **Sad**, **Angry**, **Surprise**, **Neutral**, etc.) using a Deep CNN.")

# 7 standard FER-2013 emotion classes
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOJI_MAP = {
    'Angry': ('😠', (0, 0, 255)),
    'Disgust': ('🤢', (0, 140, 255)),
    'Fear': ('😨', (200, 0, 200)),
    'Happy': ('😊', (0, 255, 0)),
    'Sad': ('😢', (255, 100, 0)),
    'Surprise': ('😲', (255, 255, 0)),
    'Neutral': ('😐', (180, 180, 180))
}

# Pretrained Mini-Xception model weights (FER-2013)
MODEL_FILE = "emotion_model.onnx"
MODEL_URL = "https://huggingface.co/trpakov/vit-face-expression/resolve/main/model.onnx"
BACKUP_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"

@st.cache_resource
def load_model():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Download weights if not already present
    if not os.path.exists(MODEL_FILE):
        with st.spinner("Downloading trained Deep Neural Network model..."):
            try:
                r = requests.get(BACKUP_URL, allow_redirects=True, timeout=30)
                with open(MODEL_FILE, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                st.error(f"Failed to download model weights: {e}")
                return None, cascade

    net = cv2.dnn.readNetFromONNX(MODEL_FILE)
    return net, cascade

net, face_cascade = load_model()

def classify_expression(img_bgr):
    h_img, w_img = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.15, 
        minNeighbors=5, 
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return img_bgr, []

    results = []

    for (x, y, w, h) in faces:
        # 10% bounding box expansion to keep jaw and eyebrows intact
        margin_x = int(w * 0.10)
        margin_y = int(h * 0.10)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_img, x + w + margin_x)
        y2 = min(h_img, y + h + margin_y)

        face_roi = gray[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        # Resize to 64x64 input dimension for CNN
        face_resized = cv2.resize(face_roi, (64, 64))
        
        # Standard input normalization
        blob = cv2.dnn.blobFromImage(
            face_resized, 
            scalefactor=1.0, 
            size=(64, 64), 
            mean=(0,), 
            swapRB=False, 
            crop=False
        )

        net.setInput(blob)
        raw_output = net.forward()[0]
        
        # Softmax computation
        exp_vals = np.exp(raw_output - np.max(raw_output))
        probs = exp_vals / exp_vals.sum()

        # Map predictions
        # Model output order: [Neutral, Happiness, Surprise, Sadness, Anger, Disgust, Fear, Contempt]
        model_classes = ['Neutral', 'Happy', 'Surprise', 'Sadness', 'Angry', 'Disgust', 'Fear', 'Neutral']
        
        emotion_scores = {}
        for cls_name, prob in zip(model_classes, probs):
            emotion_scores[cls_name] = emotion_scores.get(cls_name, 0.0) + float(prob)

        # Determine dominant emotion
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[dominant_emotion] * 100
        emoji, color = EMOJI_MAP.get(dominant_emotion, ('😐', (0, 255, 0)))

        # Draw bounding box and label
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 3)
        tag = f"{dominant_emotion} {emoji} ({confidence:.0f}%)"
        cv2.putText(img_bgr, tag, (x, max(y - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        results.append((dominant_emotion, emoji, confidence, emotion_scores))

    return img_bgr, results

# Tabs
tab1, tab2 = st.tabs(["📸 Take Snapshot", "📁 Upload Image"])

def display_dashboard(img, results):
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    if not results:
        st.warning("No face detected. Please ensure proper front lighting and face the camera directly.")
        return

    st.subheader("Detected Facial Expressions")
    for i, (name, emoji, conf, scores) in enumerate(results):
        st.success(f"**Face #{i+1}:** {name} {emoji} (**{conf:.1f}%**)")
        with st.expander(f"Full Probability Distribution (Face #{i+1})"):
            for emo_name, prob in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                emoji_icon, _ = EMOJI_MAP.get(emo_name, ('😐', ()))
                col1, col2 = st.columns([3, 7])
                col1.write(f"**{emo_name} {emoji_icon}**")
                col2.progress(min(prob, 1.0), text=f"{prob*100:.1f}%")

with tab1:
    cam_shot = st.camera_input("Take a photo:")
    if cam_shot is not None:
        file_bytes = np.asarray(bytearray(cam_shot.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated_img, results = classify_expression(img_bgr)
        display_dashboard(annotated_img, results)

with tab2:
    uploaded = st.file_uploader("Upload an image (JPG, PNG):", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated_img, results = classify_expression(img_bgr)
        display_dashboard(annotated_img, results)
