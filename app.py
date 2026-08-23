import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import mediapipe as mp

st.set_page_config(page_title="Accurate Facial Emotion Recognition", page_icon="😊", layout="centered")

st.title("Facial Emotion Recognition")
st.write("Powered by **MediaPipe Landmark Face Tracking** and **HuggingFace Vision Transformers (ViT)** for high accuracy.")

EMOTION_META = {
    'happy': ('Happy', '😊', (0, 255, 0)),
    'sad': ('Sad', '😢', (255, 120, 0)),
    'angry': ('Angry', '😠', (0, 0, 255)),
    'surprise': ('Surprise', '😲', (255, 255, 0)),
    'fear': ('Fear', '😨', (200, 0, 200)),
    'disgust': ('Disgust', '🤢', (0, 140, 255)),
    'neutral': ('Neutral', '😐', (180, 180, 180))
}

@st.cache_resource
def load_models():
    # Hugging Face ViT model fine-tuned specifically on facial emotions
    model_id = "dima806/facial_emotions_image_detection"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()

    # MediaPipe Face Detection for tight, accurate face bounding boxes
    mp_face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=1, 
        min_detection_confidence=0.5
    )
    return processor, model, mp_face_detection

processor, model, face_detector = load_models()

def analyze_facial_emotions(img_bgr):
    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detection_results = face_detector.process(img_rgb)

    if not detection_results.detections:
        return img_bgr, []

    results = []

    for detection in detection_results.detections:
        bbox = detection.location_data.relative_bounding_box
        
        # Convert relative coordinates to pixel values
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        # Margin expansion to preserve chin, forehead, and cheek curves
        mx = int(bw * 0.10)
        my = int(bh * 0.10)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + bw + mx)
        y2 = min(h, y + bh + my)

        face_crop = img_rgb[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # Convert crop to PIL for Vision Transformer inference
        pil_face = Image.fromarray(face_crop)
        inputs = processor(images=pil_face, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].numpy()

        id2label = model.config.id2label
        raw_scores = {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}

        # Determine dominant prediction
        dominant_key = max(raw_scores, key=raw_scores.get)
        label_name, emoji, color = EMOTION_META.get(dominant_key, (dominant_key.capitalize(), '😐', (0, 255, 0)))
        confidence = raw_scores[dominant_key] * 100

        # Draw bounding box and label badge
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
        badge = f"{label_name} {emoji} ({confidence:.0f}%)"
        cv2.putText(img_bgr, badge, (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

        formatted_scores = {EMOTION_META.get(k, (k.capitalize(), ''))[0] + " " + EMOTION_META.get(k, ('', '😐'))[1]: v for k, v in raw_scores.items()}
        results.append((label_name, emoji, confidence, formatted_scores))

    return img_bgr, results

# UI
tab1, tab2 = st.tabs(["📸 Take Snapshot", "📁 Upload Image"])

def display_dashboard(img_bgr, results):
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    if not results:
        st.warning("No face detected. Please face the camera directly in clear lighting.")
        return

    st.subheader("Emotion Analysis Output")
    for i, (name, emoji, conf, scores) in enumerate(results):
        st.success(f"**Face #{i+1}:** {name} {emoji} (**{conf:.1f}%**)")
        with st.expander(f"Full Confidence Breakdown (Face #{i+1})"):
            for emo_label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                col1, col2 = st.columns([3, 7])
                col1.write(f"**{emo_label}**")
                col2.progress(min(score, 1.0), text=f"{score*100:.1f}%")

with tab1:
    camera_file = st.camera_input("Snap a live photo:")
    if camera_file is not None:
        file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated, results = analyze_facial_emotions(img_bgr)
        display_dashboard(annotated, results)

with tab2:
    uploaded = st.file_uploader("Upload a face image (JPG/PNG):", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated, results = analyze_facial_emotions(img_bgr)
        display_dashboard(annotated, results)
