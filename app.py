import streamlit as st
from PIL import Image, ImageOps
from transformers import pipeline
from facenet_pytorch import MTCNN
import numpy as np
import torch

st.set_page_config(page_title="Deep Learning Emotion AI", page_icon="😊", layout="centered")
st.title("Facial Emotion Recognition")
st.write("Deep Learning facial expression classifier powered by **Vision Transformer (ViT)**.")

EMOJI_DICT = {
    "happy": ("Happy", "😊", "#28a745"),
    "sad": ("Sad", "😢", "#007bff"),
    "angry": ("Angry", "😠", "#dc3545"),
    "surprise": ("Surprise", "😲", "#ffc107"),
    "fear": ("Fear", "😨", "#6f42c1"),
    "disgust": ("Disgust", "🤢", "#fd7e14"),
    "neutral": ("Neutral", "😐", "#6c757d"),
}


@st.cache_resource
def load_deep_learning_pipeline():
    # 91% accuracy Vision Transformer fine-tuned on facial emotions
    return pipeline(
        "image-classification",
        model="dima806/facial_emotions_image_detection",
        device=-1  # CPU inference
    )


@st.cache_resource
def load_face_detector():
    # MTCNN face detector - keep_all=True so we can flag/handle multiple faces
    return MTCNN(keep_all=True, device="cpu", post_process=False)


classifier = load_deep_learning_pipeline()
face_detector = load_face_detector()


def detect_and_crop_face(pil_image, margin=0.35):
    """
    Detects the largest face in the image and returns a cropped, margin-padded
    version. Falls back to the original image (with a warning flag) if no
    face is found, since the classifier still needs *some* input.
    """
    img_rgb = pil_image.convert("RGB")
    boxes, probs = face_detector.detect(img_rgb)

    if boxes is None or len(boxes) == 0:
        return img_rgb, False, 0

    # Pick the largest detected face by box area (most likely the main subject)
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    largest_idx = int(np.argmax(areas))
    box = boxes[largest_idx]

    w, h = img_rgb.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1

    # Add margin around the face so we don't crop off chin/forehead,
    # which the ViT model relies on for expression cues
    x1 = max(0, x1 - bw * margin)
    y1 = max(0, y1 - bh * margin)
    x2 = min(w, x2 + bw * margin)
    y2 = min(h, y2 + bh * margin)

    cropped = img_rgb.crop((int(x1), int(y1), int(x2), int(y2)))
    return cropped, True, len(boxes)


def classify_with_tta(face_img):
    """
    Runs the classifier on the face crop and its horizontal flip, then
    averages the two probability distributions. This smooths out
    borderline predictions that flip on tiny pixel-level differences.
    """
    flipped = ImageOps.mirror(face_img)

    results_a = classifier(face_img)
    results_b = classifier(flipped)

    scores_a = {r["label"].lower(): r["score"] for r in results_a}
    scores_b = {r["label"].lower(): r["score"] for r in results_b}

    combined = {
        label: (scores_a.get(label, 0) + scores_b.get(label, 0)) / 2
        for label in set(scores_a) | set(scores_b)
    }
    return combined


def predict_emotion(pil_image):
    # Fix orientation first (EXIF), then detect + crop the face
    oriented = ImageOps.exif_transpose(pil_image).convert("RGB")
    face_img, face_found, num_faces = detect_and_crop_face(oriented)

    scores = classify_with_tta(face_img)

    top_label = max(scores, key=scores.get)
    name, emoji, color = EMOJI_DICT.get(top_label, (top_label.capitalize(), "😐", "#6c757d"))
    confidence = scores[top_label] * 100

    formatted_scores = {
        EMOJI_DICT.get(k, (k.capitalize(), ""))[0] + " " + EMOJI_DICT.get(k, ("", "😐"))[1]: v
        for k, v in scores.items()
    }

    return name, emoji, confidence, formatted_scores, face_img, face_found, num_faces


tab1, tab2 = st.tabs(["📸 Take Snapshot", "📁 Upload Image"])


def display_dashboard(name, emoji, conf, scores, img, face_found, num_faces):
    if not face_found:
        st.warning(
            "No face was clearly detected — classifying the full image instead. "
            "Results may be less accurate. Try a well-lit, front-facing shot."
        )
    elif num_faces > 1:
        st.info(f"{num_faces} faces detected — showing results for the largest/most prominent one.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, use_container_width=True, caption="Face used for classification")

    with col2:
        st.markdown(f"### Detected Expression\n## {name} {emoji}")
        st.metric(label="Model Confidence", value=f"{conf:.1f}%")
        if conf < 45:
            st.caption("⚠️ Low confidence — result may be unreliable.")

    st.write("---")
    st.subheader("All Emotion Probabilities")
    for emo_label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        c1, c2 = st.columns([3, 7])
        c1.write(f"**{emo_label}**")
        c2.progress(min(score, 1.0), text=f"{score * 100:.1f}%")


with tab1:
    photo = st.camera_input("Take a photo:")
    if photo is not None:
        pil_img = Image.open(photo)
        with st.spinner("Analyzing expression..."):
            name, emoji, conf, scores, face_img, found, n = predict_emotion(pil_img)
        display_dashboard(name, emoji, conf, scores, face_img, found, n)

with tab2:
    uploaded = st.file_uploader("Upload an image (JPG, PNG):", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        pil_img = Image.open(uploaded)
        with st.spinner("Analyzing expression..."):
            name, emoji, conf, scores, face_img, found, n = predict_emotion(pil_img)
        display_dashboard(name, emoji, conf, scores, face_img, found, n)
