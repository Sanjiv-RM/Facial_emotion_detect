import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="😊", layout="centered")

st.title("Facial Emotion Recognition")
st.write("Accurate emotion detection (**Happy**, **Sad**, **Angry**, **Surprise**, etc.) powered by Hugging Face Vision Transformers.")

EMOTION_META = {
    'happy': ('Happy', '😊', '#28a745'),
    'sad': ('Sad', '😢', '#007bff'),
    'angry': ('Angry', '😠', '#dc3545'),
    'surprise': ('Surprise', '😲', '#ffc107'),
    'fear': ('Fear', '😨', '#6f42c1'),
    'disgust': ('Disgust', '🤢', '#fd7e14'),
    'neutral': ('Neutral', '😐', '#6c757d')
}

@st.cache_resource
def load_model():
    model_id = "dima806/facial_emotions_image_detection"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    return processor, model

processor, model = load_model()

def classify_expression(pil_img):
    # Ensure RGB format
    image_rgb = pil_img.convert("RGB")
    
    # Process image through Vision Transformer
    inputs = processor(images=image_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0].numpy()

    id2label = model.config.id2label
    scores = {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}

    dominant_key = max(scores, key=scores.get)
    name, emoji, hex_color = EMOTION_META.get(dominant_key, (dominant_key.capitalize(), '😐', '#6c757d'))
    confidence = scores[dominant_key] * 100

    formatted_scores = {
        EMOTION_META.get(k, (k.capitalize(), ''))[0] + " " + EMOTION_META.get(k, ('', '😐'))[1]: v 
        for k, v in scores.items()
    }

    return name, emoji, confidence, formatted_scores

tab1, tab2 = st.tabs(["📸 Take Snapshot", "📁 Upload Image"])

def display_results(pil_img, name, emoji, conf, scores):
    st.image(pil_img, use_container_width=True)
    st.subheader("Emotion Analysis Result")
    st.success(f"**Dominant Emotion:** {name} {emoji} (**{conf:.1f}%**)")
    
    with st.expander("View Full Confidence Breakdown"):
        for label, prob in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            col1, col2 = st.columns([3, 7])
            col1.write(f"**{label}**")
            col2.progress(min(prob, 1.0), text=f"{prob * 100:.1f}%")

with tab1:
    photo = st.camera_input("Take a photo:")
    if photo is not None:
        pil_image = Image.open(photo)
        name, emoji, conf, scores = classify_expression(pil_image)
        display_results(pil_image, name, emoji, conf, scores)

with tab2:
    uploaded = st.file_uploader("Upload an image (JPG, PNG):", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        pil_image = Image.open(uploaded)
        name, emoji, conf, scores = classify_expression(pil_image)
        display_results(pil_image, name, emoji, conf, scores)
