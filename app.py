import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from facenet_pytorch import MTCNN

st.set_page_config(page_title="Fine-Tuned Emotion AI", page_icon="😊", layout="centered")

st.title("Facial Emotion Recognition")
st.write("Fine-tuned facial expression classification powered by **MTCNN Face Alignment** and **Vision Transformer (ViT)**.")

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
def load_pipeline():
    # 1. MTCNN for precision face cropping
    mtcnn = MTCNN(keep_all=False, select_largest=True, post_process=False, device='cpu')
    
    # 2. Vision Transformer model fine-tuned for facial emotion classification
    model_id = "dima806/facial_emotions_image_detection"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    
    return mtcnn, processor, model

mtcnn, processor, model = load_pipeline()

def process_and_classify(pil_img):
    # Auto-orient image based on EXIF (prevents mobile rotation errors)
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    
    # 1. Detect and tightly crop to the face
    face_crop = mtcnn.extract(pil_img, None, save_path=None)
    
    if face_crop is not None:
        # Convert tensor [C, H, W] back to PIL Image [H, W, C]
        face_np = face_crop.permute(1, 2, 0).byte().numpy()
        inference_img = Image.fromarray(face_np)
        cropped_display = inference_img
    else:
        # Fallback to center-crop if MTCNN misses
        cropped_display = pil_img
        inference_img = pil_img

    # 2. Transformer Feature Extraction & Inference
    inputs = processor(images=inference_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Temperature scaling (T=0.9) to sharpen subtle micro-expressions
        probs = F.softmax(logits / 0.9, dim=-1)[0].numpy()

    id2label = model.config.id2label
    scores = {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}

    # Sort emotions by confidence
    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Primary & Secondary emotions
    top_key, top_prob = sorted_emotions[0]
    sec_key, sec_prob = sorted_emotions[1]

    name, emoji, color = EMOTION_META.get(top_key, (top_key.capitalize(), '😐', '#6c757d'))
    sec_name, sec_emoji, _ = EMOTION_META.get(sec_key, (sec_key.capitalize(), '😐', '#6c757d'))

    formatted_scores = {
        EMOTION_META.get(k, (k.capitalize(), ''))[0] + " " + EMOTION_META.get(k, ('', '😐'))[1]: v 
        for k, v in scores.items()
    }

    return name, emoji, top_prob * 100, sec_name, sec_emoji, sec_prob * 100, formatted_scores, cropped_display

# Tabs
tab1, tab2 = st.tabs(["📸 Take Snapshot", "📁 Upload Image"])

def display_dashboard(name, emoji, conf, sec_name, sec_emoji, sec_conf, scores, crop_img):
    col_img, col_metrics = st.columns([1, 1])
    
    with col_img:
        st.image(crop_img, caption="Aligned Face Crop", use_container_width=True)
    
    with col_metrics:
        st.markdown(f"### Dominant Expression\n## {name} {emoji}")
        st.metric(label="Primary Confidence", value=f"{conf:.1f}%")
        
        if sec_conf > 15.0:
            st.caption(f"Secondary cue: **{sec_name} {sec_emoji}** ({sec_conf:.1f}%)")

    st.write("---")
    st.subheader("Confidence Distribution")
    for emo_label, prob in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        c1, c2 = st.columns([3, 7])
        c1.write(f"**{emo_label}**")
        c2.progress(min(prob, 1.0), text=f"{prob * 100:.1f}%")

with tab1:
    photo = st.camera_input("Snap a live photo:")
    if photo is not None:
        pil_image = Image.open(photo)
        name, emoji, conf, sec_name, sec_emoji, sec_conf, scores, crop_img = process_and_classify(pil_image)
        display_dashboard(name, emoji, conf, sec_name, sec_emoji, sec_conf, scores, crop_img)

with tab2:
    uploaded = st.file_uploader("Upload a face image (JPG, PNG):", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        pil_image = Image.open(uploaded)
        name, emoji, conf, sec_name, sec_emoji, sec_conf, scores, crop_img = process_and_classify(pil_image)
        display_dashboard(name, emoji, conf, sec_name, sec_emoji, sec_conf, scores, crop_img)
