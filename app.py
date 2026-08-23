import streamlit as st
from PIL import Image, ImageOps
from transformers import pipeline

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

classifier = load_deep_learning_pipeline()

def predict_emotion(pil_image):
    # Fix orientation and ensure RGB
    img = ImageOps.exif_transpose(pil_image).convert("RGB")
    
    # Run Vision Transformer forward pass
    raw_results = classifier(img)
    
    # Process scores
    scores = {item['label'].lower(): float(item['score']) for item in raw_results}
    
    top_label = raw_results[0]['label'].lower()
    name, emoji, color = EMOJI_DICT.get(top_label, (top_label.capitalize(), "😐", "#6c757d"))
    confidence = raw_results[0]['score'] * 100
    
    formatted_scores = {
        EMOJI_DICT.get(k, (k.capitalize(), ""))[0] + " " + EMOJI_DICT.get(k, ("", "😐"))[1]: v
        for k, v in scores.items()
    }
    
    return name, emoji, confidence, formatted_scores, img

tab1, tab2 = st.tabs(["📸 Take Snapshot", "📁 Upload Image"])

def display_dashboard(name, emoji, conf, scores, img):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(img, use_container_width=True)
    
    with col2:
        st.markdown(f"### Detected Expression\n## {name} {emoji}")
        st.metric(label="Model Confidence", value=f"{conf:.1f}%")

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
        name, emoji, conf, scores, img = predict_emotion(pil_img)
        display_dashboard(name, emoji, conf, scores, img)

with tab2:
    uploaded = st.file_uploader("Upload an image (JPG, PNG):", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        pil_img = Image.open(uploaded)
        name, emoji, conf, scores, img = predict_emotion(pil_img)
        display_dashboard(name, emoji, conf, scores, img)
