import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import mediapipe as mp

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="😊", layout="centered")

st.title("Facial Emotion Recognition")
st.write("Calculates facial expressions using **468 3D Facial Landmark Action Units** (Eye Aspect Ratio, Brow Furrow, Lip Corner Metrics).")

# Setup MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

@st.cache_resource
def get_mesh_detector():
    return mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

mesh_detector = get_mesh_detector()

def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def analyze_facial_action_units(pil_img):
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    img_np = np.array(pil_img)
    h, w, _ = img_np.shape

    results = mesh_detector.process(img_np)
    if not results.multi_face_landmarks:
        return None, "No face detected in the image.", None

    landmarks = results.multi_face_landmarks[0].landmark
    pts = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks])

    # Key landmark indices (Standard 468 MediaPipe FaceMesh)
    # Eyes & Brows
    left_brow = pts[70]
    right_brow = pts[300]
    brow_mid = (left_brow + right_brow) / 2.0
    nose_bridge = pts[168]

    left_brow_inner = pts[107]
    right_brow_inner = pts[336]

    # Mouth
    lip_left = pts[61]
    lip_right = pts[291]
    lip_top = pts[13]
    lip_bottom = pts[14]

    # Chin & Nose
    chin = pts[152]
    nose_tip = pts[1]

    # 1. Normalization unit (Face Height & Interocular Distance)
    face_height = euclidean_dist(nose_bridge[:2], chin[:2]) + 1e-6
    interocular_dist = euclidean_dist(pts[33][:2], pts[263][:2]) + 1e-6

    # 2. Geometric Action Unit (AU) Ratios
    # AU12: Smile / Lip Corner Puller (Mouth width relative to interocular distance)
    mouth_width = euclidean_dist(lip_left[:2], lip_right[:2])
    mouth_width_ratio = mouth_width / interocular_dist

    # AU12 Y-Elevation: Lip corners vs center of mouth (smiling pulls corners UP)
    mouth_center_y = (lip_top[1] + lip_bottom[1]) / 2.0
    corner_avg_y = (lip_left[1] + lip_right[1]) / 2.0
    smile_elevation = (mouth_center_y - corner_avg_y) / face_height

    # AU25/AU26: Mouth Opening (Surprise / Laugh)
    mouth_open = euclidean_dist(lip_top[:2], lip_bottom[:2])
    mouth_open_ratio = mouth_open / face_height

    # AU4: Brow Lowerer / Furrow (Anger / Concentration)
    brow_to_nose = euclidean_dist(brow_mid[:2], nose_bridge[:2]) / face_height
    inner_brow_dist = euclidean_dist(left_brow_inner[:2], right_brow_inner[:2]) / interocular_dist

    # AU1/AU15: Brow Inner Raise & Lip Corner Depressor (Sadness)
    brow_raise = (nose_tip[1] - brow_mid[1]) / face_height
    frown_depression = (corner_avg_y - mouth_center_y) / face_height

    # 3. Dynamic Score Calculation based on Biometric AUs
    scores = {
        "Happy 😊": 0.0,
        "Surprise 😲": 0.0,
        "Sad 😢": 0.0,
        "Angry 😠": 0.0,
        "Neutral 😐": 0.20
    }

    # Smile evaluation
    if mouth_width_ratio > 0.92 or smile_elevation > 0.015:
        smile_intensity = max((mouth_width_ratio - 0.88) * 4.0, 0.0) + max(smile_elevation * 30.0, 0.0)
        scores["Happy 😊"] += smile_intensity * 1.5

    # Surprise evaluation
    if mouth_open_ratio > 0.08:
        scores["Surprise 😲"] += (mouth_open_ratio - 0.07) * 8.0

    # Sadness evaluation (Corners down, inner brow elevation)
    if frown_depression > 0.008 or (brow_raise > 0.42 and mouth_width_ratio < 0.85):
        scores["Sad 😢"] += max(frown_depression * 40.0, 0.2) + 0.3

    # Anger evaluation (Brows drawn down and together, tight mouth)
    if inner_brow_dist < 0.28 or brow_to_nose < 0.045:
        scores["Angry 😠"] += (0.30 - inner_brow_dist) * 5.0 + 0.4

    # Normalize to probabilities
    total_val = sum(scores.values()) + 1e-6
    probs = {k: v / total_val for k, v in scores.items()}

    # Annotate landmarks on image
    annotated_img = img_np.copy()
    for idx in [61, 291, 13, 14, 70, 300, 107, 336, 1, 152]:
        p = pts[idx]
        cv2.circle(annotated_img, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)

    top_emotion = max(probs, key=probs.get)
    return Image.fromarray(annotated_img), top_emotion, probs

# UI
tab1, tab2 = st.tabs(["📸 Take Snapshot", "📁 Upload Image"])

def render_dashboard(annotated_img, top_emotion, probs):
    if annotated_img is None:
        st.warning(top_emotion)
        return

    c1, c2 = st.columns([1, 1])
    with c1:
        st.image(annotated_img, caption="Landmark Action Unit Tracking", use_container_width=True)
    with c2:
        st.markdown(f"### Detected Emotion\n## **{top_emotion}**")
        st.metric("Confidence", f"{probs[top_emotion]*100:.1f}%")

    st.write("---")
    st.subheader("Action Unit Breakdown")
    for emo, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        col1, col2 = st.columns([3, 7])
        col1.write(f"**{emo}**")
        col2.progress(min(prob, 1.0), text=f"{prob*100:.1f}%")

with tab1:
    shot = st.camera_input("Take a snapshot:")
    if shot is not None:
        pil_img = Image.open(shot)
        annotated, top_emo, probs = analyze_facial_action_units(pil_img)
        render_dashboard(annotated, top_emo, probs)

with tab2:
    uploaded = st.file_uploader("Upload an image (JPG, PNG):", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        pil_img = Image.open(uploaded)
        annotated, top_emo, probs = analyze_facial_action_units(pil_img)
        render_dashboard(annotated, top_emo, probs)
