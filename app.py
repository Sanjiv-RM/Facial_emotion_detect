import base64
import os
import urllib.request
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Live Facial Emotion Recognition", page_icon="😊", layout="centered")

st.title("Live Facial Emotion Recognition")
st.write("Real-time facial expression detector running directly in your browser.")

EMOTION_MAP = {
    0: ("Neutral", "😐", "#A0AEC0"),
    1: ("Happy", "😊", "#38A169"),
    2: ("Surprised", "😲", "#D69E2E"),
    3: ("Sad", "😢", "#3182CE"),
    4: ("Angry", "😠", "#E53E3E"),
    5: ("Disgusted", "🤢", "#DD6B20"),
    6: ("Fearful", "😨", "#805AD5"),
    7: ("Contempt", "😏", "#718096")
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

def process_image(img_bgr):
    h_img, w_img = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.15, 
        minNeighbors=5, 
        minSize=(50, 50)
    )

    detected_faces = []

    for (x, y, w, h) in faces:
        # Add 15% margin around face so smiles and cheek expressions aren't cut off
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_img, x + w + margin_x)
        y2 = min(h_img, y + h + margin_y)

        face_roi = gray[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        # Contrast equalization to balance lighting
        face_roi = cv2.equalizeHist(face_roi)
        face_resized = cv2.resize(face_roi, (64, 64))

        # Normalize for ONNX model
        blob = cv2.dnn.blobFromImage(
            face_resized, 
            scalefactor=1.0 / 255.0, 
            size=(64, 64), 
            mean=(0.5,), 
            swapRB=False, 
            crop=False
        )

        net.setInput(blob)
        preds = net.forward()[0]
        
        # Softmax
        exp_preds = np.exp(preds - np.max(preds))
        probs = exp_preds / exp_preds.sum()

        # Reduce neutral bias to prevent false neutrals on smiles
        probs[0] *= 0.65
        probs = probs / probs.sum()

        top_idx = int(np.argmax(probs))
        name, emoji, _ = EMOTION_MAP[top_idx]
        confidence = probs[top_idx] * 100

        # Draw bounding box and label
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{name} {emoji} ({confidence:.0f}%)"
        cv2.putText(img_bgr, label, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

        detected_faces.append((name, emoji, confidence, {EMOTION_MAP[i][0] + " " + EMOTION_MAP[i][1]: float(probs[i]) for i in range(8)}))

    return img_bgr, detected_faces

# Sidebar Controls
tab1, tab2 = st.tabs(["🎥 Continuous Live Camera", "📁 Upload Image"])

with tab1:
    st.write("Grant camera access to start live emotion recognition:")
    
    # Client-side HTML5 camera widget that sends frames through standard HTTP
    camera_html = """
    <div style="display: flex; flex-direction: column; align-items: center;">
        <video id="video" width="100%" height="auto" autoplay playsinline style="max-width: 500px; border-radius: 8px; border: 2px solid #333;"></video>
        <canvas id="canvas" width="480" height="360" style="display:none;"></canvas>
        <div style="margin-top: 10px;">
            <button id="snap-btn" style="background-color: #ff4b4b; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; cursor: pointer;">Scan Expression</button>
            <button id="auto-btn" style="background-color: #2e7bcf; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; cursor: pointer; margin-left: 8px;">Auto-Scan: OFF</button>
        </div>
    </div>
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const snapBtn = document.getElementById('snap-btn');
        const autoBtn = document.getElementById('auto-btn');
        let autoInterval = null;

        navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 }, audio: false })
            .then(stream => { video.srcObject = stream; })
            .catch(err => { console.error("Camera Error:", err); });

        function captureAndSend() {
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataURL = canvas.toDataURL('image/jpeg', 0.7);
            window.parent.postMessage({
                type: "streamlit:setComponentValue",
                value: dataURL
            }, "*");
        }

        snapBtn.onclick = () => { captureAndSend(); };

        autoBtn.onclick = () => {
            if (autoInterval) {
                clearInterval(autoInterval);
                autoInterval = null;
                autoBtn.innerText = "Auto-Scan: OFF";
                autoBtn.style.backgroundColor = "#2e7bcf";
            } else {
                captureAndSend();
                autoInterval = setInterval(captureAndSend, 1200);
                autoBtn.innerText = "Auto-Scan: ON";
                autoBtn.style.backgroundColor = "#28a745";
            }
        };
    </script>
    """

    frame_data = components.html(camera_html, height=430)

    # Standard fallback camera widget
    st.write("---")
    st.markdown("**Instant Snapshot Alternative:**")
    cam_shot = st.camera_input("Take a photo:")
    if cam_shot is not None:
        img_bytes = np.asarray(bytearray(cam_shot.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        annotated_img, face_data = process_image(img_bgr)
        
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        if face_data:
            for idx, (name, emoji, conf, breakdown) in enumerate(face_data):
                st.success(f"**Face #{idx+1}:** {name} {emoji} (**{conf:.1f}%**)")
                with st.expander("Emotion Breakdown"):
                    for em, pr in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
                        col1, col2 = st.columns([3, 7])
                        col1.write(f"**{em}**")
                        col2.progress(min(pr, 1.0), text=f"{pr*100:.1f}%")

with tab2:
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG):", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated_img, face_data = process_image(img_bgr)
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        if face_data:
            for idx, (name, emoji, conf, breakdown) in enumerate(face_data):
                st.success(f"**Face #{idx+1}:** {name} {emoji} (**{conf:.1f}%**)")
                with st.expander("Emotion Breakdown"):
                    for em, pr in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
                        col1, col2 = st.columns([3, 7])
                        col1.write(f"**{em}**")
                        col2.progress(min(pr, 1.0), text=f"{pr*100:.1f}%")
