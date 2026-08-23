import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from fer import FER

st.set_page_config(page_title="Facial Emotion Detector", page_icon="😊", layout="centered")

st.title("Facial Emotion Recognition")
st.write("Detect real-time facial expressions directly from your webcam.")

# Initialize FER emotion detector
detector = FER(mtcnn=False)

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Detect emotion and bounding boxes
        results = detector.detect_emotions(img)

        for result in results:
            (x, y, w, h) = result["box"]
            emotions = result["emotions"]
            
            # Find the dominant emotion
            dominant_emotion = max(emotions, key=emotions.get)
            score = emotions[dominant_emotion] * 100

            label = f"{dominant_emotion.capitalize()} ({score:.1f}%)"

            # Draw box and emotion tag
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img

# WebRTC Streamer for live camera
webrtc_streamer(
    key="emotion-detection",
    video_transformer_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
