import streamlit as st

# Show Streamlit footer icon but disable the link
custom_footer = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}

/* Disable footer link but keep icon visible */
footer a {
    pointer-events: none;   /* disables click */
    text-decoration: none;
}
</style>
"""
st.markdown(custom_footer, unsafe_allow_html=True)

import cv2
import numpy as np
import os
import requests  # <-- Yeh import zaroori hai
import av  # <-- Yeh import streamlit_webrtc ke liye zaroori hai
from ultralytics import YOLO
import onnxruntime as ort
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from pymongo import MongoClient

# --- CONFIGURATION ---
MONGO_URI = "mongodb+srv://manasmodi603_db_user:YatfzxpDTUrF2IFR@cluster0.7gdy0eb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
YOLO_MODEL_PATH = "yolov8m-face-lindevs.pt"
ONNX_MODEL_FILENAME = "glintr100.onnx"
# Sahi direct download link
ONNX_MODEL_URL = "https://huggingface.co/FrancisRing/StableAnimator/resolve/main/models/antelopev2/glintr100.onnx"

# --- MONGODB CONNECTION ---
@st.cache_resource
def get_mongo_client():
    client = MongoClient(MONGO_URI)
    return client

try:
    client = get_mongo_client()
    db = client["student_attendance_db"]
    students_collection = db["students"]
except Exception as e:
    st.error(f"Failed to connect to MongoDB: {e}")
    st.stop()


# --- MODEL LOADING (Ek hi baar load hoga) ---
# --- MODEL LOADING (Ek hi baar load hoga) ---
@st.cache_resource
def load_models():
    """
    Downloads models from Hugging Face if not present, then loads them.
    """
    # Configuration for models
    YOLO_MODEL_FILENAME = "yolov8m-face-lindevs.pt"
    # Yahan aapka naya link daala hai
    YOLO_MODEL_URL = "https://huggingface.co/manas06/student-attendance-models/resolve/main/yolov8m-face-lindevs.pt" 

    ONNX_MODEL_FILENAME = "glintr100.onnx"
    ONNX_MODEL_URL = "https://huggingface.co/FrancisRing/StableAnimator/resolve/main/models/antelopev2/glintr100.onnx"

    # Helper function to download a file
    def download_file(url, filename):
        if not os.path.exists(filename):
            try:
                with st.spinner(f"Downloading {filename}... This may take a moment."):
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    with open(filename, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
            except Exception as e:
                st.error(f"Failed to download {filename}: {e}")
                return False
        return True

    # Dono models ko download karein
    if not download_file(YOLO_MODEL_URL, YOLO_MODEL_FILENAME): return None, None, None
    if not download_file(ONNX_MODEL_URL, ONNX_MODEL_FILENAME): return None, None, None

    # Saare models ko load karein
    try:
        with st.spinner("Loading AI models..."):
            yolo_detector = YOLO(YOLO_MODEL_FILENAME)
            arcface_session = ort.InferenceSession(ONNX_MODEL_FILENAME, providers=['CPUExecutionProvider'])
            haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        st.success("Models loaded successfully!")
        return yolo_detector, arcface_session, haar_cascade
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Is function ke neeche aapka baaki ka code waise ka waisa hi rahega...

# Models ko load karein
yolo_detector, arcface_session, haar_cascade = load_models()

# Agar model load na ho toh app rok dein
if not all([yolo_detector, arcface_session, haar_cascade]):
    st.error("A critical error occurred during model loading. The app cannot continue.")
    st.stop()


# --- HELPER FUNCTIONS ---
def generate_embedding_and_get_image(image):
    """YOLO se face detect karta hai, box banata hai, aur ArcFace se embedding generate karta hai."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = yolo_detector(img_rgb, verbose=False)

    if len(results[0].boxes) != 1:
        return None, None

    box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    image_with_box = img_rgb.copy()
    cv2.rectangle(image_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)

    face = img_rgb[y1:y2, x1:x2]

    # ArcFace pre-processing
    face = cv2.resize(face, (112, 112))
    face = face.astype(np.float32) / 255.0
    face = np.transpose(face, (2, 0, 1))
    input_tensor = np.expand_dims(face, axis=0)

    inputs = {arcface_session.get_inputs()[0].name: input_tensor}
    embedding = arcface_session.run(None, inputs)[0].flatten()
    return embedding.tolist(), image_with_box

# --- SESSION STATE INITIALIZATION ---
if 'stage' not in st.session_state:
    st.session_state.stage = "details"
    st.session_state.student_info = {}
    st.session_state.captured_images = []
    st.session_state.capture_instructions = [
        "Look STRAIGHT and click Capture",
        "Turn your face slightly LEFT and click Capture",
        "Turn your face slightly RIGHT and click Capture",
        "Look slightly UP and click Capture"
    ]

# --- STREAMLIT APP UI ---
st.title("🎓 Student Registration Portal")

# STAGE 1 & 2: Get Student Details
if st.session_state.stage == "details":
    st.header("Step 1: Enter Your Details")

    branch = st.selectbox("Select Branch", ["CSIT", "CSE"])
    section = st.selectbox("Select Section", ["A", "B", "C", "D"])
    name = st.text_input("Enter Student Name")
    roll_no = st.text_input("Enter University Roll Number")

    if st.button("Next: Capture Photos"):
        if name and roll_no and branch and section:
            st.session_state.student_info = {
                "name": name, "roll_no": roll_no, "branch": branch, "section": section
            }
            st.session_state.stage = "capture"
            st.rerun()
        else:
            st.warning("Please fill all details.")

# STAGE 3: Guided Photo Capture
elif st.session_state.stage == "capture":
    st.header("Step 2: Capture Your Photos")

    num_captured = len(st.session_state.captured_images)

    if num_captured < 4:
        st.info(f"Pose {num_captured + 1}/4: **{st.session_state.capture_instructions[num_captured]}**")

        captured_image = st.camera_input("Click here to capture the photo", key=f"photo_capture_{num_captured}")

        if captured_image is not None:
            file_bytes = np.asarray(bytearray(captured_image.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            st.session_state.captured_images.append(img)
            st.rerun()
    else:
        st.session_state.stage = "process"
        st.rerun()

# STAGE 4: Process Photos
elif st.session_state.stage == "process":
    st.header("Step 3: Processing Photos")
    st.success("All 4 photos captured successfully!")

    cols = st.columns(4)
    for i, image in enumerate(st.session_state.captured_images):
        with cols[i]:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Pose {i+1}", use_column_width=True)

    if st.button("Process Photos and Check Faces"):
        embeddings = []
        processed_images_with_boxes = []

        with st.spinner("Analyzing photos and generating embeddings..."):
            all_faces_detected = True
            for i, img in enumerate(st.session_state.captured_images):
                embedding, processed_image = generate_embedding_and_get_image(img)
                if embedding is None:
                    st.error(f"Could not detect a single face in Photo {i+1}. Please start over.")
                    all_faces_detected = False
                    break
                embeddings.append(embedding)
                processed_images_with_boxes.append(processed_image)
        
        if all_faces_detected:
            st.session_state.embeddings = embeddings
            st.session_state.processed_images = processed_images_with_boxes
            st.session_state.stage = "confirm"
            st.rerun()
        else:
            # Optionally reset to capture stage
            st.session_state.stage = "capture"
            st.session_state.captured_images = []


# STAGE 5: Confirm and Save
elif st.session_state.stage == "confirm":
    st.header("Step 4: Confirm and Save")
    st.info("Please check if the faces were detected correctly in your photos.")

    st.subheader("YOLO Face Detection Result:")
    cols_processed = st.columns(4)
    for i, image in enumerate(st.session_state.processed_images):
        with cols_processed[i]:
            st.image(image, caption=f"Detected Face {i+1}", use_column_width=True)

    st.write("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Looks Good! Save My Registration"):
            student_id = f"{st.session_state.student_info['branch']}-{st.session_state.student_info['section']}_{st.session_state.student_info['roll_no']}"
            student_doc = {
                "student_id": student_id,
                "name": st.session_state.student_info['name'],
                "roll_no": st.session_state.student_info['roll_no'],
                "branch": st.session_state.student_info['branch'],
                "section": st.session_state.student_info['section'],
                "embeddings": st.session_state.embeddings
            }
            try:
                with st.spinner("Saving data to database..."):
                    students_collection.update_one(
                        {"roll_no": st.session_state.student_info['roll_no']},
                        {"$set": student_doc},
                        upsert=True
                    )
                st.session_state.stage = "complete"
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred while saving to MongoDB: {e}")

    with col2:
        if st.button("❌ No, Start Over"):
            # Clear session state to reset the app
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# STAGE 6: Completion
elif st.session_state.stage == "complete":
    st.success(f"🎉 Registration successful for {st.session_state.student_info['name']}!")
    st.balloons()
    st.write("You can now close this window.")
    if st.button("Register Another Student"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
