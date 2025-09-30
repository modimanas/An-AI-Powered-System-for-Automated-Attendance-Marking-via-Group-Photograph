# # # import streamlit as st
# # # import cv2
# # # import numpy as np
# # # import os
# # # import json
# # # from ultralytics import YOLO
# # # import onnxruntime as ort
# # # from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
# # # import threading
# # # import time

# # # # --- CONFIGURATION --
# # # DATABASE_FILE = "students_data.json"
# # # YOLO_MODEL_PATH = "yolov8m-face-lindevs.pt"
# # # ARCFACE_MODEL_PATH = "glintr100.onnx"

# # # # --- MODEL LOADING (Ek hi baar load hoga) ---
# # # @st.cache_resource
# # # def load_models():
# # #     try:
# # #         yolo_detector = YOLO(YOLO_MODEL_PATH)
# # #         arcface_session = ort.InferenceSession(ARCFACE_MODEL_PATH, providers=['CPUExecutionProvider'])
# # #         haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# # #         return yolo_detector, arcface_session, haar_cascade
# # #     except Exception as e:
# # #         st.error(f"Error loading models: {e}")
# # #         st.stop()

# # # yolo_detector, arcface_session, haar_cascade = load_models()

# # # # --- HELPER FUNCTIONS ---
# # # def generate_embedding(image):
# # #     """YOLO se face detect karke ArcFace se embedding generate karta hai."""
# # #     results = yolo_detector(image, verbose=False)
# # #     if len(results[0].boxes) != 1:
# # #         return None  # Agar ek face nahi mila to None return karo

# # #     box = results[0].boxes[0]
# # #     x1, y1, x2, y2 = map(int, box.xyxy[0])
# # #     face = image[y1:y2, x1:x2]
    
# # #     # ArcFace pre-processing
# # #     face = cv2.resize(face, (112, 112))
# # #     face = face.astype(np.float32) / 255.0
# # #     face = np.transpose(face, (2, 0, 1))
# # #     input_tensor = np.expand_dims(face, axis=0)
    
# # #     inputs = {arcface_session.get_inputs()[0].name: input_tensor}
# # #     embedding = arcface_session.run(None, inputs)[0].flatten()
# # #     return embedding.tolist()

# # # # --- VIDEO PROCESSOR FOR GUIDANCE ---
# # # class GuidedVideoProcessor(VideoTransformerBase):
# # #     def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
# # #         img = frame.to_ndarray(format="bgr24")
# # #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # #         faces = haar_cascade.detectMultiScale(gray, 1.1, 4)
        
# # #         h, w, _ = img.shape
# # #         center_x, center_y = w // 2, h // 2
# # #         oval_w, oval_h = w // 3, h // 2
        
# # #         border_color = (255, 255, 255) # White
# # #         if len(faces) == 1:
# # #             x, y, fw, fh = faces[0]
# # #             face_center_x = x + fw // 2
# # #             face_center_y = y + fh // 2
# # #             if (abs(face_center_x - center_x) < 30 and abs(face_center_y - center_y) < 40):
# # #                 border_color = (0, 255, 0) # Green
        
# # #         cv2.ellipse(img, (center_x, center_y), (oval_w, oval_h), 0, 0, 360, border_color, 3)
# # #         return av.VideoFrame.from_ndarray(img, format="bgr24")

# # # # --- SESSION STATE INITIALIZATION ---
# # # if 'stage' not in st.session_state:
# # #     st.session_state.stage = "details"
# # #     st.session_state.student_info = {}
# # #     st.session_state.captured_images = []
# # #     st.session_state.capture_instructions = [
# # #         "Look STRAIGHT and click Capture",
# # #         "Turn your face slightly LEFT and click Capture",
# # #         "Turn your face slightly RIGHT and click Capture",
# # #         "Look slightly UP and click Capture"
# # #     ]

# # # # --- STREAMLIT APP UI ---
# # # st.title("🎓 Student Registration Portal")

# # # # STAGE 1 & 2: Get Student Details
# # # if st.session_state.stage == "details":
# # #     st.header("Step 1: Enter Your Details")
    
# # #     branch = st.selectbox("Select Branch", ["CSE", "IT", "ECE", "MECH", "CS-AIML", "CS-DS"])
# # #     section = st.selectbox("Select Section", ["A", "B", "C", "D"])
# # #     name = st.text_input("Enter Student Name")
# # #     roll_no = st.text_input("Enter University Roll Number")

# # #     if st.button("Next: Capture Photos"):
# # #         if name and roll_no and branch and section:
# # #             st.session_state.student_info = {
# # #                 "name": name, "roll_no": roll_no, "branch": branch, "section": section
# # #             }
# # #             st.session_state.stage = "capture"
# # #             st.rerun()
# # #         else:
# # #             st.warning("Please fill all details.")

# # # # STAGE 3: Guided Photo Capture
# # # elif st.session_state.stage == "capture":
# # #     st.header("Step 2: Capture Your Photos")
    
# # #     num_captured = len(st.session_state.captured_images)
    
# # #     if num_captured < 4:
# # #         st.info(f"Pose {num_captured + 1}/4: **{st.session_state.capture_instructions[num_captured]}**")
        
# # #         webrtc_streamer(
# # #             key="face-capture",
# # #             video_processor_factory=GuidedVideoProcessor,
# # #             media_stream_constraints={"video": True, "audio": False},
# # #             async_processing=True,
# # #         )
        
# # #         captured_image = st.camera_input("Click here to capture the photo")
        
# # #         if captured_image is not None:
# # #             file_bytes = np.asarray(bytearray(captured_image.read()), dtype=np.uint8)
# # #             img = cv2.imdecode(file_bytes, 1)
# # #             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert for correct display
# # #             st.session_state.captured_images.append(img_rgb)
# # #             st.rerun()
# # #     else:
# # #         st.session_state.stage = "process"
# # #         st.rerun()

# # # # STAGE 4: Process and Save
# # # elif st.session_state.stage == "process":
# # #     st.header("Step 3: Confirm and Save")
# # #     st.success("All 4 photos captured successfully!")
    
# # #     cols = st.columns(4)
# # #     for i, image in enumerate(st.session_state.captured_images):
# # #         with cols[i]:
# # #             st.image(image, caption=f"Pose {i+1}", use_column_width=True)
            
# # #     if st.button("✅ Confirm and Save My Registration"):
# # #         embeddings = []
# # #         with st.spinner("Analyzing photos and generating embeddings... This may take a moment."):
# # #             for img in st.session_state.captured_images:
# # #                 embedding = generate_embedding(img)
# # #                 if embedding is None:
# # #                     st.error(f"Could not detect a single face in one of the photos. Please start over.")
# # #                     st.stop()
# # #                 embeddings.append(embedding)

# # #         # Save data to JSON file
# # #         student_id = f"{st.session_state.student_info['branch']}-{st.session_state.student_info['section']}_{st.session_state.student_info['roll_no']}"
        
# # #         db = {}
# # #         # FIX: Check if file exists AND is not empty before trying to load it
# # #         if os.path.exists(DATABASE_FILE) and os.path.getsize(DATABASE_FILE) > 0:
# # #             with open(DATABASE_FILE, 'r') as f:
# # #                 db = json.load(f)

# # #         db[student_id] = {
# # #             "name": st.session_state.student_info['name'],
# # #             "roll_no": st.session_state.student_info['roll_no'],
# # #             "branch": st.session_state.student_info['branch'],
# # #             "section": st.session_state.student_info['section'],
# # #             "embeddings": embeddings
# # #         }

# # #         with open(DATABASE_FILE, 'w') as f:
# # #             json.dump(db, f, indent=4)
        
# # #         st.session_state.stage = "complete"
# # #         st.rerun()

# # # # STAGE 5: Completion
# # # elif st.session_state.stage == "complete":
# # #     st.success(f"🎉 Registration successful for {st.session_state.student_info['name']}!")
# # #     st.balloons()
# # #     st.write("You can now close this window.")

# # # # Add a button to reset the process anytime
# # # if st.button("Start Over"):
# # #     for key in st.session_state.keys():
# # #         del st.session_state[key]
# # #     st.rerun()


# # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
# # import streamlit as st
# # import cv2
# # import numpy as np
# # import os
# # import json
# # from ultralytics import YOLO
# # import onnxruntime as ort
# # from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
# # import threading
# # import time

# # # --- CONFIGURATION --
# # DATABASE_FILE = "students_data.json"
# # YOLO_MODEL_PATH = "yolov8m-face-lindevs.pt"
# # ARCFACE_MODEL_PATH = "glintr100.onnx"

# # # --- MODEL LOADING (Ek hi baar load hoga) ---
# # @st.cache_resource
# # def load_models():
# #     try:
# #         yolo_detector = YOLO(YOLO_MODEL_PATH)
# #         arcface_session = ort.InferenceSession(ARCFACE_MODEL_PATH, providers=['CPUExecutionProvider'])
# #         haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# #         return yolo_detector, arcface_session, haar_cascade
# #     except Exception as e:
# #         st.error(f"Error loading models: {e}")
# #         st.stop()

# # yolo_detector, arcface_session, haar_cascade = load_models()

# # # --- HELPER FUNCTIONS ---
# # # MODIFIED: Ab yeh function embedding ke saath-saath box waali image bhi return karega
# # def generate_embedding_and_get_image(image):
# #     """YOLO se face detect karta hai, box banata hai, aur ArcFace se embedding generate karta hai."""
# #     # Convert BGR (from OpenCV) to RGB for YOLO model
# #     img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #     results = yolo_detector(img_rgb, verbose=False)
    
# #     if len(results[0].boxes) != 1:
# #         return None, None  # Agar ek face nahi mila to None return karo

# #     box = results[0].boxes[0]
# #     x1, y1, x2, y2 = map(int, box.xyxy[0])
    
# #     # --- YAHAN NAYA CHANGE HAI ---
# #     # Original RGB image par box draw karo
# #     image_with_box = img_rgb.copy()
# #     cv2.rectangle(image_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #     # --- YAHAN TAK ---
    
# #     face = img_rgb[y1:y2, x1:x2]
    
# #     # ArcFace pre-processing
# #     face = cv2.resize(face, (112, 112))
# #     face = (face.astype(np.float32) - 127.5) / 128.0    
# #     face = np.transpose(face, (2, 0, 1)) # it ensures model receive output in correct format ie(channel,height , width ) here in this case 
# #     input_tensor = np.expand_dims(face, axis=0)
    
# #     inputs = {arcface_session.get_inputs()[0].name: input_tensor}
# #     embedding = arcface_session.run(None, inputs)[0].flatten()
# #     return embedding.tolist(), image_with_box

# # # --- VIDEO PROCESSOR FOR GUIDANCE ---
# # class GuidedVideoProcessor(VideoTransformerBase):
# #     def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
# #         img = frame.to_ndarray(format="bgr24")
# #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #         faces = haar_cascade.detectMultiScale(gray, 1.1, 4)
        
# #         h, w, _ = img.shape
# #         center_x, center_y = w // 2, h // 2
# #         oval_w, oval_h = w // 3, h // 2
        
# #         border_color = (255, 255, 255) # White
# #         if len(faces) == 1:
# #             x, y, fw, fh = faces[0]
# #             face_center_x = x + fw // 2
# #             face_center_y = y + fh // 2
# #             if (abs(face_center_x - center_x) < 30 and abs(face_center_y - center_y) < 40):
# #                 border_color = (0, 255, 0) # Green
        
# #         cv2.ellipse(img, (center_x, center_y), (oval_w, oval_h), 0, 0, 360, border_color, 3)
# #         return av.VideoFrame.from_ndarray(img, format="bgr24")

# # # --- SESSION STATE INITIALIZATION ---
# # if 'stage' not in st.session_state:
# #     st.session_state.stage = "details"
# #     st.session_state.student_info = {}
# #     st.session_state.captured_images = []
# #     st.session_state.capture_instructions = [
# #         "Look STRAIGHT and click Capture",
# #         "Turn your face slightly LEFT and click Capture",
# #         "Turn your face slightly RIGHT and click Capture",
# #         "Look slightly UP and click Capture"
# #     ]

# # # --- STREAMLIT APP UI ---
# # st.title("🎓 Student Registration Portal")

# # # STAGE 1 & 2: Get Student Details
# # if st.session_state.stage == "details":
# #     st.header("Step 1: Enter Your Details")
    
# #     branch = st.selectbox("Select Branch", ["CSE", "IT", "ECE", "MECH", "CS-AIML", "CS-DS"])
# #     section = st.selectbox("Select Section", ["A", "B", "C", "D"])
# #     name = st.text_input("Enter Student Name")
# #     roll_no = st.text_input("Enter University Roll Number")

# #     if st.button("Next: Capture Photos"):
# #         if name and roll_no and branch and section:
# #             st.session_state.student_info = {
# #                 "name": name, "roll_no": roll_no, "branch": branch, "section": section
# #             }
# #             st.session_state.stage = "capture"
# #             st.rerun()
# #         else:
# #             st.warning("Please fill all details.")

# # # STAGE 3: Guided Photo Capture
# # elif st.session_state.stage == "capture":
# #     st.header("Step 2: Capture Your Photos")
    
# #     num_captured = len(st.session_state.captured_images)
    
# #     if num_captured < 4:
# #         st.info(f"Pose {num_captured + 1}/4: **{st.session_state.capture_instructions[num_captured]}**")
        
# #         captured_image = st.camera_input("Click here to capture the photo", key=f"photo_capture_{num_captured}")
        
# #         if captured_image is not None:
# #             file_bytes = np.asarray(bytearray(captured_image.read()), dtype=np.uint8)
# #             img = cv2.imdecode(file_bytes, 1)
# #             st.session_state.captured_images.append(img)
# #             st.rerun()
# #     else:
# #         st.session_state.stage = "process"
# #         st.rerun()

# # # STAGE 4: Process Photos
# # elif st.session_state.stage == "process":
# #     st.header("Step 3: Processing Photos")
# #     st.success("All 4 photos captured successfully!")
    
# #     cols = st.columns(4)
# #     for i, image in enumerate(st.session_state.captured_images):
# #         with cols[i]:
# #             st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Pose {i+1}", use_column_width=True)
            
# #     if st.button("Process Photos and Check Faces"):
# #         embeddings = []
# #         processed_images_with_boxes = []
        
# #         with st.spinner("Analyzing photos and generating embeddings..."):
# #             for img in st.session_state.captured_images:
# #                 embedding, processed_image = generate_embedding_and_get_image(img)
# #                 if embedding is None:
# #                     st.error(f"Could not detect a single face in one of the photos. Please start over.")
# #                     st.stop()
# #                 embeddings.append(embedding)
# #                 processed_images_with_boxes.append(processed_image)
        
# #         st.session_state.embeddings = embeddings
# #         st.session_state.processed_images = processed_images_with_boxes
# #         st.session_state.stage = "confirm"
# #         st.rerun()

# # # STAGE 5: Confirm and Save
# # elif st.session_state.stage == "confirm":
# #     st.header("Step 4: Confirm and Save")
# #     st.info("Please check if the faces were detected correctly in your photos.")
    
# #     st.subheader("YOLO Face Detection Result:")
# #     cols_processed = st.columns(4)
# #     for i, image in enumerate(st.session_state.processed_images):
# #         with cols_processed[i]:
# #             st.image(image, caption=f"Detected Face {i+1}", use_column_width=True)
    
# #     st.write("---")
    
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         if st.button("✅ Looks Good! Save My Registration"):
# #             student_id = f"{st.session_state.student_info['branch']}-{st.session_state.student_info['section']}_{st.session_state.student_info['roll_no']}"
# #             db = {}
# #             if os.path.exists(DATABASE_FILE) and os.path.getsize(DATABASE_FILE) > 0:
# #                 with open(DATABASE_FILE, 'r') as f:
# #                     db = json.load(f)
# #             db[student_id] = {
# #                 "name": st.session_state.student_info['name'],
# #                 "roll_no": st.session_state.student_info['roll_no'],
# #                 "branch": st.session_state.student_info['branch'],
# #                 "section": st.session_state.student_info['section'],
# #                 "embeddings": st.session_state.embeddings
# #             }
# #             with open(DATABASE_FILE, 'w') as f:
# #                 json.dump(db, f, indent=4)
# #             st.session_state.stage = "complete"
# #             st.rerun()

# #     with col2:
# #         if st.button("❌ No, Start Over"):
# #             for key in list(st.session_state.keys()):
# #                 del st.session_state[key]
# #             st.rerun()

# # # STAGE 6: Completion
# # elif st.session_state.stage == "complete":
# #     st.success(f"🎉 Registration successful for {st.session_state.student_info['name']}!")
# #     st.balloons()
# #     st.write("You can now close this window.")
# #     if st.button("Register Another Student"):
# #         for key in list(st.session_state.keys()):
# #             del st.session_state[key]
# #         st.rerun()

# # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# import streamlit as st
# import cv2
# import numpy as np
# import os
# import json
# from ultralytics import YOLO
# import onnxruntime as ort
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
# import threading
# import time
# import sqlite3 

# # --- CONFIGURATION --
# DATABASE_FILE = "students_data.db" # Database file ka naam .db hai
# YOLO_MODEL_PATH = "yolov8m-face-lindevs.pt"
# ARCFACE_MODEL_PATH = "glintr100.onnx"

# # --- DATABASE INITIALIZATION ---
# # Yeh function database aur table banayega agar woh exist nahi karte.
# def init_db():
#     conn = sqlite3.connect(DATABASE_FILE)
#     c = conn.cursor()
#     # Unique Roll No check add kiya
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS students (
#             student_id TEXT PRIMARY KEY,
#             name TEXT,
#             roll_no TEXT UNIQUE,
#             branch TEXT,
#             section TEXT,
#             embeddings TEXT
#         )
#     ''')
#     conn.commit()
#     # conn.close()

# # App shuru hone par database ko initialize karo
# init_db()

# # --- MODEL LOADING (Ek hi baar load hoga) ---
# @st.cache_resource
# def load_models():
#     try:
#         yolo_detector = YOLO(YOLO_MODEL_PATH)
#         arcface_session = ort.InferenceSession(ARCFACE_MODEL_PATH, providers=['CPUExecutionProvider'])
#         haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         return yolo_detector, arcface_session, haar_cascade
#     except Exception as e:
#         st.error(f"Error loading models: {e}")
#         st.stop()

# yolo_detector, arcface_session, haar_cascade = load_models()

# # --- HELPER FUNCTIONS ---
# def generate_embedding_and_get_image(image):
#     """YOLO se face detect karta hai, box banata hai, aur ArcFace se embedding generate karta hai."""
#     # Convert BGR (from OpenCV) to RGB for YOLO model
#     img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = yolo_detector(img_rgb, verbose=False)
    
#     if len(results[0].boxes) != 1:
#         return None, None  # Agar ek face nahi mila to None return karo

#     box = results[0].boxes[0]
#     x1, y1, x2, y2 = map(int, box.xyxy[0])
    
#     image_with_box = img_rgb.copy()
#     cv2.rectangle(image_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
#     face = img_rgb[y1:y2, x1:x2]
    
#     # ArcFace pre-processing
#     face = cv2.resize(face, (112, 112))
#     face = face.astype(np.float32) / 255.0
#     face = np.transpose(face, (2, 0, 1))
#     input_tensor = np.expand_dims(face, axis=0)
    
#     inputs = {arcface_session.get_inputs()[0].name: input_tensor}
#     embedding = arcface_session.run(None, inputs)[0].flatten()
#     return embedding.tolist(), image_with_box

# # --- VIDEO PROCESSOR FOR GUIDANCE ---
# class GuidedVideoProcessor(VideoTransformerBase):
#     def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
#         img = frame.to_ndarray(format="bgr24")
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = haar_cascade.detectMultiScale(gray, 1.1, 4)
        
#         h, w, _ = img.shape
#         center_x, center_y = w // 2, h // 2
#         oval_w, oval_h = w // 3, h // 2
        
#         border_color = (255, 255, 255) # White
#         if len(faces) == 1:
#             x, y, fw, fh = faces[0]
#             face_center_x = x + fw // 2
#             face_center_y = y + fh // 2
#             if (abs(face_center_x - center_x) < 30 and abs(face_center_y - center_y) < 40):
#                 border_color = (0, 255, 0) # Green
        
#         cv2.ellipse(img, (center_x, center_y), (oval_w, oval_h), 0, 0, 360, border_color, 3)
#         return av.VideoFrame.from_ndarray(img, format="bgr24")

# # --- SESSION STATE INITIALIZATION ---
# if 'stage' not in st.session_state:
#     st.session_state.stage = "details"
#     st.session_state.student_info = {}
#     st.session_state.captured_images = []
#     st.session_state.capture_instructions = [
#         "Look STRAIGHT and click Capture",
#         "Turn your face slightly LEFT and click Capture",
#         "Turn your face slightly RIGHT and click Capture",
#         "Look slightly UP and click Capture"
#     ]

# # --- STREAMLIT APP UI ---
# st.title("🎓 Student Registration Portal")

# # STAGE 1 & 2: Get Student Details
# if st.session_state.stage == "details":
#     st.header("Step 1: Enter Your Details")
    
#     branch = st.selectbox("Select Branch", ["CSIT","CSE"])
#     section = st.selectbox("Select Section", ["A", "B", "C", "D"])
#     name = st.text_input("Enter Student Name")
#     roll_no = st.text_input("Enter University Roll Number")

#     if st.button("Next: Capture Photos"):
#         if name and roll_no and branch and section:
#             st.session_state.student_info = {
#                 "name": name, "roll_no": roll_no, "branch": branch, "section": section
#             }
#             st.session_state.stage = "capture"
#             st.rerun()
#         else:
#             st.warning("Please fill all details.")

# # STAGE 3: Guided Photo Capture
# elif st.session_state.stage == "capture":
#     st.header("Step 2: Capture Your Photos")
    
#     num_captured = len(st.session_state.captured_images)
    
#     if num_captured < 4:
#         st.info(f"Pose {num_captured + 1}/4: **{st.session_state.capture_instructions[num_captured]}**")
        
#         captured_image = st.camera_input("Click here to capture the photo", key=f"photo_capture_{num_captured}")
        
#         if captured_image is not None:
#             file_bytes = np.asarray(bytearray(captured_image.read()), dtype=np.uint8)
#             img = cv2.imdecode(file_bytes, 1)
#             st.session_state.captured_images.append(img)
#             st.rerun()
#     else:
#         st.session_state.stage = "process"
#         st.rerun()

# # STAGE 4: Process Photos
# elif st.session_state.stage == "process":
#     st.header("Step 3: Processing Photos")
#     st.success("All 4 photos captured successfully!")
    
#     cols = st.columns(4)
#     for i, image in enumerate(st.session_state.captured_images):
#         with cols[i]:
#             st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Pose {i+1}", use_column_width=True)
            
#     if st.button("Process Photos and Check Faces"):
#         embeddings = []
#         processed_images_with_boxes = []
        
#         with st.spinner("Analyzing photos and generating embeddings..."):
#             for img in st.session_state.captured_images:
#                 embedding, processed_image = generate_embedding_and_get_image(img)
#                 if embedding is None:
#                     st.error(f"Could not detect a single face in one of the photos. Please start over.")
#                     st.stop()
#                 embeddings.append(embedding)
#                 processed_images_with_boxes.append(processed_image)
        
#         st.session_state.embeddings = embeddings
#         st.session_state.processed_images = processed_images_with_boxes
#         st.session_state.stage = "confirm"
#         st.rerun()

# # STAGE 5: Confirm and Save
# elif st.session_state.stage == "confirm":
#     st.header("Step 4: Confirm and Save")
#     st.info("Please check if the faces were detected correctly in your photos.")
    
#     st.subheader("YOLO Face Detection Result:")
#     cols_processed = st.columns(4)
#     for i, image in enumerate(st.session_state.processed_images):
#         with cols_processed[i]:
#             st.image(image, caption=f"Detected Face {i+1}", use_column_width=True)
    
#     st.write("---")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("✅ Looks Good! Save My Registration"):
#             # --- DATABASE SAVE LOGIC (UPDATED FOR SQLITE) ---
#             student_id = f"{st.session_state.student_info['branch']}-{st.session_state.student_info['section']}_{st.session_state.student_info['roll_no']}"
            
#             # Embeddings ko JSON string mein convert karo
#             embeddings_json = json.dumps(st.session_state.embeddings)
            
#             try:
#                 # Database se connect karke data save karo
#                 conn = sqlite3.connect(DATABASE_FILE)
#                 c = conn.cursor()
#                 c.execute('''
#                     INSERT OR REPLACE INTO students (student_id, name, roll_no, branch, section, embeddings)
#                     VALUES (?, ?, ?, ?, ?, ?)
#                 ''', (
#                     student_id,
#                     st.session_state.student_info['name'],
#                     st.session_state.student_info['roll_no'],
#                     st.session_state.student_info['branch'],
#                     st.session_state.student_info['section'],
#                     embeddings_json
#                 ))
#                 conn.commit()
#                 conn.close()
#                 st.session_state.stage = "complete"
#                 st.rerun()
#             except sqlite3.IntegrityError:
#                 st.error(f"Error: Roll number '{st.session_state.student_info['roll_no']}' is already registered.")
#             except Exception as e:
#                 st.error(f"An error occurred while saving to the database: {e}")

#     with col2:
#         if st.button("❌ No, Start Over"):
#             for key in list(st.session_state.keys()):
#                 del st.session_state[key]
#             st.rerun()

# # STAGE 6: Completion
# elif st.session_state.stage == "complete":
#     st.success(f"🎉 Registration successful for {st.session_state.student_info['name']}!")
#     st.balloons()
#     st.write("You can now close this window.")
#     if st.button("Register Another Student"):
#         for key in list(st.session_state.keys()):
#             del st.session_state[key]
#         st.rerun()


import streamlit as st
import cv2
import numpy as np
import os
import json
from ultralytics import YOLO
import onnxruntime as ort
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import threading
import time
from pymongo import MongoClient

# --- CONFIGURATION ---
MONGO_URI = "mongodb+srv://manasmodi603_db_user:YatfzxpDTUrF2IFR@cluster0.7gdy0eb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"  # <-- Replace with your Mongo URI
YOLO_MODEL_PATH = "yolov8m-face-lindevs.pt"
ARCFACE_MODEL_PATH = "glintr100.onnx"

# --- MONGODB CONNECTION ---
client = MongoClient(MONGO_URI)
db = client["student_attendance_db"]
students_collection = db["students"]

# --- MODEL LOADING (Ek hi baar load hoga) ---
@st.cache_resource
def load_models():
    try:
        yolo_detector = YOLO(YOLO_MODEL_PATH)
        arcface_session = ort.InferenceSession(ARCFACE_MODEL_PATH, providers=['CPUExecutionProvider'])
        haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return yolo_detector, arcface_session, haar_cascade
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

yolo_detector, arcface_session, haar_cascade = load_models()

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

# --- VIDEO PROCESSOR FOR GUIDANCE ---
class GuidedVideoProcessor(VideoTransformerBase):
    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, 1.1, 4)
        
        h, w, _ = img.shape
        center_x, center_y = w // 2, h // 2
        oval_w, oval_h = w // 3, h // 2
        
        border_color = (255, 255, 255) # White
        if len(faces) == 1:
            x, y, fw, fh = faces[0]
            face_center_x = x + fw // 2
            face_center_y = y + fh // 2
            if (abs(face_center_x - center_x) < 30 and abs(face_center_y - center_y) < 40):
                border_color = (0, 255, 0) # Green
        
        cv2.ellipse(img, (center_x, center_y), (oval_w, oval_h), 0, 0, 360, border_color, 3)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

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
            for img in st.session_state.captured_images:
                embedding, processed_image = generate_embedding_and_get_image(img)
                if embedding is None:
                    st.error(f"Could not detect a single face in one of the photos. Please start over.")
                    st.stop()
                embeddings.append(embedding)
                processed_images_with_boxes.append(processed_image)
        
        st.session_state.embeddings = embeddings
        st.session_state.processed_images = processed_images_with_boxes
        st.session_state.stage = "confirm"
        st.rerun()

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
