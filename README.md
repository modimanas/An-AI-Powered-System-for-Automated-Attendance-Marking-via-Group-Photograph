<h1 align="center">📸 AI-Powered Automated Attendance System via Group Photograph</h1>

<p align="center">
  <b>Face Detection + Face Recognition + Automated Attendance</b><br>
  Built using YOLOv8, ArcFace ONNX, Streamlit, and MongoDB
</p>

---

# 🚀 Project Overview

This project provides a **complete pipeline** for automatically marking attendance from **group photographs**.

It consists of two main components:

### 1️⃣ **Student Registration (register.py)**
A guided **Streamlit-based interface**:
- Captures **6 face poses** using webcam  
- Runs **YOLOv8 face detection**  
- Generates embeddings using **ArcFace (glintr100.onnx)**  
- Creates a **master embedding** (mean of 6 poses)  
- Saves student details + embeddings to **MongoDB**

### 2️⃣ **Group Photo Attendance (detect.py)**
A high-performance script that:
- Detects all faces in a group photo using **YOLOv8**  
- Extracts embeddings for all faces in **batch mode**  
- Compares them with registered student embeddings using **Cosine Similarity**  
- Marks attendance for all matched students  
- Annotates the group image with names & confidence scores  



# 🧠 System Architecture

Registration → YOLOv8 → ArcFace Embeddings → MongoDB

↓

Group Photo Detection

↓

ArcFace Batch Embeddings

↓

Cosine Similarity Matching

↓

Automated Attendance


---


# 📂 Repository Structure
```marmaid
.
├── assets/                      # Optional: put banner.png or diagram images here
├── register.py                  # Streamlit student registration app
├── detect.py                    # Group photo attendance script
├── models/
│   ├── glintr100.onnx
│   └── yolov8m-face-lindevs.pt
├── requirements.txt
├── .env.example
├── .github/
│   └── workflows/
│       └── ci.yml               # optional CI for tests / lint
└── README.md
```
---

# 🧩 Models Used

| Component | Model | Description |
|----------|--------|-------------|
| **Face Detection** | `yolov8m-face-lindevs.pt` | YOLOv8 model optimized for face detection |
| **Face Embeddings** | `glintr100.onnx` | ArcFace ONNX 512-dim embedding model |

---

# 🔧 Installation
---

### 1️⃣ Clone the repository

```bash
git clone <https://github.com/modimanas/An-AI-Powered-System-for-Automated-Attendance-Marking-via-Group-Photograph>
cd An-AI-Powered-System-for-Automated-Attendance-Marking-via-Group-Photograph
```


### 2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Setup environment variables
```bash
Create .env using .env.example:

MONGO_URI="your-mongodb-uri"
DB_NAME="student_attendance_db"
COLLECTION_NAME="students"
```
---
## ▶️ How to Run
### 🧑‍🎓 1. Student Registration (Streamlit UI)

Run:
```bash
streamlit run register.py
```

The UI will:

-> Collect student details

-> Capture 6 guided face poses

-> Detect & verify each face

-> Generate embeddings

-> Save to MongoDB

```bash
Sample stored document:
{
  "student_id": "CSIT-A_2300290110146",
  "name": "Rohan",
  "roll_no": "2300290110146",
  "branch": "CSIT",
  "section": "B",
  "master_embeddings": [ ...512 floats... ]
}
```
### 👥 2. Attendance from Group Photo

Set image path in detect.py:
```bash
group_photo_path = "path/to/group_photo.jpg"
```

Run:
```bash
python detect.py
```

Outputs include:

-> Number of faces detected

-> Names + similarity scores

-> Annotated image with bounding boxes

-> Final attendance list:

Example:
```bash
Present Students:
 - 2300290110124 (Chodu CID)
 - 2300290110146 (Rohan)
```
---

## 🛢 MongoDB Schema

| Field            | Type         | Description                         |
|------------------|--------------|-------------------------------------|
| `student_id`     | `string`     | `<branch>-<section>_<rollno>`       |
| `name`           | `string`     | Student full name                   |
| `roll_no`        | `string`     | University roll number              |
| `branch`         | `string`     | CSIT / CSE                          |
| `section`        | `string`     | A / B / C / D                       |
| `master_embeddings` | `list[float]` | 512-dim ArcFace embedding        |

---

## ⚙️ Key Features

- High-accuracy **YOLOv8** face detection  
- **ArcFace (ONNX)** based 512-dim embeddings  
- **Batch inference** for speed and efficiency  
- Robust **cosine similarity**–based matching  
- Guided **multi-pose registration** (improves recognition)  
- Fully **automated attendance extraction** from group photos

---

## 🛠 Troubleshooting

| Issue                                 | Fix / Recommendation |
|---------------------------------------|----------------------|
| `cv2.imshow` fails (headless server)  | Use `cv2.imwrite("output.jpg", annotated_image)` to save output instead of displaying it. |
| WebRTC / camera errors (Streamlit)    | Install FFmpeg and system video libs: `sudo apt install ffmpeg` (Linux) or install appropriate OS packages. |
| No face detected during registration  | Ensure the captured pose contains **exactly one** clear, well-lit face and retry capture. |
| ONNX runtime errors                    | Check `onnxruntime` version and model opset compatibility; try upgrading `onnxruntime`. |
| Large memory usage for big batches     | Chunk face batch into smaller batches (e.g., 32 faces per ONNX call). |

---

## 🔐 Security Notes

- **Never commit** `.env` or any file containing credentials to source control.  
- Use **environment variables** (e.g., via `.env` + `python-dotenv`) for `MONGO_URI`, DB names, and other secrets.  
- Restrict MongoDB network access with **IP whitelisting**, strong passwords, and TLS.  
- Treat stored embeddings as **sensitive biometric data** — obtain user consent and follow applicable privacy laws.  
- Rotate credentials regularly and monitor DB access logs.

---

## 📈 Future Enhancements

- Add a web **attendance dashboard** (Streamlit / React) with history and filters.  
- Support **multi-embedding** per student and use max-per-student similarity to improve recognition.  
- Integrate **GPU ONNXExecutionProvider** (CUDA) for faster inference.  
- Provide **cloud deployment** scripts (Docker + CI/CD) for Streamlit + inference service.  
- Add a **QR-code verification** fallback for manual checks and hybrid attendance workflows.  

---

## 🤝 Contributing

Contributions are welcome — please follow these guidelines:

- Fork the repository and create a feature branch.  
- Keep PRs focused and include tests where possible.  
- **Do not** commit secrets or large model blobs (>100MB). Use links or instructions to obtain models.  
- Add clear documentation for new features and update `requirements.txt` if dependencies change.

---

## 📄 License

This project is released under the **MIT License**. See the `LICENSE` file for details.

---

## 👤 Author

**Manas Modi**


