# Student Emotion Detection System

## Introduction
Student Emotion Detection is a system that identifies the emotional state of students using facial expression analysis and machine learning techniques. The system analyzes facial features captured through a camera and classifies emotions such as happy, sad, angry, surprised, neutral, fear, and disgust.

In modern digital learning environments, understanding student emotions is important for improving learning engagement and teaching effectiveness. Emotion detection helps educators identify whether students are attentive, confused, or bored during lectures.

This system uses computer vision and deep learning algorithms to automatically detect emotions from facial expressions. By analyzing student emotions in real time, the system can help teachers improve teaching methods and provide better learning experiences.

## Literature Review
| SL NO | Paper Details | Dataset | Methodology Used | Gaps Identified |
|---|---|---|---|---|
| 1 | Zhang, K. et al. (2016). “Face Detection using Multi-task CNN.” | WIDER FACE | CNN used for detecting faces before emotion analysis | Focus mainly on face detection, not emotion recognition |
| 2 | Li, S., Deng, W., & Du, J. (2017). “Reliable Facial Expression Recognition.” | FER Dataset | Deep learning model for emotion classification | Performance decreases when the face is partially covered |
| 3 | Sariyanidi, E., et al. (2015). “Automatic Analysis of Facial Affect.” | Multiple datasets | Machine learning techniques for emotion analysis | Accuracy affected by lighting conditions and head movement |

## Problem Statement
In traditional classrooms and online learning environments, it is difficult for teachers to understand the emotional state of every student.

Some challenges include:
- Teachers cannot monitor each student's attention level
- Students may feel confused or bored without expressing it
- Online learning reduces face-to-face interaction

Therefore, there is a need for an automated system that can detect student emotions in real time using facial expressions to help teachers understand student engagement and improve teaching effectiveness.

## Feasibility Analysis and Expected Outcomes
### Technical Feasibility
The system can be implemented using existing technologies such as Python, OpenCV, and deep learning frameworks. These tools are freely available and easy to integrate.

### Economic Feasibility
The project requires minimal cost because most software tools are open-source.

### Operational Feasibility
The system can easily run on a computer with a webcam and does not require special hardware.

### Expected Outcomes
- Real-time detection of student emotions
- Improved understanding of student engagement
- Better interaction between students and teachers
- Data insights about student learning behavior

## Technology and Tool Identification
### Programming Language
- Python

### Libraries and Frameworks
- OpenCV (face detection)
- TensorFlow (deep learning)
- NumPy
- FER library

### Dataset
- FER2013 dataset

### Hardware Requirements
- Webcam
- Computer with GPU/CPU

### Software Requirements
- Python environment
- VS Code
- OpenCV library

## Project Structure
- `main.py` (runs the application)
- `face_detection.py` (handles face detection)
- `emotion_model.py` (handles emotion prediction)

## How to Run
1. Open a terminal in the project folder:
   - `cd path/to/Emotionn-detection-System`
2. (Recommended) Create and activate a virtual environment:
   - Linux/macOS:
     - `python3 -m venv .venv`
     - `source .venv/bin/activate`
   - Windows (PowerShell):
     - `python -m venv .venv`
     - `.venv\Scripts\Activate.ps1`
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Start the real-time system:
   - `python main.py`
5. If your webcam is not detected, try another camera index:
   - `python main.py --camera-index 1`
6. Press `q` to close the webcam window.

### Output
- The first label above the face is the **top emotion** with confidence.
- The second label shows **all emotion scores** (`happy`, `sad`, `angry`, `surprised`, `fear`, `disgust`, `neutral`).

## Conclusion
Student Emotion Detection systems play an important role in improving modern education by analyzing student engagement and emotional responses during learning sessions.

By using computer vision and machine learning techniques, the system can automatically detect student emotions from facial expressions. This technology helps educators understand student behavior, improve teaching methods, and enhance the overall learning experience.

With further improvements, emotion detection systems can be integrated into online learning platforms, smart classrooms, and educational analytics systems.

## Web Dashboard (Teacher Login + PostgreSQL)

This project includes a Flask dashboard where a **teacher logs in**, uploads a **student image or video**, and views stored emotion analysis history.

### 1) Configure PostgreSQL
Set a PostgreSQL connection URL:

- Linux/macOS:
  - `export DATABASE_URL='postgresql+psycopg2://postgres:postgres@localhost:5432/emotion_detection'`
- Windows (PowerShell):
  - `$env:DATABASE_URL='postgresql+psycopg2://postgres:postgres@localhost:5432/emotion_detection'`

### 2) Teacher credentials
Set teacher credentials with environment variables before starting the dashboard:
- `TEACHER_USERNAME`
- `TEACHER_PASSWORD`
- `SECRET_KEY`

If `TEACHER_USERNAME` is not provided, it defaults to `teacher`.

### 3) Run dashboard
- `python app.py`
- Open `http://127.0.0.1:5000`

### 4) Teacher workflow
1. Login using teacher credentials.
2. Upload a student **image** (`jpg`, `jpeg`, `png`, `bmp`) or **video** (`mp4`, `avi`, `mov`, `mkv`).
3. View dominant emotion, confidence, and full emotion scores on the dashboard history.

---

## Real-Time Classroom Monitoring System (FastAPI + WebSocket)

A second, fully independent system provides **live** per-student emotion and
attendance monitoring with intelligent alerting.

### Architecture

| File | Role |
|------|------|
| `classroom_backend.py` | FastAPI server — enrolment, attendance state, alerts, WebSocket broadcast |
| `video_processor.py` | Camera capture, face recognition, emotion detection, head-pose estimation |
| `static/classroom_dashboard.html` | Teacher dashboard (served at `/`) |

### Features

- **Face recognition** — identifies each enrolled student in the live feed.
- **Attendance tracking** — marks students Present on first detection; marks
  them Away if unseen for **5 minutes**; reverts to Present when they return.
- **Real-time emotion analysis** — per-student emotion streamed to the
  dashboard via WebSocket.
- **Head-pose estimation** — detects whether a student is looking left, right,
  down, or forward using `solvePnP`.
- **Intelligent alerts**
  - 🚨 **Emotional Distress** — triggered when a student shows `sad` or
    `angry` continuously for **10 minutes**.
  - ⚠️ **Low Engagement** — triggered when a student looks away from the
    camera AND shows `neutral` emotion for **2 minutes**.
- **Enrolment UI** — teachers capture webcam frames directly from the
  dashboard to register new students; no command-line steps required.

### Quick start

#### 1. Install dependencies (requires cmake for dlib/face-recognition)
```bash
pip install -r requirements.txt
```

#### 2. Start the backend
```bash
python classroom_backend.py
# Listening on http://0.0.0.0:8000
```

#### 3. Open the dashboard
Navigate to `http://localhost:8000` in your browser.

#### 4. Enrol students
1. Type a student's name in the **Enrol New Student** panel.
2. Click **Open Camera** and position the student's face.
3. Click **Capture & Enrol** (repeat with different angles for better accuracy).

#### 5. Start the video processor
```bash
python video_processor.py
# Optional flags:
#   --camera-index 1        (use a different webcam)
#   --backend-url http://...  (remote backend)
```

The dashboard updates in real-time as soon as the processor detects faces.
