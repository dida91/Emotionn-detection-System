import hmac
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from emotion_model import EmotionModel
from face_detection import FaceDetector

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
VIDEO_FRAME_STEP = 10

app = Flask(__name__)
app_env = os.environ.get("APP_ENV", "development").lower()
secret_key = os.environ.get("SECRET_KEY")
if not secret_key and app_env == "production":
    raise RuntimeError("SECRET_KEY must be set in production.")
app.config["SECRET_KEY"] = secret_key or secrets.token_hex(32)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/emotion_detection",
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


class Teacher(db.Model):
    __tablename__ = "teachers"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)


class EmotionRecord(db.Model):
    __tablename__ = "emotion_records"

    id = db.Column(db.Integer, primary_key=True)
    teacher_id = db.Column(db.Integer, db.ForeignKey("teachers.id"), nullable=False)
    source_filename = db.Column(db.String(255), nullable=False)
    media_type = db.Column(db.String(16), nullable=False)
    dominant_emotion = db.Column(db.String(32), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    emotion_scores = db.Column(db.JSON, nullable=False)
    analyzed_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


face_detector = FaceDetector()
emotion_model = EmotionModel()



def _ensure_default_teacher() -> None:
    username = os.environ.get("TEACHER_USERNAME", "teacher")
    password = os.environ.get("TEACHER_PASSWORD")
    if not password:
        raise RuntimeError("TEACHER_PASSWORD must be set before starting the app.")
    if len(password) < 12:
        raise RuntimeError("TEACHER_PASSWORD must be at least 12 characters long.")

    teacher = Teacher.query.filter_by(username=username).first()
    if teacher is None:
        db.session.add(
            Teacher(username=username, password_hash=generate_password_hash(password))
        )
        db.session.commit()


with app.app_context():
    db.create_all()
    _ensure_default_teacher()


def _csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_hex(32)
        session["csrf_token"] = token
    return token


@app.context_processor
def csrf_context():
    return {"csrf_token": _csrf_token()}


def _validate_csrf() -> bool:
    submitted = request.form.get("csrf_token", "")
    expected = session.get("csrf_token", "")
    return bool(submitted and expected and hmac.compare_digest(submitted, expected))



def _aggregate_scores(frames_scores: list[dict[str, float]]) -> dict[str, float]:
    if not frames_scores:
        return {
            "happy": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "surprised": 0.0,
            "fear": 0.0,
            "disgust": 0.0,
            "neutral": 1.0,
        }

    labels = frames_scores[0].keys()
    output = {}
    for label in labels:
        output[label] = sum(score[label] for score in frames_scores) / len(frames_scores)
    return output



def _analyze_image(image_bytes: bytes) -> tuple[str, float, dict[str, float]]:
    np_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    image_array = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)

    if image_array is None:
        raise ValueError("Unable to decode image file.")

    faces = face_detector.detect_faces(image_array)
    if len(faces) == 0:
        return "neutral", 0.0, {
            "happy": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "surprised": 0.0,
            "fear": 0.0,
            "disgust": 0.0,
            "neutral": 1.0,
        }

    best_emotion = "neutral"
    best_confidence = 0.0
    scores_list = []

    for (x, y, w, h) in faces:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(image_array.shape[1], x + w)
        y1 = min(image_array.shape[0], y + h)
        face = image_array[y0:y1, x0:x1]
        if face.size == 0:
            continue
        emotion, confidence, scores = emotion_model.predict_emotions(face)
        scores_list.append(scores)
        if confidence > best_confidence:
            best_emotion = emotion
            best_confidence = confidence

    if not scores_list:
        return "neutral", 0.0, {
            "happy": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "surprised": 0.0,
            "fear": 0.0,
            "disgust": 0.0,
            "neutral": 1.0,
        }

    return best_emotion, best_confidence, _aggregate_scores(scores_list)



def _analyze_video(video_path: str) -> tuple[str, float, dict[str, float]]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Unable to open video file.")

    frame_index = 0
    frame_step = VIDEO_FRAME_STEP
    frame_scores = []
    best_emotion = "neutral"
    best_confidence = 0.0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            faces = face_detector.detect_faces(frame)
            for (x, y, w, h) in faces:
                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(frame.shape[1], x + w)
                y1 = min(frame.shape[0], y + h)
                face = frame[y0:y1, x0:x1]
                if face.size == 0:
                    continue
                emotion, confidence, scores = emotion_model.predict_emotions(face)
                frame_scores.append(scores)
                if confidence > best_confidence:
                    best_emotion = emotion
                    best_confidence = confidence

            frame_index += 1
    finally:
        capture.release()

    if not frame_scores:
        return "neutral", 0.0, {
            "happy": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "surprised": 0.0,
            "fear": 0.0,
            "disgust": 0.0,
            "neutral": 1.0,
        }

    return best_emotion, best_confidence, _aggregate_scores(frame_scores)



def _is_logged_in() -> bool:
    return "teacher_id" in session


@app.route("/", methods=["GET"])
def index():
    if _is_logged_in():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if not _validate_csrf():
            flash("Invalid session token. Please try again.", "error")
            return redirect(url_for("login"))

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        teacher = Teacher.query.filter_by(username=username).first()
        if teacher and check_password_hash(teacher.password_hash, password):
            session["teacher_id"] = teacher.id
            session["teacher_username"] = teacher.username
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "error")

    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    if not _validate_csrf():
        flash("Invalid session token. Please try again.", "error")
        return redirect(url_for("dashboard"))
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if not _is_logged_in():
        return redirect(url_for("login"))

    if request.method == "POST":
        if not _validate_csrf():
            flash("Invalid session token. Please try again.", "error")
            return redirect(url_for("dashboard"))

        uploaded_file = request.files.get("media")
        if uploaded_file is None or uploaded_file.filename == "":
            flash("Please upload an image or video file.", "error")
            return redirect(url_for("dashboard"))

        filename = secure_filename(uploaded_file.filename)
        extension = Path(filename).suffix.lower().lstrip(".")

        try:
            if extension in ALLOWED_IMAGE_EXTENSIONS:
                dominant, confidence, scores = _analyze_image(uploaded_file.read())
                media_type = "image"
            elif extension in ALLOWED_VIDEO_EXTENSIONS:
                # Keep file on disk temporarily so OpenCV can open it by path.
                with tempfile.NamedTemporaryFile(suffix=f".{extension}", delete=False) as tmp:
                    temp_path = tmp.name
                    uploaded_file.save(temp_path)
                try:
                    dominant, confidence, scores = _analyze_video(temp_path)
                    media_type = "video"
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                flash("Unsupported file type. Upload image or video.", "error")
                return redirect(url_for("dashboard"))
        except ValueError as exc:
            flash(f"Analysis failed: {exc}", "error")
            return redirect(url_for("dashboard"))
        except Exception as exc:
            app.logger.exception("Media analysis failed: %s", exc)
            flash(
                "Analysis failed. Please check the file and try again.",
                "error",
            )
            return redirect(url_for("dashboard"))

        record = EmotionRecord(
            teacher_id=session["teacher_id"],
            source_filename=filename,
            media_type=media_type,
            dominant_emotion=dominant,
            confidence=confidence,
            emotion_scores=scores,
        )
        db.session.add(record)
        db.session.commit()
        flash("Emotion analysis completed.", "success")
        return redirect(url_for("dashboard"))

    records = (
        EmotionRecord.query.filter_by(teacher_id=session["teacher_id"])
        .order_by(EmotionRecord.analyzed_at.desc())
        .all()
    )
    return render_template(
        "dashboard.html",
        username=session.get("teacher_username", "teacher"),
        records=records,
    )


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
