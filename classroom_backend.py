"""
FastAPI backend for real-time classroom emotion and attendance monitoring.

Manages enrolled student face encodings, tracks attendance and emotion
history, evaluates alert conditions, and streams live data to the teacher
dashboard via WebSocket.

Endpoints
---------
GET  /                  Serve the teacher dashboard HTML.
POST /enroll            Register a new student face from a base64 image.
POST /update            Receive a per-student update from the video processor.
GET  /students          Return the current snapshot of all student states.
WS   /ws/dashboard      Push live JSON packets to connected dashboard clients.
"""

import asyncio
import base64
import json
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import face_recognition
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# File that persists face encodings across restarts.
ENCODINGS_FILE = "face_encodings.json"

# Directory that contains the static dashboard HTML.
STATIC_DIR = Path(__file__).parent / "static"

# Student marked 'Away' when not seen for this many seconds.
AWAY_THRESHOLD_SECONDS = 5 * 60        # 5 minutes

# 'Emotional Distress' alert fires after this many continuous seconds of
# sad/angry emotion.
DISTRESS_THRESHOLD_SECONDS = 10 * 60   # 10 minutes

# 'Low Engagement' alert fires after this many continuous seconds of the
# student looking away AND showing neutral emotion.
ENGAGEMENT_THRESHOLD_SECONDS = 2 * 60  # 2 minutes

# Keep up to this much emotion/pose history per student.
HISTORY_WINDOW_SECONDS = 15 * 60       # 15 minutes

# Emotions that can trigger the Emotional Distress alert.
DISTRESS_EMOTIONS = {"sad", "angry"}

# Emotions and head poses that contribute to the Low Engagement alert.
DISENGAGED_EMOTIONS = {"neutral"}
DISENGAGED_POSES = {"left", "right", "down"}

# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class StudentUpdate(BaseModel):
    """Sent by the video processor each time a student is detected."""
    student_name: str
    emotion: str
    head_pose: str      # "forward" | "left" | "right" | "down"
    timestamp: float    # Unix epoch seconds


class EnrollRequest(BaseModel):
    """Sent by the enrollment UI with a captured webcam frame."""
    student_name: str
    image_base64: str   # base64-encoded JPEG/PNG frame


# ---------------------------------------------------------------------------
# Shared in-memory state
# ---------------------------------------------------------------------------

# Enrolled face encodings: { student_name: [[float, ...], ...] }
# Each inner list is a 128-dimensional face encoding stored as a plain list
# so it can be round-tripped through JSON.
enrolled_encodings: dict[str, list[list[float]]] = {}

# Per-student runtime state.
# emotion_history and head_pose_history are deque[(timestamp, value)].
student_states: dict[str, dict] = {}

# Active WebSocket connections to the dashboard.
connected_clients: list[WebSocket] = []


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def load_encodings() -> None:
    """Load persisted face encodings from disk and initialise student states."""
    path = Path(ENCODINGS_FILE)
    if not path.exists():
        return
    with open(path) as fh:
        data = json.load(fh)
    enrolled_encodings.update(data)
    for name in enrolled_encodings:
        if name not in student_states:
            _init_student_state(name)


def save_encodings() -> None:
    """Persist face encodings to disk."""
    with open(ENCODINGS_FILE, "w") as fh:
        json.dump(enrolled_encodings, fh)


# ---------------------------------------------------------------------------
# Student state helpers
# ---------------------------------------------------------------------------


def _init_student_state(name: str) -> None:
    """Initialise the runtime state entry for a student."""
    student_states[name] = {
        "emotion": "neutral",
        "head_pose": "forward",
        "attendance_status": "Absent",
        "last_seen": None,
        "is_present": False,
        "active_alert": None,
        # Bounded deques prevent unbounded memory growth.
        "emotion_history": deque(maxlen=50_000),
        "head_pose_history": deque(maxlen=50_000),
    }


def _trim_history(history: deque, max_age: float, now: float) -> None:
    """Remove entries older than max_age seconds from a history deque."""
    while history and now - history[0][0] > max_age:
        history.popleft()


def _check_alerts(name: str, now: float) -> Optional[str]:
    """
    Evaluate alert conditions and return the highest-priority active alert
    name, or None if no alert is currently active.
    """
    state = student_states[name]
    emotion_hist = state["emotion_history"]
    pose_hist = state["head_pose_history"]

    # -- Emotional Distress -----------------------------------------------
    # All emotion samples in the last DISTRESS_THRESHOLD_SECONDS window must
    # be 'sad' or 'angry', AND the window must be at least
    # DISTRESS_THRESHOLD_SECONDS wide (so it does not fire immediately).
    if emotion_hist:
        cutoff = now - DISTRESS_THRESHOLD_SECONDS
        recent = [(t, e) for t, e in emotion_hist if t >= cutoff]
        if recent:
            span = now - recent[0][0]
            if span >= DISTRESS_THRESHOLD_SECONDS and all(
                e in DISTRESS_EMOTIONS for _, e in recent
            ):
                return "Emotional Distress"

    # -- Low Engagement ---------------------------------------------------
    # Head consistently not facing forward for ENGAGEMENT_THRESHOLD_SECONDS
    # AND emotion in the same window is consistently neutral.
    if pose_hist:
        cutoff = now - ENGAGEMENT_THRESHOLD_SECONDS
        recent_poses = [(t, p) for t, p in pose_hist if t >= cutoff]
        if recent_poses:
            span = now - recent_poses[0][0]
            if span >= ENGAGEMENT_THRESHOLD_SECONDS and all(
                p in DISENGAGED_POSES for _, p in recent_poses
            ):
                # Also require sustained neutral emotion in the same window.
                recent_emotions = [(t, e) for t, e in emotion_hist if t >= cutoff]
                if recent_emotions and all(
                    e in DISENGAGED_EMOTIONS for _, e in recent_emotions
                ):
                    return "Low Engagement"

    return None


# ---------------------------------------------------------------------------
# WebSocket broadcast helper
# ---------------------------------------------------------------------------


async def _broadcast(payload: dict) -> None:
    """Send a JSON packet to all connected dashboard clients."""
    message = json.dumps(payload)
    stale: list[WebSocket] = []
    for ws in connected_clients:
        try:
            await ws.send_text(message)
        except Exception:
            stale.append(ws)
    for ws in stale:
        if ws in connected_clients:
            connected_clients.remove(ws)


# ---------------------------------------------------------------------------
# Background task: mark absent students as Away
# ---------------------------------------------------------------------------


async def _attendance_watchdog() -> None:
    """
    Every 30 seconds, mark students as 'Away' if they have not been seen
    within the AWAY_THRESHOLD_SECONDS window.
    """
    while True:
        await asyncio.sleep(30)
        now = time.time()
        for name, state in student_states.items():
            if state["is_present"] and state["last_seen"] is not None:
                if now - state["last_seen"] > AWAY_THRESHOLD_SECONDS:
                    state["is_present"] = False
                    state["attendance_status"] = "Away"
                    await _broadcast({
                        "student_name": name,
                        "emotion": state["emotion"],
                        "head_pose": state["head_pose"],
                        "timestamp": now,
                        "is_present": False,
                        "attendance_status": "Away",
                        "active_alert": state["active_alert"],
                    })


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    load_encodings()
    asyncio.create_task(_attendance_watchdog())
    yield


app = FastAPI(title="Classroom Monitoring System", lifespan=lifespan)

STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the teacher dashboard single-page application."""
    dashboard_path = STATIC_DIR / "classroom_dashboard.html"
    if not dashboard_path.exists():
        return HTMLResponse("<h1>Dashboard file not found.</h1>", status_code=404)
    return HTMLResponse(dashboard_path.read_text(encoding="utf-8"))


@app.post("/enroll")
async def enroll_student(request: EnrollRequest):
    """
    Register a new student.

    Accepts a base64-encoded image, detects the face, computes a 128-d
    encoding, and appends it to the student's known-encoding list.
    Multiple enrolment frames improve recognition accuracy.
    """
    name = request.student_name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Student name is required.")

    # Decode image --------------------------------------------------------
    try:
        img_bytes = base64.b64decode(request.image_base64)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Image could not be decoded.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image decode error: {exc}") from exc

    # Detect and encode face ----------------------------------------------
    locations = face_recognition.face_locations(rgb)
    if not locations:
        raise HTTPException(status_code=400, detail="No face detected in the provided image.")

    encodings = face_recognition.face_encodings(rgb, locations)
    if not encodings:
        raise HTTPException(status_code=400, detail="Face detected but could not be encoded.")

    encoding: list[float] = encodings[0].tolist()

    if name not in enrolled_encodings:
        enrolled_encodings[name] = []
    enrolled_encodings[name].append(encoding)
    save_encodings()

    if name not in student_states:
        _init_student_state(name)

    # Broadcast updated student list to dashboard.
    await _broadcast({
        "event": "enrolled",
        "student_name": name,
        "face_count": len(enrolled_encodings[name]),
    })

    return {
        "message": f"Student '{name}' enrolled successfully.",
        "face_count": len(enrolled_encodings[name]),
    }


@app.post("/update")
async def update_student(update: StudentUpdate):
    """
    Receive a real-time update from the video processor for one student.

    Updates attendance status, emotion/head-pose histories, evaluates alerts,
    and broadcasts the new state to all dashboard WebSocket clients.
    """
    name = update.student_name
    now = update.timestamp

    # Lazily initialise state for newly recognised students.
    if name not in student_states:
        _init_student_state(name)

    state = student_states[name]
    was_present = state["is_present"]

    state["emotion"] = update.emotion
    state["head_pose"] = update.head_pose
    state["last_seen"] = now
    state["is_present"] = True

    if not was_present:
        state["attendance_status"] = "Present"

    # Update histories ----------------------------------------------------
    _trim_history(state["emotion_history"], HISTORY_WINDOW_SECONDS, now)
    _trim_history(state["head_pose_history"], HISTORY_WINDOW_SECONDS, now)
    state["emotion_history"].append((now, update.emotion))
    state["head_pose_history"].append((now, update.head_pose))

    # Evaluate alert conditions -------------------------------------------
    state["active_alert"] = _check_alerts(name, now)

    # Broadcast to dashboard ----------------------------------------------
    await _broadcast({
        "student_name": name,
        "emotion": update.emotion,
        "head_pose": update.head_pose,
        "timestamp": now,
        "is_present": True,
        "attendance_status": state["attendance_status"],
        "active_alert": state["active_alert"],
    })

    return {"status": "ok"}


@app.get("/students")
async def get_students():
    """
    Return the current snapshot of all enrolled students.

    Used by the dashboard to populate itself when it first loads, before
    any WebSocket messages arrive.
    """
    result = []
    for name, state in student_states.items():
        result.append({
            "student_name": name,
            "emotion": state["emotion"],
            "head_pose": state["head_pose"],
            "attendance_status": state["attendance_status"],
            "is_present": state["is_present"],
            "active_alert": state["active_alert"],
            "last_seen": state["last_seen"],
        })
    return result


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """
    Long-lived WebSocket connection for the teacher dashboard.

    On connect the current student snapshot is sent as individual packets
    so the dashboard can hydrate immediately.  Subsequent updates are pushed
    by the /update endpoint via _broadcast().
    """
    await websocket.accept()
    connected_clients.append(websocket)

    # Send current state to the newly connected client.
    now = time.time()
    for name, state in student_states.items():
        try:
            await websocket.send_text(json.dumps({
                "student_name": name,
                "emotion": state["emotion"],
                "head_pose": state["head_pose"],
                "timestamp": now,
                "is_present": state["is_present"],
                "attendance_status": state["attendance_status"],
                "active_alert": state["active_alert"],
            }))
        except Exception:
            break

    try:
        # Keep the connection open; client may send pings or enrollment events.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
