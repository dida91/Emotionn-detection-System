"""
FastAPI backend for real-time classroom emotion and attendance monitoring.

Manages enrolled student face encodings, tracks attendance and emotion
history, evaluates alert conditions, and streams live data to the teacher
dashboard via WebSocket.

Endpoints
---------
GET  /                  Serve the teacher dashboard HTML (auth required).
GET  /login             Serve the teacher login page.
POST /login             Authenticate and start a session.
POST /logout            Destroy the current session.
GET  /me                Return the authenticated teacher's username.
POST /enroll            Register a new student face from a base64 image.
POST /update            Receive a per-student update from the video processor.
GET  /students          Return the current snapshot of all student states.
GET  /attendance        Return day-wise attendance records.
WS   /ws/dashboard      Push live JSON packets to connected dashboard clients.
"""

import asyncio
import base64
import json
import os
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import date as date_cls, datetime, timezone
from functools import partial
from pathlib import Path
from typing import Optional

import cv2
import face_recognition
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from starlette.middleware.sessions import SessionMiddleware

# Load environment variables from .env file before reading any os.environ values.
load_dotenv()

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
DISTRESS_THRESHOLD_SECONDS = 20 * 60   # 20 minutes

# 'Low Engagement' alert fires after this many continuous seconds of the
# student looking away AND showing neutral emotion.
ENGAGEMENT_THRESHOLD_SECONDS = 2 * 60  # 2 minutes

# Keep up to this much emotion/pose history per student.
HISTORY_WINDOW_SECONDS = 25 * 60       # 25 minutes

# Emotions that can trigger the Emotional Distress alert.
DISTRESS_EMOTIONS = {"sad", "angry"}

# Emotions and head poses that contribute to the Low Engagement alert.
DISENGAGED_EMOTIONS = {"neutral"}
DISENGAGED_POSES = {"left", "right", "down"}

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

_SECRET_KEY = os.environ.get("SECRET_KEY", secrets.token_hex(32))
_TEACHER_USERNAME = os.environ.get("TEACHER_USERNAME", "teacher")
_TEACHER_PASSWORD = os.environ.get("TEACHER_PASSWORD", "")

if not _TEACHER_PASSWORD:
    import warnings
    warnings.warn(
        "TEACHER_PASSWORD is not set. Login will be disabled until it is configured.",
        RuntimeWarning,
        stacklevel=1,
    )

if not os.environ.get("SECRET_KEY"):
    import warnings
    warnings.warn(
        "SECRET_KEY is not set. A random key will be generated, invalidating sessions on restart.",
        RuntimeWarning,
        stacklevel=1,
    )

# ---------------------------------------------------------------------------
# Database setup (PostgreSQL via DATABASE_URL from .env)
# ---------------------------------------------------------------------------

_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///./classroom_attendance.db",
)

_engine = create_engine(_DATABASE_URL, pool_pre_ping=True)
_SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
_Base = declarative_base()


class AttendanceRecord(_Base):
    """One row per student per calendar day."""

    __tablename__ = "classroom_attendance"

    id           = Column(Integer, primary_key=True)
    student_name = Column(String(120), nullable=False)
    date         = Column(Date, nullable=False)
    status       = Column(String(20), nullable=False, default="Absent")
    first_seen   = Column(DateTime(timezone=True), nullable=True)
    last_seen    = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("student_name", "date", name="uq_classroom_student_date"),
    )


def _init_db() -> None:
    """Create all tables if they don't exist yet."""
    _Base.metadata.create_all(bind=_engine)


# ---------------------------------------------------------------------------
# Database helper functions (synchronous, called via run_in_executor)
# ---------------------------------------------------------------------------


def _db_upsert_attendance(student_name: str, status: str, ts: float) -> None:
    """
    Insert or update today's attendance record for *student_name*.

    If the record doesn't exist yet it is created; otherwise only *status*
    and *last_seen* (and *first_seen* when transitioning to Present for the
    first time) are updated.
    """
    today = datetime.fromtimestamp(ts, tz=timezone.utc).date()
    dt_now = datetime.fromtimestamp(ts, tz=timezone.utc)
    with _SessionLocal() as session:
        record = (
            session.query(AttendanceRecord)
            .filter_by(student_name=student_name, date=today)
            .first()
        )
        if record is None:
            record = AttendanceRecord(
                student_name=student_name,
                date=today,
                status=status,
                first_seen=dt_now if status == "Present" else None,
                last_seen=dt_now if status == "Present" else None,
            )
            session.add(record)
        else:
            record.status = status
            record.last_seen = dt_now if status in ("Present", "Away") else record.last_seen
            if status == "Present" and record.first_seen is None:
                record.first_seen = dt_now
        session.commit()


def _db_get_attendance(filter_date: Optional[date_cls] = None) -> list[dict]:
    """
    Return attendance records for *filter_date* (defaults to today).

    Each row is a plain dict with keys: student_name, date, status,
    first_seen, last_seen.
    """
    if filter_date is None:
        filter_date = date_cls.today()
    with _SessionLocal() as session:
        rows = (
            session.query(AttendanceRecord)
            .filter_by(date=filter_date)
            .order_by(AttendanceRecord.student_name)
            .all()
        )
        return [
            {
                "student_name": r.student_name,
                "date": r.date.isoformat(),
                "status": r.status,
                "first_seen": r.first_seen.isoformat() if r.first_seen else None,
                "last_seen": r.last_seen.isoformat() if r.last_seen else None,
            }
            for r in rows
        ]


def _db_get_attendance_dates() -> list[str]:
    """Return sorted list of distinct dates that have attendance records."""
    with _SessionLocal() as session:
        rows = session.query(AttendanceRecord.date).distinct().order_by(AttendanceRecord.date.desc()).all()
        return [r.date.isoformat() for r in rows]

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
    loop = asyncio.get_event_loop()
    while True:
        await asyncio.sleep(30)
        now = time.time()
        for name, state in student_states.items():
            if state["is_present"] and state["last_seen"] is not None:
                if now - state["last_seen"] > AWAY_THRESHOLD_SECONDS:
                    state["is_present"] = False
                    state["attendance_status"] = "Away"
                    await loop.run_in_executor(
                        None, partial(_db_upsert_attendance, name, "Away", now)
                    )
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
    _init_db()
    load_encodings()
    asyncio.create_task(_attendance_watchdog())
    yield


app = FastAPI(title="Classroom Monitoring System", lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=_SECRET_KEY,
    same_site="lax",
    https_only=False,
)
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _is_authenticated(request: Request) -> bool:
    """Return True if the request carries a valid teacher session."""
    return bool(request.session.get("authenticated"))


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serve the teacher dashboard single-page application (requires login)."""
    if not _is_authenticated(request):
        return RedirectResponse("/login", status_code=303)
    dashboard_path = STATIC_DIR / "classroom_dashboard.html"
    if not dashboard_path.exists():
        return HTMLResponse("<h1>Dashboard file not found.</h1>", status_code=404)
    return HTMLResponse(dashboard_path.read_text(encoding="utf-8"))


@app.get("/login", response_class=HTMLResponse)
async def serve_login(request: Request):
    """Serve the teacher login page."""
    if _is_authenticated(request):
        return RedirectResponse("/", status_code=303)
    login_path = STATIC_DIR / "classroom_login.html"
    if not login_path.exists():
        return HTMLResponse("<h1>Login page not found.</h1>", status_code=404)
    return HTMLResponse(login_path.read_text(encoding="utf-8"))


@app.post("/login")
async def do_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    """Validate credentials and establish an authenticated session."""
    if not _TEACHER_PASSWORD:
        raise HTTPException(status_code=503, detail="Authentication is not configured on this server.")
    if username == _TEACHER_USERNAME and password == _TEACHER_PASSWORD:
        request.session["authenticated"] = True
        request.session["username"] = username
        return RedirectResponse("/", status_code=303)
    login_path = STATIC_DIR / "classroom_login.html"
    html = login_path.read_text(encoding="utf-8") if login_path.exists() else "<h1>Login</h1>"
    # Inject an error message into the page.
    html = html.replace(
        'id="login-error" class="login-error"',
        'id="login-error" class="login-error visible"',
    )
    return HTMLResponse(html, status_code=401)


@app.post("/logout")
async def do_logout(request: Request):
    """Clear the teacher session."""
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/me")
async def get_me(request: Request):
    """Return the authenticated teacher's username."""
    if not _is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return {"username": request.session.get("username", "Teacher")}


@app.post("/enroll")
async def enroll_student(request: Request, req: EnrollRequest):
    """
    Register a new student.

    Accepts a base64-encoded image, detects the face, computes a 128-d
    encoding, and appends it to the student's known-encoding list.
    Multiple enrolment frames improve recognition accuracy.
    """
    if not _is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required.")
    name = req.student_name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Student name is required.")

    # Decode image --------------------------------------------------------
    try:
        img_bytes = base64.b64decode(req.image_base64)
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

    # Persist attendance to database (only when status changes to Present or
    # periodically to keep last_seen fresh; we update every time is_present
    # transitions or every 60 s by checking last DB write time).
    if not was_present:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, partial(_db_upsert_attendance, name, "Present", now)
        )

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
async def get_students(request: Request):
    """
    Return the current snapshot of all enrolled students.

    Used by the dashboard to populate itself when it first loads, before
    any WebSocket messages arrive.
    """
    if not _is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required.")
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


@app.get("/attendance")
async def get_attendance(
    request: Request,
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format (defaults to today)"),
):
    """
    Return day-wise attendance records from the database.

    Query parameter ``date`` selects the calendar day (ISO format).
    Omit it to get today's records.  Also returns the list of all dates
    that have records so the dashboard can build a date-picker.
    """
    if not _is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required.")

    filter_date: Optional[date_cls] = None
    if date:
        try:
            filter_date = date_cls.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    loop = asyncio.get_event_loop()
    records = await loop.run_in_executor(None, partial(_db_get_attendance, filter_date))
    dates = await loop.run_in_executor(None, _db_get_attendance_dates)
    return JSONResponse({"records": records, "available_dates": dates})


@app.delete("/students/{student_name}")
async def remove_student(student_name: str, request: Request):
    """
    Remove a specific enrolled student.

    Deletes all stored face encodings and runtime state for the named student,
    persists the change to disk, and broadcasts a removal event to the dashboard.
    """
    if not _is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required.")
    if student_name not in enrolled_encodings:
        raise HTTPException(status_code=404, detail=f"Student '{student_name}' not found.")

    enrolled_encodings.pop(student_name, None)
    student_states.pop(student_name, None)
    save_encodings()

    await _broadcast({"event": "removed", "student_name": student_name})
    return {"message": f"Student '{student_name}' removed successfully."}


@app.delete("/students")
async def remove_all_students(request: Request):
    """
    Remove all enrolled students.

    Clears every face encoding and runtime state entry, persists the change to
    disk, and broadcasts a removal event to the dashboard.
    """
    if not _is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required.")

    enrolled_encodings.clear()
    student_states.clear()
    save_encodings()

    await _broadcast({"event": "removed_all"})
    return {"message": "All students removed successfully."}


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
    # Reject unauthenticated WebSocket connections.
    # Read the session from the ASGI scope directly, since SessionMiddleware
    # sets scope["session"] for both HTTP and WebSocket connections.
    session = websocket.scope.get("session", {})
    if not session.get("authenticated"):
        await websocket.close(code=1008)
        return

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
