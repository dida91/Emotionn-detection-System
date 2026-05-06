"""
Video processor for the real-time classroom monitoring system.

Captures frames from a camera, detects and recognises student faces using
the ``face_recognition`` library, estimates head pose with OpenCV's solvePnP,
detects eye state (open / drowsy / closed) via Eye Aspect Ratio (EAR),
runs emotion detection via the existing EmotionModel, and POST-s per-student
updates to the classroom backend API.

Usage
-----
    python video_processor.py [--camera-index 0] [--backend-url http://localhost:8000]

The video processor reads enrolled face encodings from ``face_encodings.json``
(written by the backend's /enroll endpoint) and reloads the file every few
seconds to pick up newly enrolled students without a restart.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import face_recognition
import numpy as np
import requests

from emotion_model import EmotionModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BACKEND_URL = "http://localhost:8000"
ENCODINGS_FILE = "face_encodings.json"

# Maximum normalised distance for a positive face match.
RECOGNITION_TOLERANCE = 0.6

# Minimum elapsed seconds between two updates sent for the same student.
# Reduces HTTP traffic without losing fidelity.
UPDATE_INTERVAL_SECONDS = 1.0

# How often (in seconds) the processor reloads the encodings file so that
# newly enrolled students are picked up without a restart.
ENCODINGS_RELOAD_INTERVAL = 5.0

# Scale factor applied before running face detection on each frame.
# A smaller value speeds up detection at some cost to accuracy for small faces.
DETECTION_SCALE = 0.5

# Eye Aspect Ratio thresholds.
# EAR < EYES_CLOSED_THRESH  → closed
# EAR < EYES_DROWSY_THRESH  → drowsy
# EAR >= EYES_DROWSY_THRESH → open
EYES_CLOSED_THRESH = 0.20
EYES_DROWSY_THRESH = 0.25

# ---------------------------------------------------------------------------
# Standard 3-D face model used for head-pose estimation via solvePnP.
# Coordinates are in an arbitrary metric unit; only ratios matter.
# Points: nose tip, chin centre, left-eye outer corner, right-eye outer
#         corner, left-mouth corner, right-mouth corner.
# ---------------------------------------------------------------------------
_FACE_3D_MODEL = np.array([
    (0.0,    0.0,    0.0),     # Nose tip
    (0.0,  -330.0, -65.0),    # Chin
    (-225.0, 170.0, -135.0),  # Left eye outer corner
    (225.0,  170.0, -135.0),  # Right eye outer corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0,  -150.0, -125.0), # Right mouth corner
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Eye state detection (Eye Aspect Ratio)
# ---------------------------------------------------------------------------


# Minimum eye width (pixels) below which EAR is considered unreliable.
MIN_EYE_WIDTH = 1e-6


def _eye_aspect_ratio(eye_landmarks: list) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for a single eye.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    where p1..p6 are the 6 landmark points of the eye in order:
    left-corner, top-left, top-right, right-corner, bottom-right, bottom-left.

    A high EAR (~0.3+) means the eye is open; a low EAR means it is closed.
    """
    p = [np.array(pt, dtype=np.float64) for pt in eye_landmarks]
    # Vertical distances
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    # Horizontal distance
    C = np.linalg.norm(p[0] - p[3])
    if C < MIN_EYE_WIDTH:
        return 0.0
    return (A + B) / (2.0 * C)


def _detect_eye_state(landmarks: dict) -> str:
    """
    Return the eye state based on the average Eye Aspect Ratio.

    Returns
    -------
    str
        ``"open"``, ``"drowsy"``, or ``"closed"``.
    """
    left_eye  = landmarks.get("left_eye", [])
    right_eye = landmarks.get("right_eye", [])

    if len(left_eye) < 6 or len(right_eye) < 6:
        return "open"   # not enough landmarks; assume open

    ear = (_eye_aspect_ratio(left_eye) + _eye_aspect_ratio(right_eye)) / 2.0  # average both eyes

    if ear < EYES_CLOSED_THRESH:
        return "closed"
    if ear < EYES_DROWSY_THRESH:
        return "drowsy"
    return "open"


# ---------------------------------------------------------------------------
# Head pose estimation
# ---------------------------------------------------------------------------


def _estimate_head_pose(landmarks: dict, image_shape: tuple) -> str:
    """
    Estimate the coarse head orientation from ``face_recognition`` landmarks.

    Parameters
    ----------
    landmarks : dict
        Face landmark dictionary as returned by
        ``face_recognition.face_landmarks()``.
    image_shape : tuple
        Shape of the full (pre-scale) image (height, width[, channels]).

    Returns
    -------
    str
        One of ``"forward"``, ``"left"``, ``"right"``, or ``"down"``.
    """
    try:
        # Map named landmarks to 2-D image points matching _FACE_3D_MODEL.
        nose_tip          = landmarks["nose_tip"][2]       # centre of nose tip
        chin              = landmarks["chin"][8]           # centre of chin
        left_eye_corner   = landmarks["left_eye"][0]       # outer corner
        right_eye_corner  = landmarks["right_eye"][3]      # outer corner
        left_mouth        = landmarks["top_lip"][0]        # left corner
        right_mouth       = landmarks["top_lip"][6]        # right corner

        image_points = np.array(
            [nose_tip, chin, left_eye_corner, right_eye_corner,
             left_mouth, right_mouth],
            dtype=np.float64,
        )

        h, w = image_shape[:2]
        focal_length = float(w)
        camera_matrix = np.array(
            [[focal_length, 0.0, w / 2.0],
             [0.0, focal_length, h / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(
            _FACE_3D_MODEL,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return "forward"

        rmat, _ = cv2.Rodrigues(rvec)
        # RQDecomp3x3 returns angles in degrees.
        euler, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, _ = euler

        if yaw < -20:
            return "left"
        if yaw > 20:
            return "right"
        if pitch < -15:
            return "down"
        return "forward"

    except (KeyError, IndexError):
        return "forward"


# ---------------------------------------------------------------------------
# Face-recognition helpers
# ---------------------------------------------------------------------------


def _load_encodings() -> dict[str, list[np.ndarray]]:
    """
    Load enrolled face encodings from disk.

    Returns a dict mapping student names to a list of 128-d numpy arrays.
    """
    path = Path(ENCODINGS_FILE)
    if not path.exists():
        return {}
    with open(path) as fh:
        raw: dict[str, list[list[float]]] = json.load(fh)
    return {
        name: [np.array(enc, dtype=np.float64) for enc in encs]
        for name, encs in raw.items()
    }


def _match_face(
    encoding: np.ndarray,
    known: dict[str, list[np.ndarray]],
    tolerance: float,
) -> str:
    """
    Find the closest enrolled student for a given face encoding.

    Returns the student name, or ``"unknown"`` if no match is within
    *tolerance*.
    """
    best_name = "unknown"
    best_dist = tolerance
    for name, name_encs in known.items():
        if not name_encs:
            continue
        dists = face_recognition.face_distance(name_encs, encoding)
        min_dist = float(dists.min())
        if min_dist < best_dist:
            best_dist = min_dist
            best_name = name
    return best_name


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


def run(camera_index: int = 0, backend_url: str = DEFAULT_BACKEND_URL) -> None:
    """
    Capture frames from *camera_index*, process faces, and forward updates
    to the backend at *backend_url*.

    Press ``q`` in the preview window to stop.
    """
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera at index {camera_index}.")

    emotion_model = EmotionModel()
    known_encodings = _load_encodings()
    last_update: dict[str, float] = {}   # student_name -> last POST time
    last_reload = time.time()

    print(f"[VideoProcessor] Camera index : {camera_index}")
    print(f"[VideoProcessor] Backend URL  : {backend_url}")
    print(f"[VideoProcessor] Known students: {list(known_encodings)}")

    frame_count = 0
    process_every_n_frames = 5
    last_detections = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            now = time.time()

            # Periodically reload encodings for newly enrolled students.
            if now - last_reload >= ENCODINGS_RELOAD_INTERVAL:
                fresh = _load_encodings()
                if set(fresh) != set(known_encodings):
                    known_encodings = fresh
                    print(f"[VideoProcessor] Reloaded — students: {list(known_encodings)}")
                last_reload = now

            if frame_count % process_every_n_frames == 0:
                last_detections = []
                # Downscale for faster detection.
                small = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                face_locs  = face_recognition.face_locations(rgb_small, model="hog")
                face_encs  = face_recognition.face_encodings(rgb_small, face_locs)
                face_lands = face_recognition.face_landmarks(rgb_small, face_locs)

                for loc, enc, land in zip(face_locs, face_encs, face_lands):
                    name = _match_face(enc, known_encodings, RECOGNITION_TOLERANCE)
                    if name == "unknown":
                        last_detections.append({"loc": loc, "name": "unknown"})
                        continue

                    # Scale landmarks back to original frame coordinates.
                    scale_inv = 1.0 / DETECTION_SCALE
                    scaled_land = {
                        feat: [(int(x * scale_inv), int(y * scale_inv)) for x, y in pts]
                        for feat, pts in land.items()
                    }

                    head_pose = _estimate_head_pose(scaled_land, frame.shape)

                    # Detect eye state from landmarks.
                    eye_state = _detect_eye_state(scaled_land)

                    # Crop face for emotion detection.
                    top, right, bottom, left = [int(c / DETECTION_SCALE) for c in loc]
                    face_crop = frame[top:bottom, left:right]
                    if face_crop.size > 0:
                        emotion, _, _ = emotion_model.predict_emotions(face_crop)
                    else:
                        emotion = "neutral"
                        
                    last_detections.append({
                        "loc": loc, 
                        "name": name, 
                        "emotion": emotion, 
                        "head_pose": head_pose,
                        "eye_state": eye_state,
                    })

                    # Rate-limit updates per student.
                    if now - last_update.get(name, 0.0) >= UPDATE_INTERVAL_SECONDS:
                        last_update[name] = now
                        # POST update to backend.
                        try:
                            requests.post(
                                f"{backend_url}/update",
                                json={
                                    "student_name": name,
                                    "emotion": emotion,
                                    "head_pose": head_pose,
                                    "eye_state": eye_state,
                                    "timestamp": now,
                                },
                                timeout=2.0,
                            )
                        except requests.RequestException as exc:
                            print(f"[VideoProcessor] Backend error: {exc}")

            # Draw annotated bounding box and labels using last_detections.
            for det in last_detections:
                loc = det["loc"]
                top, right, bottom, left = [int(c / DETECTION_SCALE) for c in loc]
                
                if det["name"] == "unknown":
                    cv2.rectangle(frame, (left, top), (right, bottom), (128, 128, 128), 1)
                    cv2.putText(
                        frame, "Unknown",
                        (left, max(20, top - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1,
                    )
                else:
                    name = det["name"]
                    emotion = det["emotion"]
                    head_pose = det["head_pose"]
                    eye_state = det.get("eye_state", "open")
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 220, 0), 2)
                    cv2.putText(
                        frame,
                        f"{name}: {emotion} [{head_pose}] eyes:{eye_state}",
                        (left, max(20, top - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2,
                    )

            cv2.imshow("Classroom Monitor  (press q to quit)", frame)
            if cv2.waitKey(1) == ord("q"):
                break
                
            frame_count += 1

    finally:
        capture.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classroom video processor — face recognition, emotion detection, head pose."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        metavar="N",
        help="OpenCV VideoCapture camera index (default: 0).",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default=DEFAULT_BACKEND_URL,
        metavar="URL",
        help=f"Base URL of the classroom backend (default: {DEFAULT_BACKEND_URL}).",
    )
    args = parser.parse_args()
    run(camera_index=args.camera_index, backend_url=args.backend_url)


if __name__ == "__main__":
    main()
