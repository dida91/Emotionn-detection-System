import argparse

import cv2

from emotion_model import EmotionModel
from face_detection import FaceDetector

LABEL_MIN_Y_POSITION = 20
LABEL_LINE_HEIGHT = 22

EMOTION_DISPLAY_ORDER = (
    "happy",
    "sad",
    "angry",
    "surprised",
    "fear",
    "disgust",
    "neutral",
)


def run(camera_index: int = 0) -> None:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to access webcam at index {camera_index}.")

    face_detector = FaceDetector()
    emotion_model = EmotionModel()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            faces = face_detector.detect_faces(frame)
            for (x, y, w, h) in faces:
                # Guard against occasional boundary overflow on partial edge detections.
                pad_x = int(w * 0.2)
                pad_y = int(h * 0.2)
                x0 = max(0, x - pad_x)
                y0 = max(0, y - pad_y)
                x1 = min(frame.shape[1], x + w + pad_x)
                y1 = min(frame.shape[0], y + h + pad_y)
                face_roi = frame[y0:y1, x0:x1]

                if face_roi.size == 0:
                    continue

                emotion, confidence, all_scores = emotion_model.predict_emotions(face_roi)
                top_label = f"{emotion}: {confidence:.2f}"
                all_labels = " ".join(
                    f"{label[:3]}:{all_scores[label]:.2f}" for label in EMOTION_DISPLAY_ORDER
                )
                label_y = max(LABEL_MIN_Y_POSITION, y0 - 10)

                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    top_label,
                    (x0, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    all_labels,
                    (x0, label_y + LABEL_LINE_HEIGHT),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                )

            cv2.imshow("Student Emotion Detection", frame)
            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Student Emotion Detection")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index for OpenCV VideoCapture (default: 0)",
    )
    args = parser.parse_args()
    run(camera_index=args.camera_index)


if __name__ == "__main__":
    main()
