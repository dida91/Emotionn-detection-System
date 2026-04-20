import argparse

import cv2
from fer import FER


def run_emotion_detection(camera_index: int = 0) -> None:
    detector = FER(mtcnn=True)
    capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        raise RuntimeError(f"Unable to access webcam at index {camera_index}.")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect_emotions(rgb_frame)

            for detected_face in detections:
                x, y, w, h = detected_face["box"]
                emotions = detected_face.get("emotions", {})
                if not emotions:
                    continue

                top_emotion = max(emotions, key=emotions.get)
                confidence = emotions[top_emotion]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{top_emotion}: {confidence:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Student Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time student emotion detection")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index for OpenCV VideoCapture (default: 0)",
    )
    args = parser.parse_args()
    run_emotion_detection(camera_index=args.camera_index)


if __name__ == "__main__":
    main()
