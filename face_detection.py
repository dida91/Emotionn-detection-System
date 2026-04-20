import cv2


class FaceDetector:
    def __init__(self) -> None:
        classifier_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(classifier_path)

    def detect_faces(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
        )
