import cv2
import numpy as np
from fer import FER

SUPPORTED_LABELS = {
    "happy",
    "sad",
    "angry",
    "surprised",
    "neutral",
    "fear",
    "disgust",
}

LABEL_MAP = {
    "surprise": "surprised",
}


class EmotionModel:
    def __init__(self) -> None:
        self.detector = FER(mtcnn=True)

    def predict_emotions(self, face_image: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """Predict dominant emotion and return all emotion scores from a BGR face crop."""
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        scores = {label: 0.0 for label in SUPPORTED_LABELS}
        detections = self.detector.detect_emotions(rgb_face)

        if detections:
            detected_emotions = detections[0].get("emotions", {})
            for label, value in detected_emotions.items():
                mapped_label = LABEL_MAP.get(label, label)
                if mapped_label in scores:
                    scores[mapped_label] = float(value or 0.0)

        dominant_label = max(scores, key=scores.get)
        dominant_score = float(scores[dominant_label])

        if dominant_score <= 0.0:
            return "neutral", 0.0, scores

        return dominant_label, dominant_score, scores
