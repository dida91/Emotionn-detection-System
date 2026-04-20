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
        self.detector = FER(mtcnn=False)

    def predict_emotion(self, face_image: np.ndarray) -> tuple[str, float]:
        """Predict emotion from a single BGR face crop and return label/confidence."""
        # FER2013-based emotion models are trained on 48x48 facial inputs.
        resized_face = cv2.resize(face_image, (48, 48))
        rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
        prediction = self.detector.top_emotion(rgb_face)

        if not prediction:
            return "neutral", 0.0

        label, confidence = prediction
        if label is None:
            return "neutral", 0.0

        mapped_label = LABEL_MAP.get(label, label)
        if mapped_label not in SUPPORTED_LABELS:
            return "neutral", float(confidence or 0.0)

        return mapped_label, float(confidence or 0.0)
