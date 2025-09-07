import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import os
from typing import Union, Tuple

class FacialEmotionDetector:
    """
    A class to detect facial emotions from an image or video frame using a YOLO model.
    """
    def __init__(self, model_path='best.pt'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                f"Please ensure the YOLO model is in the correct directory."
            )
        self.model = YOLO(model_path)
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        print("FacialEmotionDetector initialized successfully.")

    def detect_emotion(self, frame: np.ndarray) -> Tuple[np.ndarray, Union[str, None]]:
        result = self.model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(result)

        dominant_emotion = None
        if len(detections) > 0:
            most_confident_idx = np.argmax(detections.confidence)
            dominant_emotion = detections.data['class_name'][most_confident_idx]

        labels = [
            f"{self.model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _ in detections
        ]

        annotated_frame = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        return annotated_frame, dominant_emotion
