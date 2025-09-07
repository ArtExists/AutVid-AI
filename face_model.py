import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import os
from typing import Union, Tuple


class FacialEmotionDetector:
    """
    Detect facial emotions from an image or video frame using a YOLO model.
    """

    def __init__(self, model_path: str = "best.pt"):
        """
        Initialize the detector.

        Args:
            model_path (str): Path to YOLO model weights (.pt).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"❌ Model file not found at '{model_path}'. "
                f"Please ensure 'best.pt' is available in the project directory."
            )

        # Load YOLO model
        self.model = YOLO(model_path)

        # Supervision annotators for boxes + labels
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

        print("✅ FacialEmotionDetector initialized successfully.")

    def detect_emotion(self, frame: np.ndarray) -> Tuple[np.ndarray, Union[str, None]]:
        """
        Detect emotions in a single frame.

        Args:
            frame (np.ndarray): BGR image (OpenCV).

        Returns:
            Tuple[np.ndarray, str|None]:
                - Annotated frame (with boxes + labels).
                - Most confident emotion label (or None if no detection).
        """
        # YOLO inference
        result = self.model(frame, agnostic_nms=True)[0]

        # Convert YOLO results → Supervision detections
        detections = sv.Detections.from_ultralytics(result)

        # Find dominant (highest confidence) detection
        dominant_emotion = None
        if len(detections) > 0:
            most_confident_idx = np.argmax(detections.confidence)
            dominant_emotion = detections.data["class_name"][most_confident_idx]

        # Build label strings
        labels = [
            f"{self.model.model.names[class_id]} {confidence:.2f}"
            for _, _, confidence, class_id, _, _ in detections
        ]

        # Annotate boxes
        annotated = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        # Annotate labels
        annotated = self.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        return annotated, dominant_emotion


if __name__ == "__main__":
    # Quick webcam test
    try:
        detector = FacialEmotionDetector(model_path="best.pt")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Could not open webcam.")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated_frame, emotion = detector.detect_emotion(frame)
                cv2.imshow("Facial Emotion Detection", annotated_frame)

                if emotion:
                    print(f"Detected Emotion: {emotion}")

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
