from pathlib import Path

import cv2
import torch

from model import EmotionCNN
from tts_helper import SpeechEngine
from utils import PredictionSmoother, StableEmotionTrigger, preprocess_face


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "emotion_model.pth"
CONF_THRESHOLD = 0.70
CAMERA_ID = 0


def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH.resolve()}\n"
            "Please run train.py first, or copy emotion_model.pth into checkpoints/."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    classes = checkpoint.get("classes")
    if not classes:
        raise ValueError("Checkpoint is missing 'classes'.")

    model = EmotionCNN(num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes, device


def build_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade from: {cascade_path}")
    return detector


def detect_largest_face(detector, frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80),
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


@torch.no_grad()
def predict_emotion(model, classes, device, face_bgr):
    tensor = preprocess_face(face_bgr).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_idx].item()
    return classes[pred_idx], confidence


def main():
    model, classes, device = load_model()
    detector = build_face_detector()
    speech_engine = SpeechEngine(cooldown=15, global_cooldown=8)
    smoother = PredictionSmoother(window_size=5)
    stable_trigger = StableEmotionTrigger(stable_count=3)
    last_debug_state = None

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera permissions or CAMERA_ID.")

    print("Press 'q' to quit.")
    print(
        f"Speech rule: conf>={CONF_THRESHOLD}, stable_count>=3, "
        "then try to speak."
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        face_box = detect_largest_face(detector, frame)
        if face_box is not None:
            x, y, w, h = face_box
            x = max(0, x)
            y = max(0, y)
            face = frame[y:y + h, x:x + w]

            if face.size > 0:
                emotion, conf = predict_emotion(model, classes, device, face)
                smooth_emotion = smoother.update(emotion)
                is_stable = stable_trigger.update(smooth_emotion)

                color = (0, 255, 0) if conf >= CONF_THRESHOLD else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    f"raw={emotion} {conf:.2f}",
                    (x, max(y - 30, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"stable={smooth_emotion}",
                    (x, max(y - 5, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                debug_state = (emotion, smooth_emotion, round(conf, 2), is_stable)
                if debug_state != last_debug_state:
                    print(
                        "Detect:",
                        f"raw={emotion}",
                        f"stable={smooth_emotion}",
                        f"conf={conf:.2f}",
                        f"is_stable={is_stable}",
                    )
                    last_debug_state = debug_state

                if conf >= CONF_THRESHOLD and is_stable:
                    speech_engine.speak_emotion(smooth_emotion)
        else:
            smoother.clear()
            stable_trigger.clear()
            if last_debug_state is not None:
                print("Detect: no face")
                last_debug_state = None
            cv2.putText(
                frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
