import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_face(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(face_rgb)
    tensor = transform(img).unsqueeze(0)
    return tensor


class PredictionSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = []

    def update(self, emotion):
        self.history.append(emotion)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        values, counts = np.unique(self.history, return_counts=True)
        return values[np.argmax(counts)]

    def clear(self):
        self.history = []


class StableEmotionTrigger:
    def __init__(self, stable_count=3):
        self.stable_count = stable_count
        self.last_emotion = None
        self.count = 0

    def update(self, emotion):
        if emotion == self.last_emotion:
            self.count += 1
        else:
            self.last_emotion = emotion
            self.count = 1
        return self.count >= self.stable_count

    def clear(self):
        self.last_emotion = None
        self.count = 0
