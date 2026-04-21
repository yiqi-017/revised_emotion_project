import threading
import time

import pyttsx3


class SpeechEngine:
    def __init__(self, cooldown: int = 5):
        self.engine = pyttsx3.init()
        self.last_emotion = None
        self.last_time = 0.0
        self.cooldown = cooldown
        self._lock = threading.Lock()
        self._speaking = False

        self.text_map = {
            "happy": "You look happy. Keep it up!",
            "sad": "You seem a little sad. Take a short break.",
            "angry": "You look upset. Try to relax and breathe deeply.",
            "surprise": "You look surprised. Did something happen?",
            "fear": "You seem worried. Take a slow breath.",
            "disgust": "You seem uncomfortable. Maybe step away for a moment.",
        }

    def _speak_worker(self, text: str):
        with self._lock:
            self._speaking = True
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            finally:
                self._speaking = False

    def speak_emotion(self, emotion: str):
        now = time.time()
        text = self.text_map.get(emotion)
        if not text:
            return
        if self._speaking:
            return
        if self.last_emotion == emotion and now - self.last_time < self.cooldown:
            return

        self.last_emotion = emotion
        self.last_time = now
        thread = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
        thread.start()
