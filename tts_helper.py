import subprocess
import threading
import time
import winreg


class SpeechEngine:
    def __init__(self, cooldown: int = 12, global_cooldown: int = 6):
        self.cooldown = cooldown
        self.global_cooldown = global_cooldown
        self.last_emotion = None
        self.last_time = 0.0
        self.last_spoken_time = 0.0
        self._lock = threading.Lock()
        self._speaking = False
        self._warned_unavailable = False
        self.voice_name = None
        self.has_us_english_voice = False
        self.ps_timeout = 10

        self.text_map = {
            "happy": "You look happy. Enjoy it, but stay grounded.",
            "sad": "You seem a little down. Take a breath. Things will get better.",
            "angry": "You look upset. Slow down and take a deep breath.",
            "surprise": "You look surprised. Take a second and stay calm.",
            "fear": "You seem nervous. Breathe slowly. You're okay.",
            "disgust": "You look uncomfortable. Take a moment to reset.",
            "neutral": "You look calm and steady. Keep it going.",
        }
        self._init_voice()

    def _load_windows_voices(self):
        voices = []
        reg_path = r"SOFTWARE\Microsoft\Speech\Voices\Tokens"
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as root_key:
                count = winreg.QueryInfoKey(root_key)[0]
                for index in range(count):
                    token_name = winreg.EnumKey(root_key, index)
                    with winreg.OpenKey(root_key, token_name) as voice_key:
                        voice_name = winreg.QueryValueEx(voice_key, None)[0]
                        voices.append((token_name, voice_name))
        except OSError:
            return []
        return voices

    def _init_voice(self):
        voices = self._load_windows_voices()
        preferred_voice = None

        for token_name, voice_name in voices:
            voice_info = f"{token_name} {voice_name}".lower()
            if "en-us" in voice_info or "united states" in voice_info or "zira" in voice_info:
                preferred_voice = voice_name
                self.has_us_english_voice = True
                break

        if preferred_voice is None and voices:
            preferred_voice = voices[0][1]

        self.voice_name = preferred_voice or "unknown"
        print(f"Speech engine ready. voice={self.voice_name}")
        if not self.has_us_english_voice:
            print(
                "Speech warning: no US English voice found in Windows SAPI. "
                "The current voice may not sound like standard American English."
            )

    def _ps_quote(self, text: str) -> str:
        return "'" + text.replace("'", "''") + "'"

    def _build_ps_script(self, text: str) -> str:
        quoted_text = self._ps_quote(text)
        quoted_voice = self._ps_quote(self.voice_name) if self.voice_name != "unknown" else "$null"
        return (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"if ({quoted_voice} -ne $null) {{ "
            f"try {{ $speaker.SelectVoice({quoted_voice}) }} catch {{ }} "
            "}; "
            "$speaker.Rate = 1; "
            f"$speaker.Speak({quoted_text})"
        )

    def _speak_worker(self, text: str):
        with self._lock:
            self._speaking = True
            try:
                subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        self._build_ps_script(text),
                    ],
                    check=False,
                    timeout=self.ps_timeout,
                    capture_output=True,
                    text=True,
                )
            except subprocess.TimeoutExpired:
                print("Speech timeout: the voice process was reset.")
            finally:
                self._speaking = False

    def speak_emotion(self, emotion: str):
        now = time.time()
        text = self.text_map.get(emotion)
        if not text:
            print(f"Speech skipped: no text configured for emotion={emotion}")
            return
        if self._speaking:
            print(f"Speech skipped: engine busy, emotion={emotion}")
            return
        if now - self.last_spoken_time < self.global_cooldown:
            print(f"Speech skipped: global cooldown, emotion={emotion}")
            return
        if self.last_emotion == emotion and now - self.last_time < self.cooldown:
            print(f"Speech skipped: same emotion cooldown, emotion={emotion}")
            return

        self.last_emotion = emotion
        self.last_time = now
        self.last_spoken_time = now
        print(f"Speech triggered: emotion={emotion}, text={text}")
        thread = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
        thread.start()
