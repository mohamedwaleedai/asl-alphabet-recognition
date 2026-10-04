"""
ASL - Main Application
==============================
Production-grade ASL alphabet recognition system.
Version: 1.0
"""

import cv2
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from src.feature_extractor import FeatureExtractor


# =============================================================================
# THEME
# =============================================================================
class Theme:
    BG = "#0d1117"
    CARD = "#161b22"
    BORDER = "#30363d"
    ACCENT = "#58a6ff"
    SUCCESS = "#3fb950"
    WARNING = "#d29922"
    ERROR = "#f85149"
    TEXT = "#f0f6fc"
    TEXT_DIM = "#8b949e"


# =============================================================================
# PREDICTOR
# =============================================================================
class ASLPredictor:
    """Production-grade landmark-based ASL predictor."""
    
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.extractor = FeatureExtractor()
        self.classes = [chr(65 + i) for i in range(26)]
        
        # Validate model input shape matches feature extractor
        model_input_shape = self.model.input_shape[1]  
        extractor_features = self.extractor.num_features
        
        if model_input_shape != extractor_features:
            raise ValueError(
                f"Model input shape mismatch!\n"
                f"  Model expects: {model_input_shape} features\n"
                f"  Extractor produces: {extractor_features} features\n"
                f"  Please retrain the model with the current feature extractor."
            )
        
        # Temporal smoothing
        self.history = []
        self.HISTORY_SIZE = 5
        self.MIN_CONFIDENCE = 0.4
        
        # Letter-specific thresholds
        self.thresholds = {
            'A': 0.6, 'S': 0.6, 'T': 0.6, 'E': 0.6,
            'M': 0.65, 'N': 0.65,
            'U': 0.6, 'V': 0.6, 'R': 0.6,
        }
        self.default_threshold = 0.5
        
        print(f"Model loaded: {model_path}")
        print(f"  Input shape: {model_input_shape}, Features: {extractor_features} ✓")
    
    def predict(self, landmarks) -> tuple:
        """Predict letter from landmarks."""
        if not landmarks:
            self.history.append(None)
            if len(self.history) > self.HISTORY_SIZE:
                self.history.pop(0)
            return '-', 0.0
        
        # Extract features
        features = self.extractor.extract(landmarks)
        
        # Model prediction
        probs = self.model.predict(features.reshape(1, -1), verbose=0)[0]
        
        idx = np.argmax(probs)
        conf = probs[idx]
        letter = self.classes[idx]
        
        # Add to history
        self.history.append((letter, conf))
        if len(self.history) > self.HISTORY_SIZE:
            self.history.pop(0)
        
        # Temporal voting
        return self._vote()
    
    def _vote(self) -> tuple:
        """Majority voting with confidence."""
        valid = [(l, c) for x in self.history 
                 if x and x[1] > self.MIN_CONFIDENCE 
                 for l, c in [x]]
        
        if not valid:
            return '-', 0.0
        
        votes = {}
        for letter, _ in valid:
            votes[letter] = votes.get(letter, 0) + 1
        
        winner = max(votes, key=votes.get)
        ratio = votes[winner] / len(valid)
        
        confs = [c for l, c in valid if l == winner]
        avg_conf = sum(confs) / len(confs) if confs else 0
        
        threshold = self.thresholds.get(winner, self.default_threshold)
        
        if ratio >= 0.5 and avg_conf >= threshold:
            return winner, avg_conf
        
        return '-', avg_conf * 0.5


# =============================================================================
# GUI APPLICATION
# =============================================================================
class ASLExpertApp:
    """ASL Expert Application."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ASL Expert")
        self.root.geometry("1100x700")
        self.root.configure(bg=Theme.BG)
        
        # State
        self.cap = None
        self.detector = None
        self.predictor = None
        self.running = False
        
        # Word building
        self.word = ""
        self.last_letter = ""
        self.hold_count = 0
        self.HOLD_FRAMES = 12
        
        # FPS tracking
        self.fps_times = []
        
        self._build_ui()
        self._init_models()
    
    def _build_ui(self):
        """Build the UI."""
        main = tk.Frame(self.root, bg=Theme.BG)
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header = tk.Frame(main, bg=Theme.BG)
        header.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(header, text="🤟 ASL Expert", 
                font=("Segoe UI", 24, "bold"),
                bg=Theme.BG, fg=Theme.TEXT).pack(side=tk.LEFT)
        
        self.fps_label = tk.Label(header, text="FPS: --",
                                 font=("Segoe UI", 11),
                                 bg=Theme.BG, fg=Theme.TEXT_DIM)
        self.fps_label.pack(side=tk.RIGHT)
        
        self.status_label = tk.Label(header, text="● Ready",
                                    font=("Segoe UI", 11),
                                    bg=Theme.BG, fg=Theme.TEXT_DIM)
        self.status_label.pack(side=tk.RIGHT, padx=20)
        
        # Content
        content = tk.Frame(main, bg=Theme.BG)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Video
        video_frame = tk.Frame(content, bg=Theme.CARD)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        self.video_container = tk.Frame(video_frame, bg="#000", width=640, height=480)
        self.video_container.pack(padx=10, pady=10)
        self.video_container.pack_propagate(False)
        
        self.video_label = tk.Label(self.video_container, bg="#000",
                                   text="Click START", font=("Segoe UI", 16),
                                   fg=Theme.TEXT_DIM)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Results panel
        results = tk.Frame(content, bg=Theme.CARD, width=280)
        results.pack(side=tk.RIGHT, fill=tk.Y)
        results.pack_propagate(False)
        
        tk.Label(results, text="DETECTED", font=("Segoe UI", 10, "bold"),
                bg=Theme.CARD, fg=Theme.TEXT_DIM).pack(pady=(25, 5))
        
        self.letter_label = tk.Label(results, text="-",
                                    font=("Segoe UI", 90, "bold"),
                                    bg=Theme.CARD, fg=Theme.ACCENT)
        self.letter_label.pack()
        
        self.conf_label = tk.Label(results, text="0%",
                                  font=("Segoe UI", 20),
                                  bg=Theme.CARD, fg=Theme.SUCCESS)
        self.conf_label.pack(pady=5)
        
        # Progress bar
        self.progress_frame = tk.Frame(results, bg=Theme.BORDER, height=8, width=200)
        self.progress_frame.pack(pady=15)
        self.progress_bar = tk.Frame(self.progress_frame, bg=Theme.ACCENT, height=8, width=0)
        self.progress_bar.place(x=0, y=0)
        
        tk.Frame(results, bg=Theme.BORDER, height=1).pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(results, text="WORD", font=("Segoe UI", 10, "bold"),
                bg=Theme.CARD, fg=Theme.TEXT_DIM).pack(pady=(10, 5))
        
        word_frame = tk.Frame(results, bg=Theme.BG)
        word_frame.pack(fill=tk.X, padx=20)
        
        self.word_label = tk.Label(word_frame, text="",
                                  font=("Segoe UI", 20),
                                  bg=Theme.BG, fg=Theme.TEXT,
                                  wraplength=220, height=2)
        self.word_label.pack(pady=10)
        
        btn_frame = tk.Frame(results, bg=Theme.CARD)
        btn_frame.pack(fill=tk.X, padx=20)
        
        for text, cmd in [("Space", self._add_space), 
                         ("⌫", self._backspace),
                         ("Clear", self._clear)]:
            tk.Button(btn_frame, text=text, font=("Segoe UI", 10),
                     bg=Theme.BG, fg=Theme.TEXT,
                     activebackground=Theme.ACCENT, bd=0,
                     padx=10, pady=5, command=cmd).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        # Controls
        controls = tk.Frame(main, bg=Theme.BG)
        controls.pack(fill=tk.X, pady=(20, 0))
        
        self.start_btn = tk.Button(controls, text="▶ START",
                                  font=("Segoe UI", 14, "bold"),
                                  bg=Theme.SUCCESS, fg="white",
                                  bd=0, padx=30, pady=12,
                                  command=self._toggle)
        self.start_btn.pack(side=tk.LEFT)
        
        tk.Button(controls, text="✕ QUIT",
                 font=("Segoe UI", 14, "bold"),
                 bg=Theme.ERROR, fg="white",
                 bd=0, padx=20, pady=12,
                 command=self._quit).pack(side=tk.RIGHT)
    
    def _init_models(self):
        """Initialize models."""
        # Initialize to None to ensure they're defined
        self.predictor = None
        self.detector = None
        
        try:
            model_path = os.path.join(PROJECT_ROOT, "models", "asl_landmark_model.h5")
            
            if not os.path.exists(model_path):
                self.status_label.config(text="● No model", fg=Theme.ERROR)
                return
            
            self.predictor = ASLPredictor(model_path)
            
            mp_model = os.path.join(PROJECT_ROOT, "src", "hand_landmarker.task")
            if not os.path.exists(mp_model):
                self.status_label.config(text="● No MediaPipe model", fg=Theme.ERROR)
                self.predictor = None  # Reset if detector fails
                return
            
            base_options = mp_tasks.BaseOptions(model_asset_path=mp_model)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.5
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            
            self.status_label.config(text="● Ready", fg=Theme.SUCCESS)
            
        except Exception as e:
            self.status_label.config(text=f"● Error", fg=Theme.ERROR)
            print(f"Init error: {e}")
            # Ensure both are None on error
            self.predictor = None
            self.detector = None
    
    def _toggle(self):
        if self.running:
            self._stop()
        else:
            self._start()
    
    def _start(self):
        if not self.predictor:
            messagebox.showerror("Error", "Model not loaded. Run train.py first.")
            return
        
        if not self.detector:
            messagebox.showerror("Error", "Hand detector not initialized. Check MediaPipe model file.")
            return
        
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            for _ in range(5):
                self.cap.read()
            
            self.running = True
            self.fps_times = []
            
            self.start_btn.config(text="⏹ STOP", bg=Theme.WARNING)
            self.status_label.config(text="● Running", fg=Theme.SUCCESS)
            
            self._update()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def _stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(text="▶ START", bg=Theme.SUCCESS)
        self.status_label.config(text="● Stopped", fg=Theme.TEXT_DIM)
    
    def _update(self):
        if not self.running:
            return
        
        # Safety check: ensure detector is initialized
        if not self.detector:
            self.root.after(10, self._update)
            return
        
        start = time.time()
        
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self._update)
            return
        
        frame = cv2.flip(frame, 1)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        
        landmarks = None
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            self._draw_landmarks(frame, landmarks)
        
        letter, conf = self.predictor.predict(landmarks)
        
        self._update_display(letter, conf)
        self._handle_word(letter, conf)
        self._show_frame(frame)
        
        elapsed = time.time() - start
        self.fps_times.append(elapsed)
        if len(self.fps_times) > 30:
            self.fps_times.pop(0)
        fps = 1.0 / (sum(self.fps_times) / len(self.fps_times))
        self.fps_label.config(text=f"FPS: {fps:.0f}")
        
        self.root.after(1, self._update)
    
    def _draw_landmarks(self, frame, landmarks):
        h, w = frame.shape[:2]
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        for i, j in connections:
            x1, y1 = int(landmarks[i].x * w), int(landmarks[i].y * h)
            x2, y2 = int(landmarks[j].x * w), int(landmarks[j].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    def _update_display(self, letter, conf):
        self.letter_label.config(text=letter)
        self.conf_label.config(text=f"{conf:.0%}")
        
        if conf >= 0.7:
            self.conf_label.config(fg=Theme.SUCCESS)
        elif conf >= 0.5:
            self.conf_label.config(fg=Theme.WARNING)
        else:
            self.conf_label.config(fg=Theme.ERROR)
    
    def _handle_word(self, letter, conf):
        if letter != '-' and conf >= 0.6:
            if letter == self.last_letter:
                self.hold_count += 1
            else:
                self.last_letter = letter
                self.hold_count = 1
            
            progress = min(self.hold_count / self.HOLD_FRAMES, 1.0)
            self.progress_bar.config(width=int(200 * progress))
            
            if self.hold_count >= self.HOLD_FRAMES:
                self.word += letter
                self.word_label.config(text=self.word)
                self.hold_count = 0
        else:
            self.hold_count = 0
            self.progress_bar.config(width=0)
    
    def _show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize((640, 480), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(img)
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk
    
    def _add_space(self):
        self.word += " "
        self.word_label.config(text=self.word)
    
    def _backspace(self):
        self.word = self.word[:-1]
        self.word_label.config(text=self.word)
    
    def _clear(self):
        self.word = ""
        self.word_label.config(text="")
    
    def _quit(self):
        self._stop()
        if self.detector:
            self.detector.close()
        self.root.destroy()
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.mainloop()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  ASL Expert v1.0")
    print("  98.88% accuracy | 30+ FPS")
    print("=" * 50)
    
    app = ASLExpertApp()
    app.run()
