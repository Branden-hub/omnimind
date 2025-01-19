"""
OmniMind: Interaction systems
"""

import cv2
import pyttsx3
import speech_recognition as sr
import mss
import numpy as np
import threading

class OmniMindInteractions:
    def __init__(self):
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        # Flags for camera and screen
        self.camera_running = False
        self.screen_running = False
        # Threads
        self.camera_thread = None
        self.screen_thread = None

    def text_to_speech(self, text):
        """Convert text to speech."""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def handle_voice_command(self, command):
        """Handle voice commands to control camera and screen."""
        if "start camera" in command.lower():
            if not self.camera_running:
                self.start_camera_thread()
                self.text_to_speech("Camera started.")
            else:
                self.text_to_speech("Camera is already running.")
        elif "stop camera" in command.lower():
            if self.camera_running:
                self.camera_running = False
                self.text_to_speech("Camera stopped.")
            else:
                self.text_to_speech("Camera is not running.")
        elif "start screen" in command.lower():
            if not self.screen_running:
                self.start_screen_thread()
                self.text_to_speech("Screen capture started.")
            else:
                self.text_to_speech("Screen capture is already running.")
        elif "stop screen" in command.lower():
            if self.screen_running:
                self.screen_running = False
                self.text_to_speech("Screen capture stopped.")
            else:
                self.text_to_speech("Screen capture is not running.")

    def speech_to_text(self):
        """Convert speech to text."""
        with sr.Microphone() as source:
            while True:
                print("Listening...")
                audio = self.recognizer.listen(source)
                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"You said: {text}")
                    self.handle_voice_command(text)
                except sr.UnknownValueError:
                    print("Sorry, I did not understand that.")
                except sr.RequestError:
                    print("Could not request results; check your network connection.")

    def access_camera(self):
        """Access and display the device camera feed."""
        self.camera_running = True
        cap = cv2.VideoCapture(0)
        while self.camera_running:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Camera Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def capture_screen(self):
        """Capture and display the device screen."""
        self.screen_running = True
        with mss.mss() as sct:
            while self.screen_running:
                screenshot = sct.shot()
                img = cv2.imread(screenshot)
                cv2.imshow('Screen Capture', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()

    def start_camera_thread(self):
        """Start a thread for camera access."""
        if not self.camera_running:
            self.camera_thread = threading.Thread(target=self.access_camera)
            self.camera_thread.start()

    def start_screen_thread(self):
        """Start a thread for screen capture."""
        if not self.screen_running:
            self.screen_thread = threading.Thread(target=self.capture_screen)
            self.screen_thread.start()
