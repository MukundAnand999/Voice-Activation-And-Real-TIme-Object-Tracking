import threading
import time
import cv2
import numpy as np
import speech_recognition as sr
from ultralytics import YOLO
import pyttsx3
import datetime
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QTextEdit, QPushButton)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5 import QtGui
from PyQt5.QtGui import QPalette, QColor

class ObjectTrackerGUI(QWidget):
    update_text_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.engine = pyttsx3.init('sapi5')
        self.engine.setProperty('rate', 150)
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[0].id)
        self.model = YOLO('yolov8s.pt')
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.target_object_name = None
        self.is_tracking = False
        self.speech_thread = None
        self.cap = None
        self.initUI()
        self.wishMe()
        self.update_text_signal.connect(self.update_text_edit)
        self.speak("HI Master Mukund")
        # self.speak("This is a Voice activation real time object tracking Project")
        self.focal_length = 500  # Adjust based on your camera
        self.real_object_width = 0.2  # Approximate width in meters (adjust)

    def update_text_edit(self, text):
        self.input_edit.setPlainText(text)

    def speak(self, audio):
        self.engine.say(audio)
        self.engine.runAndWait()

    def wishMe(self):
        hour = datetime.datetime.now().hour
        if 0 <= hour < 12:
            self.speak("Good morning!")
        elif 12 <= hour < 18:
            self.speak("Good afternoon!")
        else:
            self.speak("Good evening!")

    def initUI(self):
        self.setWindowTitle('Object Tracker')
        self.setGeometry(200, 200, 1000, 800)

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(0, 0, 0))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 0))
        palette.setColor(QPalette.Base, QColor(0, 0, 0))
        palette.setColor(QPalette.Text, QColor(255, 255, 0))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        main_layout = QVBoxLayout()
        input_layout = QHBoxLayout()
        self.input_label = QLabel("Enter object name:")
        self.input_edit = QTextEdit()
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_edit)
        main_layout.addLayout(input_layout)
        button_layout = QHBoxLayout()
        self.track_button = QPushButton("Track Object (Text)")
        self.track_button.clicked.connect(self.start_tracking_text)
        self.voice_button = QPushButton("Track Object (Voice)")
        self.voice_button.clicked.connect(self.start_tracking_voice)
        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close_application)

        button_layout.addWidget(self.track_button)
        button_layout.addWidget(self.voice_button)
        button_layout.addWidget(self.exit_button)

        main_layout.addLayout(button_layout)
        self.video_frame = QLabel()
        self.video_frame.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.video_frame)
        self.setLayout(main_layout)

    def close_application(self):
        self.close()

    def start_tracking_text(self):
        self.target_object_name = self.input_edit.toPlainText().strip()
        if self.target_object_name:
            self.start_video_tracking()
        else:
            self.speak("Please enter an object name.")

    def start_tracking_voice(self):
        if self.speech_thread is None or not self.speech_thread.is_alive():
            self.speech_thread = threading.Thread(target=self.recognize_speech_input)
            self.speech_thread.start()
        self.start_video_tracking()

    def recognize_speech_input(self):
        with self.microphone as source:
            print("Say the name of the object you want to track:")
            try:
                audio = self.recognizer.listen(source)
                command = self.recognizer.recognize_google(audio)
                if command.lower().startswith("track"):
                    self.target_object_name = command.split("track")[1].strip()
                    print(f"Tracking object: {self.target_object_name}")
                    self.update_text_signal.emit(self.target_object_name)
                    self.speak(f"Tracking {self.target_object_name}")
                else:
                    print("Please say 'track' followed by the name of the object.")
            except sr.UnknownValueError:
                print("Could not understand the audio")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
            except Exception as e:
                print(f"An unexpected error occurred during voice input: {e}")

    def estimate_distance(self, box_width):
        if box_width == 0:
            return "Unknown"
        distance = (self.real_object_width * self.focal_length) / box_width
        return f"{distance:.2f} meters"

    def start_video_tracking(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.speak("Cannot open camera")
                return
        self.is_tracking = True
        while self.is_tracking and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(img)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f'{self.model.names[cls]} {conf:.2f}'
                    if self.target_object_name and self.model.names[cls].lower() == self.target_object_name.lower():
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        box_width = x2 - x1
                        distance = self.estimate_distance(box_width)
                        cv2.putText(frame, f"Distance: {distance}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        self.update_text_signal.emit(f"Tracking {self.target_object_name}. Distance: {distance}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = np.require(frame, np.uint8, 'C')
            q_img = q_img.data
            qt_img = QtGui.QImage(q_img, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qt_img)
            self.video_frame.setPixmap(pixmap)
            QApplication.processEvents()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_tracking = False
                break
        if not self.is_tracking:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ObjectTrackerGUI()
    gui.show()
    sys.exit(app.exec_())


#Number of listed items to track 
'''
Person, Bicycle, Car, Motorcycle, Airplane, Bus,Train, Truck, Boat, Traffic light
,Fire hydrant, Stop sign, Parking meter, Bench, Bird, Cat, Dog, Horse, Sheep, Cow
Elephant, Bear, Zebra, Giraffe, Backpack, Umbrella, Handbag, Tie, Suitcase, Frisbee
Skis, Snowboard, Sports ball, Kite, Baseball bat, Baseball glove, Skateboard, Surfboard
Tennis racket, Bottle, Wine glass, Cup, Fork, Knife, Spoon, Bowl, Banana, Apple, Sandwich
Orange, Broccoli, Carrot, Hot dog, Pizza, Donut, Cake, Chair, Couch, Potted plant, Bed
Dining table, Toilet, TV, Laptop, Mouse, Remote, Keyboard, Cell phone, Microwave, Oven
Toaster, Sink, Refrigerator, Book, Clock, Vase, Scissors, Teddy bear, Hair drier, Toothbrush
'''




