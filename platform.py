import sys  # Import system-specific parameters and functions
import os  # Import operating system interface
import pyaudio  # Import PyAudio for handling audio streams
import wave  # Import wave module for reading and writing WAV files

# Set the Qt platform plugin path
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms"
# Set the Qt plugin path
os.environ["QT_PLUGIN_PATH"] = "/usr/lib/aarch64-linux-gnu/qt5/plugins"
# os.environ["QT_QPA_PLAT9FORM"] = "minimal"  # This line is commented out

from picamera2 import Picamera2  # Import Picamera2 for camera access

import cv2  # Import OpenCV for image processing
import csv  # Import CSV module for handling CSV files
import re  # Import regular expressions module
import whisper  # Import Whisper for speech recognition
import mediapipe as mp  # Import MediaPipe for hand tracking
import numpy as np  # Import NumPy for numerical operations
import pandas as pd  # Import pandas for data manipulation
import joblib  # Import joblib for saving and loading models
import threading  # Import threading for handling threads

from collections import deque  # Import deque from collections for efficient queue operations
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting data
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Import preprocessing tools
from tensorflow.keras.models import Sequential, load_model  # Import Keras models
from tensorflow.keras.layers import Dense, Dropout, Input  # Import Keras layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Import Keras callbacks
from tensorflow.keras.optimizers import Adam  # Import Adam optimizer
import pyttsx3  # Import pyttsx3 for text-to-speech
from gtts import gTTS  # Import gTTS for Google Text-to-Speech
from playsound import playsound  # Import playsound to play sound files
import time  # Import time module for time-related functions

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit,
                             QPushButton, QMessageBox, QListWidget, QHBoxLayout, QFrame, QDialog, QDialogButtonBox, QSizePolicy)  # Import PyQt5 widgets
from PyQt5.QtGui import QPixmap, QImage, QFont  # Import PyQt5 GUI components
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal  # Import PyQt5 core components

from PIL import ImageFont, ImageDraw, Image  # Import PIL for image handling
import arabic_reshaper  # Import arabic_reshaper for reshaping Arabic text
from bidi.algorithm import get_display  # Import bidi for bidirectional text
import subprocess  # Import subprocess to run external processes
import threading  # Import threading again (redundant)
import re  # Import re again (redundant)

def speak_text_async(text):
    def _speak():
        if re.search(r'[\u0600-\u06FF]', text):  # Check if text contains Arabic characters
            try:
                from gtts import gTTS  # Import gTTS inside the function
                import os  # Import os inside the function
                tts = gTTS(text=text, lang='ar')  # Create a gTTS object for Arabic text
                filename = "temp_arabic.mp3"  # Temporary filename
                tts.save(filename)  # Save the audio file
                # Use mpg123 to play the Arabic MP3 file
                subprocess.run(['mpg123', filename])  # Play the audio file
                os.remove(filename)  # Remove the temporary file
            except Exception as e:
                print(f"Error in gTTS: {e}")  # Print error if something goes wrong
        else:
            subprocess.run(['espeak', text])  # Use espeak for non-Arabic text
    threading.Thread(target=_speak, daemon=True).start()  # Start the speak function in a new thread

def speak_text(engine, text):
    if re.search(r'[\u0600-\u06FF]', text):  # Check for Arabic characters
        try:
            from gtts import gTTS  # Import gTTS
            import os  # Import os
            tts = gTTS(text=text, lang='ar')  # Create gTTS object
            filename = "temp_arabic.mp3"  # Temporary filename
            tts.save(filename)  # Save the audio file
            subprocess.run(['mpg123', filename])  # Play the audio file
            os.remove(filename)  # Remove the file after playing
        except Exception as e:
            print(f"Error in gTTS: {e}")  # Print error message
    else:
        subprocess.run(['espeak', text])  # Use espeak for other languages

def putTextArabic(img, text, org, font_path="Amiri-Bold.ttf", font_size=32, color=(0,255,0)):
    reshaped_text = arabic_reshaper.reshape(text)  # Reshape Arabic text
    # Remove or comment out the bidi reordering
    # bidi_text = get_display(reshaped_text)
    bidi_text = reshaped_text  # Use reshaped text without bidi

    if len(img.shape) == 3 and img.shape[2] == 3:  # Check if image has 3 channels
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    else:
        img_rgb = img.copy()  # Make a copy if not

    pil_im = Image.fromarray(img_rgb)  # Convert to PIL image
    draw = ImageDraw.Draw(pil_im)  # Create a drawing context
    try:
        font = ImageFont.truetype(font_path, font_size)  # Load the specified font
    except IOError:
        font = ImageFont.load_default()  # Load default font if specified font not found
    draw.text(org, bidi_text, font=font, fill=color)  # Draw the text on the image
    img = np.array(pil_im)  # Convert back to NumPy array
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR
    return img_bgr  # Return the modified image

def speak_text(engine, text):
    if re.search(r'[\u0600-\u06FF]', text):  # Check for Arabic characters
        try:
            tts = gTTS(text=text, lang='ar')  # Create gTTS object
            filename = "temp_arabic.mp3"  # Temporary filename
            tts.save(filename)  # Save the audio
            playsound(filename)  # Play the audio
            os.remove(filename)  # Remove the file
        except Exception as e:
            print(f"Error in gTTS: {e}")  # Print error
    else:
        engine.say(text)  # Use the engine to say text
        engine.runAndWait()  # Wait until speaking is done

class AddGestureWindow(QWidget):
    GESTURES_FILE = 'gesture_data/gestures.txt'  # Path to gestures file

    def __init__(self):
        super().__init__()  # Initialize the parent class
        self.setWindowTitle("Add / Manage Gestures")  # Set window title
        self.setGeometry(0, 0, 400, 800)  # Set window size and position
        self.init_ui()  # Initialize the UI

    def init_ui(self):
        layout = QVBoxLayout()  # Create a vertical layout

        # Title Label with increased font size
        self.title_label = QLabel("Manage Gestures")  # Create title label
        self.title_label.setFont(QFont("Arial", 50, QFont.Bold))  # Set font
        self.title_label.setAlignment(Qt.AlignCenter)  # Center the text
        self.title_label.setStyleSheet("color: #d9f1ff; padding-bottom: 20px;")  # Set style

        # Gesture List with increased font size
        self.gesture_list = QListWidget()  # Create a list widget
        self.gesture_list.setLayoutDirection(Qt.RightToLeft)  # Set text direction
        self.gesture_list.setFont(QFont("Arial", 50))  # Set font size
        self.load_gestures()  # Load existing gestures

        # Gesture Input with increased font size
        self.gesture_input = QLineEdit()  # Create input field
        self.gesture_input.setPlaceholderText("Enter gesture name (supports Arabic/English)")  # Set placeholder
        self.gesture_input.setLayoutDirection(Qt.RightToLeft)  # Set text direction
        self.gesture_input.setFont(QFont("Arial", 50))  # Set font size

        # Button Layout
        button_layout = QHBoxLayout()  # Create a horizontal layout for buttons

        # Add Gesture Button with increased font size
        self.add_button = QPushButton("Add Gesture")  # Create add button
        self.add_button.setFont(QFont("Arial", 50, QFont.Bold))  # Set font
        self.add_button.setStyleSheet("""
            QPushButton {
                background-color: #2ECC71;
                color: #FFFFFF;
                border-radius: 15px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #27AE60;
            }
        """)  # Set button style

        # Modify Gesture Button with increased font size
        self.modify_button = QPushButton("Modify Gesture")  # Create modify button
        self.modify_button.setFont(QFont("Arial", 30, QFont.Bold))  # Set font
        self.modify_button.setStyleSheet("""
            QPushButton {
                background-color: #F1C40F;
                color: #FFFFFF;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #D4AC0D;
            }
        """)  # Set button style

        # Delete Gesture Button with increased font size
        self.delete_button = QPushButton("Delete Gesture")  # Create delete button
        self.delete_button.setFont(QFont("Arial", 30, QFont.Bold))  # Set font
        self.delete_button.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: #FFFFFF;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)  # Set button style

        # Add buttons to the button layout
        button_layout.addWidget(self.add_button)  # Add add button
        button_layout.addWidget(self.modify_button)  # Add modify button
        button_layout.addWidget(self.delete_button)  # Add delete button

        # Create Back and Exit buttons
        back_button = QPushButton("Back")  # Create back button
        back_button.setFont(QFont("Arial", 20, QFont.Bold))  # Set font
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #F1C40F;
                color: white;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #D8AC1D;
            }
        """)  # Set button style
        back_button.clicked.connect(self.close)  # Connect to close the window

        self.add_button.clicked.connect(self.add_gesture)  # Connect add button
        self.modify_button.clicked.connect(self.modify_gesture)  # Connect modify button
        self.delete_button.clicked.connect(self.delete_gesture)  # Connect delete button

        # Add widgets to the main layout
        layout.addWidget(self.title_label)  # Add title
        layout.addWidget(self.gesture_list)  # Add gesture list
        layout.addWidget(self.gesture_input)  # Add input field
        layout.addLayout(button_layout)  # Add button layout
        layout.addWidget(back_button)  # Add back button

        self.setLayout(layout)  # Set the main layout

    def load_gestures(self):
        self.gesture_list.clear()  # Clear the list
        if os.path.exists(self.GESTURES_FILE):  # Check if gestures file exists
            with open(self.GESTURES_FILE, 'r', encoding='utf-8') as file:
                gestures = file.read().splitlines()  # Read gestures
                self.gesture_list.addItems(gestures)  # Add to list widget

    def add_gesture(self):
        gesture = self.gesture_input.text().strip()  # Get input text
        if not gesture:
            QMessageBox.warning(self, "Error", "Please enter a gesture name.")  # Warn if empty
            return
        if gesture in self.get_gesture_list():
            QMessageBox.warning(self, "Error", "Gesture already exists.")  # Warn if duplicate
            return
        with open(self.GESTURES_FILE, 'a', encoding='utf-8') as file:
            file.write(gesture + '\n')  # Append new gesture
        self.gesture_list.addItem(gesture)  # Add to list widget
        self.gesture_input.clear()  # Clear input field
        QMessageBox.information(self, "Success", f"Gesture '{gesture}' added successfully.")  # Inform success

    def modify_gesture(self):
        selected_item = self.gesture_list.currentItem()  # Get selected gesture
        if not selected_item:
            QMessageBox.warning(self, "Error", "Please select a gesture to modify.")  # Warn if none selected
            return
        new_gesture = self.gesture_input.text().strip()  # Get new gesture name
        if not new_gesture:
            QMessageBox.warning(self, "Error", "Please enter the new gesture name.")  # Warn if empty
            return
        if new_gesture in self.get_gesture_list():
            QMessageBox.warning(self, "Error", "Gesture already exists.")  # Warn if duplicate
            return
        old_gesture = selected_item.text()  # Get old gesture name
        selected_item.setText(new_gesture)  # Update the list widget
        self.save_gestures()  # Save changes
        QMessageBox.information(self, "Success", f"Gesture '{old_gesture}' modified to '{new_gesture}'.")  # Inform success

    def delete_gesture(self):
        selected_item = self.gesture_list.currentItem()  # Get selected gesture
        if not selected_item:
            QMessageBox.warning(self, "Error", "Please select a gesture to delete.")  # Warn if none selected
            return
        reply = QMessageBox.question(self, "Delete Gesture", 
                                     f"Are you sure you want to delete '{selected_item.text()}'?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)  # Confirm deletion
        if reply == QMessageBox.Yes:
            text = selected_item.text()  # Get gesture name
            self.gesture_list.takeItem(self.gesture_list.row(selected_item))  # Remove from list widget
            self.save_gestures()  # Save changes
            QMessageBox.information(self, "Success", f"Gesture '{text}' deleted successfully.")  # Inform success

    def save_gestures(self):
        with open(self.GESTURES_FILE, 'w', encoding='utf-8') as file:
            for i in range(self.gesture_list.count()):
                file.write(self.gesture_list.item(i).text() + '\n')  # Write all gestures to file

    def get_gesture_list(self):
        return [self.gesture_list.item(i).text() for i in range(self.gesture_list.count())]  # Get list of gestures

# Collect Data Window
class CollectDataWindow(QWidget):
    GESTURES_FILE = 'gesture_data/gestures.txt'  # Path to gestures file

    def __init__(self):
        super().__init__()  # Initialize parent class
        self.setWindowTitle("Collect Data (BGR Fix)")  # Set window title
        self.setGeometry(0, 0, 650, 800)  # Set window size and position

        self.current_gesture = None  # Current gesture being collected
        self.collected_samples = 0  # Number of samples collected
        self.required_samples = 200  # Number of samples required

        self.hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)  # Initialize MediaPipe hands

        self.picam2 = None  # Initialize camera as None
        self.timer = QTimer(self)  # Create a timer
        self.timer.timeout.connect(self.update_frame)  # Connect timer to update_frame
        self.last_frame = None  # To store the last frame

        self.init_ui()  # Initialize the UI

    def init_ui(self):
        layout = QVBoxLayout()  # Create a vertical layout
        title_label = QLabel("Collect Data")  # Create title label
        title_label.setFont(QFont("Amiri-Bold", 20, QFont.Bold))  # Set font
        title_label.setAlignment(Qt.AlignCenter)  # Center the text
        layout.addWidget(title_label)  # Add title to layout

        self.gesture_list = QListWidget()  # Create a list widget
        self.load_gestures()  # Load existing gestures

        self.start_button = QPushButton("Start Collecting Data")  # Create start button
        self.start_button.clicked.connect(self.start_collection)  # Connect to start_collection

        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #2ECC71;
                color: #FFFFFF;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #27AE60;
            }
        """)  # Set button style
        self.video_label = QLabel("Camera feed will appear here.")  # Create video label
        self.video_label.setAlignment(Qt.AlignCenter)  # Center the text

        layout.addWidget(QLabel("Select a Gesture:"))  # Add instruction label
        layout.addWidget(self.gesture_list)  # Add gesture list
        layout.addWidget(self.start_button)  # Add start button
        layout.addWidget(self.video_label)  # Add video label

        # Create Back and Exit buttons
        back_button = QPushButton("Back")  # Create back button
        back_button.setFont(QFont("Arial", 30, QFont.Bold))  # Set font
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #F1C40F;
                color: white;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #D8AC1D;
            }
        """)  # Set button style
        back_button.clicked.connect(self.close)  # Connect to close window

        layout.addWidget(back_button)  # Add back button

        self.setLayout(layout)  # Set the main layout

    def load_gestures(self):
        if os.path.exists(self.GESTURES_FILE):  # Check if gestures file exists
            with open(self.GESTURES_FILE, 'r', encoding='utf-8') as file:
                gestures = file.read().splitlines()  # Read gestures
                self.gesture_list.addItems(gestures)  # Add to list widget

    def start_collection(self):
        selected_item = self.gesture_list.currentItem()  # Get selected gesture
        if not selected_item:
            QMessageBox.warning(self, "Error", "Please select a gesture first.")  # Warn if none selected
            return
        self.current_gesture = selected_item.text()  # Set current gesture
        self.collected_samples = 0  # Reset collected samples

        try:
            self.picam2 = Picamera2()  # Initialize camera
            cam_config = self.picam2.create_preview_configuration(
                main={"format": "BGR888", "size": (640, 480)}
            )  # Configure camera
            self.picam2.configure(cam_config)  # Apply configuration
            self.picam2.start()  # Start camera
        except RuntimeError as e:
            QMessageBox.critical(self, "Error", f"Camera initialization failed: {str(e)}")  # Show error
            return

        self.timer.start(30)  # Start the timer

    def update_frame(self):
        if not self.picam2:
            return  # Exit if camera not initialized
        frame = self.picam2.capture_array()  # Capture frame
        if frame is None:
            return  # Exit if no frame
        # Potential mirroring operation
        frame = cv2.flip(frame, 1)  # Flip horizontally

        # Convert BGR to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color
        results = self.hands.process(rgb_frame)  # Process frame with MediaPipe

        if results and results.multi_hand_landmarks:  # Check if any hands detected
            for hand_landmarks in results.multi_hand_landmarks:
                try:
                    landmarks = []
                    for point in hand_landmarks.landmark:
                        landmarks.extend([point.x, point.y, point.z])  # Extract landmarks
                    self.save_landmarks_to_csv(self.current_gesture, landmarks)  # Save landmarks
                    self.collected_samples += 1  # Increment sample count
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,  
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS
                    )  # Draw landmarks on frame
                except Exception as e:
                    print(f"Error processing landmarks: {e}")  # Print error

        # Save the last frame for image saving
        self.last_frame = frame.copy()  # Store the frame

        # Embed RTL markers around the Arabic gesture name
        text1 = f"Gesture: \u202B{self.current_gesture}\u202C"  # Format gesture text
        text2 = f"Samples: {self.collected_samples}/{self.required_samples}"  # Format sample count
        frame = putTextArabic(frame, text1, (10, 30), font_size=32, color=(0,255,0))  # Add gesture text
        frame = putTextArabic(frame, text2, (10, 70), font_size=32, color=(0,255,0))  # Add sample count

        show_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color for display
        qt_image = QImage(
            show_rgb.data, show_rgb.shape[1], show_rgb.shape[0],
            QImage.Format_RGB888
        )  # Create QImage from frame
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))  # Display the image

        if self.collected_samples >= self.required_samples:  # Check if required samples are collected
            self.finish_collection()  # Finish collection

    def finish_collection(self):
        if self.picam2:
            self.picam2.stop()  # Stop camera
            self.picam2.close()  # Close camera
            self.picam2 = None  # Reset camera
        self.timer.stop()  # Stop the timer

        # Save the last frame as an image
        if self.last_frame is not None:
            images_dir = 'gesture_images'  # Directory for images
            os.makedirs(images_dir, exist_ok=True)  # Create directory if not exists
            image_path = os.path.join(images_dir, f"{self.current_gesture}.png")  # Image path
            cv2.imwrite(image_path, self.last_frame)  # Save image
            print(f"Saved image to {image_path}")  # Print confirmation

        QMessageBox.information(
            self,
            "Info",
            f"Collected {self.required_samples} samples for gesture '{self.current_gesture}'."
        )  # Inform the user

    def save_landmarks_to_csv(self, gesture_name, landmark_list):
        DATA_DIR = 'gesture_data'  # Data directory
        CSV_FILE = os.path.join(DATA_DIR, 'gesture_landmarks.csv')  # CSV file path
        os.makedirs(DATA_DIR, exist_ok=True)  # Create directory if not exists
        row = [gesture_name] + landmark_list  # Prepare row data
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)  # Create CSV writer
            writer.writerow(row)  # Write the row

    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()  # Stop timer if active
        if self.picam2:
            self.picam2.stop()  # Stop camera
            self.picam2.close()  # Close camera
            self.picam2 = None  # Reset camera
        self.hands.close()  # Close MediaPipe hands
        event.accept()  # Accept the close event

class TrainResultDialog(QDialog):
    def __init__(self, message):
        super().__init__()  # Initialize parent class
        self.setWindowTitle("Training Result")  # Set window title
        self.setFixedSize(400, 200)  # Set fixed size
        layout = QVBoxLayout()  # Create a vertical layout
        label = QLabel(message)  # Create label with message
        label.setAlignment(Qt.AlignCenter)  # Center the text
        label.setStyleSheet("font-size: 16px; font-weight: bold; color: #ECF0F1;")  # Set style
        layout.addWidget(label)  # Add label to layout
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)  # Create OK button
        button_box.accepted.connect(self.accept)  # Connect to accept the dialog
        layout.addWidget(button_box)  # Add button box to layout
        self.setStyleSheet("""
        QDialog {
            background-color: #2F3136;
        }
        QDialogButtonBox QPushButton {
            background-color: #7289DA;
            border-radius: 5px;
            color: #FFFFFF;
            padding: 5px 10px;
        }
        QDialogButtonBox QPushButton:hover {
            background-color: #5B6EAE;
        }
        """)  # Set dialog style
        self.setLayout(layout)  # Set the layout

class TrainWindow(QWidget):
    def __init__(self):
        super().__init__()  # Initialize parent class
        self.setWindowTitle("Train Model")  # Set window title
        self.setGeometry(50, 150, 500, 300)  # Set window size and position
        self.init_ui()  # Initialize the UI

    def init_ui(self):
        layout = QVBoxLayout()  # Create a vertical layout
        self.train_button = QPushButton("Start Training")  # Create training button
        self.train_button.clicked.connect(self.train_model)  # Connect to train_model
        self.train_button.setStyleSheet("""
            QPushButton {
                background-color: #2ECC71;
                color: #FFFFFF;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #27AE60;
            }
        """)  # Set button style

        # Create Back and Exit buttons
        back_button = QPushButton("Back")  # Create back button
        back_button.setFont(QFont("Arial", 30, QFont.Bold))  # Set font
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #F1C40F;
                color: white;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #D8AC1D;
            }
        """)  # Set button style
        back_button.clicked.connect(self.close)  # Connect to close window
        layout.addWidget(self.train_button, alignment=Qt.AlignCenter)  # Add training button
        layout.addWidget(back_button)  # Add back button

        self.setLayout(layout)  # Set the main layout

    def train_model(self):
        try:
            csv_path = 'gesture_data/gesture_landmarks.csv'  # Path to CSV file
            if not os.path.exists(csv_path):
                QMessageBox.warning(self, "Error", "No data found. Collect gesture data first.")  # Warn if no data
                return
            data = pd.read_csv(csv_path)  # Read CSV data
            X = data.iloc[:, 1:].values  # Features
            y = data.iloc[:, 0].values  # Labels
            le = LabelEncoder()  # Initialize label encoder
            y_encoded = le.fit_transform(y)  # Encode labels
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)  # Split data
            scaler = StandardScaler()  # Initialize scaler
            X_train = scaler.fit_transform(X_train)  # Fit and transform training data
            X_test = scaler.transform(X_test)  # Transform test data
            model = Sequential([
                Input(shape=(X_train.shape[1],)),  # Input layer
                Dense(128, activation='relu'),  # Hidden layer
                Dropout(0.2),  # Dropout for regularization
                Dense(64, activation='relu'),  # Another hidden layer
                Dense(len(np.unique(y_encoded)), activation='softmax')  # Output layer
            ])  # Define the model architecture
            model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Compile the model
            early_stopping = EarlyStopping(monitor='val_loss', patience=3)  # Early stopping callback
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)  # Reduce learning rate callback
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, callbacks=[early_stopping, reduce_lr])  # Train the model
            os.makedirs('models', exist_ok=True)  # Create models directory
            model.save('models/gesture_recognition_model.keras')  # Save the trained model
            joblib.dump(scaler, 'models/scaler.save')  # Save the scaler
            joblib.dump(le, 'models/label_encoder.save')  # Save the label encoder
            dialog = TrainResultDialog("Model trained and saved successfully!")  # Create success dialog
            dialog.exec_()  # Show the dialog
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Training failed: {str(e)}")  # Show error if training fails

class TranslateWindow(QWidget):
    def __init__(self):
        super().__init__()  # Initialize parent class
        self.setWindowTitle("Translate Gestures")  # Set window title
        self.setGeometry(0, 0, 650, 850)  # Set window size and position
        self.init_ui()  # Initialize UI
        self.hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)  # Initialize MediaPipe hands
        self.engine = pyttsx3.init()  # Initialize text-to-speech engine

        voices = self.engine.getProperty('voices')  # Get available voices
        arabic_keywords = ["Arabic", "Naayf", "Zehra", "Tagh"]  # Define Arabic voice keywords
        self.arabic_voice_found = False  # Flag for Arabic voice
        for voice in voices:
            if any(keyword.lower() in voice.name.lower() for keyword in arabic_keywords):
                self.engine.setProperty('voice', voice.id)  # Set Arabic voice
                self.arabic_voice_found = True  # Set flag
                break

        self.model = load_model('models/gesture_recognition_model.keras')  # Load trained model
        self.scaler = joblib.load('models/scaler.save')  # Load scaler
        self.label_encoder = joblib.load('models/label_encoder.save')  # Load label encoder

        self.picam2 = Picamera2()  # Initialize camera
        self.picam2.configure(self.picam2.create_preview_configuration())  # Configure camera
        self.picam2.start()  # Start camera

        self.timer = QTimer(self)  # Create a timer
        self.timer.timeout.connect(self.update_frame)  # Connect timer to update_frame
        self.timer.start(30)  # Start the timer
        self.mp_drawing = mp.solutions.drawing_utils  # Initialize drawing utils
        self.mp_hands = mp.solutions.hands  # Initialize hands module
        self.last_gesture = None  # Store the last gesture

    def init_ui(self):
        main_layout = QVBoxLayout()  # Create main vertical layout
        top_button_layout = QHBoxLayout()  # Create top horizontal layout
        self.back_button = QPushButton(" Back")  # Create back button
        self.back_button.setFont(QFont("Arial", 100, QFont.Bold))  # Set font
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: #F1C40F;
                color: #FFFF;
                border-radius: 16px;
                padding: 20px 30px;
            }
            QPushButton:hover {
                background-color: #D8AC1D;
            }
        """)  # Set button style

        self.exit_button = QPushButton(" Exit")  # Create exit button
        self.exit_button.setFont(QFont("Arial", 50, QFont.Bold))  # Set font
        self.exit_button.setStyleSheet("""
            QPushButton {
                background-color: #ff0000;
                color: #FFFFFF;
                border-radius: 16px;
                padding: 20px 30px;
            }
            QPushButton:hover {
                background-color: #ff0000;
            }
        """)  # Set button style
        top_button_layout.addWidget(self.back_button)  # Add back button
        top_button_layout.addWidget(self.exit_button)  # Add exit button
        main_layout.addLayout(top_button_layout)  # Add top button layout to main layout
        self.video_label = QLabel()  # Create video label
        main_layout.addWidget(self.video_label)  # Add video label to main layout
        self.setLayout(main_layout)  # Set the main layout

        self.back_button.clicked.connect(self.close)  # Connect back button to close
        self.exit_button.clicked.connect(QApplication.quit)  # Connect exit button to quit application

    def update_frame(self):
        frame = self.picam2.capture_array()  # Capture frame from camera
        if frame is None:
            return  # Exit if no frame
        frame = cv2.flip(frame, 1)  # Flip frame horizontally
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color for processing
        results = self.hands.process(frame_rgb)  # Process frame with MediaPipe
        gesture = "Not Detected"  # Default gesture
        if results.multi_hand_landmarks:  # If hands are detected
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)  # Draw landmarks
                landmarks = [lm for point in hand_landmarks.landmark for lm in (point.x, point.y, point.z)]  # Extract landmarks
                X = self.scaler.transform(np.array(landmarks).reshape(1, -1))  # Scale features
                prediction = self.model.predict(X)  # Predict gesture
                class_id = np.argmax(prediction)  # Get class ID
                gesture = self.label_encoder.inverse_transform([class_id])[0]  # Decode gesture
                if self.last_gesture != gesture:
                    speak_text_async(gesture)  # Speak the gesture
                    self.last_gesture = gesture  # Update last gesture
        frame_rgb = putTextArabic(frame_rgb, f"Gesture: {gesture}", (40, 80), font_size=32, color=(0, 255, 0))  # Add gesture text
        qt_image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)  # Create QImage
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))  # Display the image

    def closeEvent(self, event):
        self.timer.stop()  # Stop timer
        self.picam2.stop()  # Stop camera
        self.hands.close()  # Close MediaPipe hands
        event.accept()  # Accept the close event

# …………………………………………………………………………………………………

class TranscriptionWorker(QThread):
    transcription_complete = pyqtSignal(str)  # Signal emitted when transcription is complete
    transcription_failed = pyqtSignal(str)  # Signal emitted when transcription fails

    def __init__(self, whisper_model, audio_file, gestures, gesture_images_dir):
        super().__init__()  # Initialize parent class
        self.whisper_model = whisper_model  # Whisper model for transcription
        self.audio_file = audio_file  # Path to audio file
        self.gestures = gestures  # List of gestures
        self.gesture_images_dir = gesture_images_dir  # Directory for gesture images

    def run(self):
        try:
            # Run Whisper transcription
            result = self.whisper_model.transcribe(self.audio_file, fp16=False)  # Transcribe audio
            recognized_text = result["text"].lower().strip()  # Get recognized text
            if not recognized_text:
                self.transcription_failed.emit("No speech recognized.")  # Emit failure if no text
                return
            # Emit the recognized text
            self.transcription_complete.emit(recognized_text)  # Emit success signal
        except Exception as e:
            self.transcription_failed.emit(f"Transcription error: {str(e)}")  # Emit failure signal on error

class ReverseTranslateWindow(QWidget):
    GESTURES_FILE = 'gesture_data/gestures.txt'  # Path to gestures file
    GESTURE_IMAGES_DIR = 'gesture_images'  # Directory containing gesture images

    def __init__(self):
        super().__init__()  # Initialize parent class
        self.setWindowTitle("Reverse Translate (AI-Powered Speech to Gesture)")  # Set window title
        self.setGeometry(50, 50, 600, 400)  # Set window size and position
        self.init_ui()  # Initialize UI
        self.load_gestures()  # Load gestures

        # Load a smaller Whisper model for faster processing
        try:
            self.whisper_model = whisper.load_model("small")  # Load Whisper model
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load Whisper model: {str(e)}")  # Show error
            sys.exit(1)  # Exit application

        # Initialize variables
        self.transcription_worker = None  # Initialize transcription worker
        self.audio_file = "temp_speech_input.wav"  # Path to temporary audio file

        # Start listening after a short delay
        QTimer.singleShot(500, self.start_listening)  # Start listening after 500ms

    def init_ui(self):
        self.main_layout = QVBoxLayout()  # Create main vertical layout

        # Title
        title_label = QLabel("Reverse Translate")  # Create title label
        title_label.setFont(QFont("Arial", 24, QFont.Bold))  # Set font
        title_label.setAlignment(Qt.AlignCenter)  # Center the text
        self.main_layout.addWidget(title_label)  # Add title to layout

        # Instruction label
        self.instruction_label = QLabel("Please say the gesture name...")  # Create instruction label
        self.instruction_label.setFont(QFont("Arial", 16))  # Set font
        self.instruction_label.setAlignment(Qt.AlignCenter)  # Center the text
        self.main_layout.addWidget(self.instruction_label)  # Add instruction to layout

        # Microphone and status layout
        mic_layout = QHBoxLayout()  # Create horizontal layout

        # Loading Indicator (Animated GIF)
        self.loading_label = QLabel()  # Create loading label
        loading_gif_path = "loading.gif"  # Path to loading GIF
        if os.path.exists(loading_gif_path):
            self.loading_movie = QMovie(loading_gif_path)  # Create movie from GIF
            self.loading_label.setMovie(self.loading_movie)  # Set movie to label
            self.loading_movie.start()  # Start the movie
            self.loading_label.hide()  # Hide initially
        else:
            # If no loading GIF, use a simple text indicator
            self.loading_label.setText("")  # Set empty text
            self.loading_label.setFont(QFont("Arial", 24))  # Set font
            self.loading_label.hide()  # Hide initially

        mic_layout.addWidget(self.loading_label, alignment=Qt.AlignCenter)  # Add loading label to layout

        # Status label
        self.status_label = QLabel("Preparing to listen...")  # Create status label
        self.status_label.setAlignment(Qt.AlignCenter)  # Center the text
        self.status_label.setFont(QFont("Arial", 14, QFont.Normal))  # Set font
        mic_layout.addWidget(self.status_label)  # Add status label to layout
        self.main_layout.addLayout(mic_layout)  # Add microphone layout to main layout

        # Image label for displaying recognized gesture images
        self.image_label = QLabel()  # Create image label
        self.image_label.setAlignment(Qt.AlignCenter)  # Center the image
        self.main_layout.addWidget(self.image_label)  # Add image label to layout

        # Buttons (Retry and Close)
        button_layout = QHBoxLayout()  # Create button layout
        self.retry_button = QPushButton(" Retry")  # Create retry button
        self.close_button = QPushButton(" Close")  # Create close button

        self.retry_button.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: #ECF0F1;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """)  # Set retry button style

        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: #ECF0F1;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)  # Set close button style

        self.retry_button.clicked.connect(self.start_listening)  # Connect retry button
        self.close_button.clicked.connect(self.close)  # Connect close button

        button_layout.addWidget(self.retry_button)  # Add retry button
        button_layout.addWidget(self.close_button)  # Add close button
        self.main_layout.addLayout(button_layout)  # Add button layout to main layout

        # Set the final layout
        self.setLayout(self.main_layout)  # Set the main layout

    def load_gestures(self):
        self.gestures = []  # Initialize gestures list
        if os.path.exists(self.GESTURES_FILE):  # Check if gestures file exists
            with open(self.GESTURES_FILE, 'r', encoding='utf-8') as file:
                self.gestures = [line.strip().lower() for line in file if line.strip()]  # Read and clean gestures
        else:
            QMessageBox.warning(self, "Warning", f"Gestures file '{self.GESTURES_FILE}' not found.")  # Warn if file not found

    def start_listening(self):
        self.instruction_label.setText("Please say the gesture name...")  # Update instruction
        self.status_label.setText("Adjusting microphone noise...")  # Update status
        self.image_label.clear()  # Clear image label

        # Show loading indicator
        self.loading_label.show()  # Show loading animation

        QApplication.processEvents()  # Process UI events

        # Adjust microphone noise if needed
        self.adjust_mic_noise(1.0)  # Placeholder for noise adjustment

        # Update status
        self.status_label.setText("Listening (speak now)...")  # Update status
        QApplication.processEvents()  # Process UI events

        # Start capturing speech after a short delay
        QTimer.singleShot(100, self.capture_speech)  # Start capturing after 100ms

    def adjust_mic_noise(self, duration=1.0):
        # Placeholder for microphone noise adjustment logic
        # You can implement VAD or noise reduction here if needed
        pass  # Do nothing for now

    def capture_speech(self):
        self.status_label.setText("Recording...")  # Update status
        QApplication.processEvents()  # Process UI events

        success = self.record_audio(self.audio_file, record_seconds=4)  # Record audio
        if not success:
            self.status_label.setText("No audio captured. Please try again.")  # Update status on failure
            self.loading_label.hide()  # Hide loading indicator
            return

        self.status_label.setText("Processing with Whisper...")  # Update status
        QApplication.processEvents()  # Process UI events

        # Start transcription in a separate thread
        self.transcription_worker = TranscriptionWorker(
            whisper_model=self.whisper_model,
            audio_file=self.audio_file,
            gestures=self.gestures,
            gesture_images_dir=self.GESTURE_IMAGES_DIR
        )  # Initialize transcription worker
        self.transcription_worker.transcription_complete.connect(self.on_transcription_complete)  # Connect success signal
        self.transcription_worker.transcription_failed.connect(self.on_transcription_failed)  # Connect failure signal
        self.transcription_worker.start()  # Start the worker

    def record_audio(self, filename, record_seconds=4):
        CHUNK = 1024  # Audio chunk size
        FORMAT = pyaudio.paInt16  # Audio format
        CHANNELS = 1  # Mono audio
        RATE = 16000  # Sampling rate
        p = pyaudio.PyAudio()  # Initialize PyAudio

        try:
            stream = p.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True, frames_per_buffer=CHUNK)  # Open audio stream
        except Exception as e:
            print(f"Error opening microphone stream: {e}")  # Print error
            QMessageBox.critical(self, "Error", f"Failed to access microphone: {str(e)}")  # Show error message
            p.terminate()  # Terminate PyAudio
            return False

        frames = []  # List to store audio frames
        try:
            for _ in range(0, int(RATE / CHUNK * record_seconds)):
                data = stream.read(CHUNK)  # Read audio data
                frames.append(data)  # Append to frames
        except Exception as e:
            print(f"Error during recording: {e}")  # Print error
            QMessageBox.critical(self, "Error", f"Failed during recording: {str(e)}")  # Show error message
            stream.stop_stream()  # Stop stream
            stream.close()  # Close stream
            p.terminate()  # Terminate PyAudio
            return False

        stream.stop_stream()  # Stop the stream
        stream.close()  # Close the stream
        p.terminate()  # Terminate PyAudio

        if not frames:
            return False  # Return failure if no frames

        try:
            wf = wave.open(filename, 'wb')  # Open WAV file for writing
            wf.setnchannels(CHANNELS)  # Set number of channels
            wf.setsampwidth(p.get_sample_size(FORMAT))  # Set sample width
            wf.setframerate(RATE)  # Set frame rate
            wf.writeframes(b''.join(frames))  # Write audio frames
            wf.close()  # Close the file
            return True  # Return success
        except Exception as e:
            print(f"Error saving audio file: {e}")  # Print error
            QMessageBox.critical(self, "Error", f"Failed to save audio: {str(e)}")  # Show error message
            return False  # Return failure

    def on_transcription_complete(self, recognized_text):
        print(f"Transcription complete: {recognized_text}")  # Print recognized text
        self.status_label.setText("Transcription complete.")  # Update status
        self.loading_label.hide()  # Hide loading indicator
        self.show_gesture_for_word(recognized_text)  # Show gesture

    def on_transcription_failed(self, error_message):
        print(f"Transcription failed: {error_message}")  # Print error message
        self.status_label.setText(error_message)  # Update status
        self.instruction_label.setText("Please try again.")  # Update instruction
        self.loading_label.hide()  # Hide loading indicator

    def run_whisper(self, audio_file):
        # Deprecated: Now handled in TranscriptionWorker
        pass  # Do nothing

    def show_gesture_for_word(self, word):
        print(f"Recognized word: {word}")  # Print recognized word
        normalized_word = re.sub(r'[^\w\s]', '', word).strip().lower()  # Normalize the word
        print(f"Normalized word: {normalized_word}")  # Print normalized word

        # Check if the word contains Arabic characters
        if re.search(r'[\u0600-\u06FF]', word):
            print("Arabic logic branch")  # Indicate Arabic branch
            if word in self.gestures:
                image_path = os.path.join(self.GESTURE_IMAGES_DIR, f"{word}.png")  # Path to gesture image
                print(f"Looking for image at: {image_path}")  # Print image path
                if os.path.exists(image_path):
                    print("Image found. Displaying image.")  # Indicate image found
                    pixmap = QPixmap(image_path)  # Load image
                    self.image_label.setPixmap(pixmap.scaled(
                        400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))  # Display image
                    self.instruction_label.setText(f"Understood: '{word}'")  # Update instruction
                    self.status_label.setText("Gesture found!")  # Update status
                    QApplication.processEvents()  # Process UI events
                    QTimer.singleShot(6000, self.close)  # Close after 6 seconds
                else:
                    print(f"Image not found at: {image_path}")  # Indicate image not found
                    self.status_label.setText("No matching image found.")  # Update status
                    self.instruction_label.setText(f"No image for '{word}'. Try another gesture.")  # Update instruction
            else:
                print(f"Word '{word}' not found in gestures list")  # Indicate word not found
                self.status_label.setText("Unknown gesture.")  # Update status
                self.instruction_label.setText(f"'{word}' does not match any known gesture. Try again.")  # Update instruction
        else:
            # English logic
            matched_gesture = None  # Initialize matched gesture
            for gesture in self.gestures:
                if gesture.lower() == normalized_word:
                    matched_gesture = gesture  # Match found
                    break
            if matched_gesture:
                image_path = os.path.join(self.GESTURE_IMAGES_DIR, f"{matched_gesture}.png")  # Path to gesture image
                if os.path.exists(image_path):
                    pixmap = QPixmap(image_path)  # Load image
                    self.image_label.setPixmap(pixmap.scaled(
                        400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))  # Display image
                    self.instruction_label.setText(f"Understood: '{word}'")  # Update instruction
                    self.status_label.setText("Gesture found!")  # Update status
                    QApplication.processEvents()  # Process UI events
                    QTimer.singleShot(6000, self.close)  # Close after 6 seconds
                else:
                    print(f"Image not found at: {image_path}")  # Indicate image not found
                    self.status_label.setText("No matching image found.")  # Update status
                    self.instruction_label.setText(f"No image for '{word}'. Try another gesture.")  # Update instruction
            else:
                print(f"Word '{word}' not found in gestures list")  # Indicate word not found
                self.status_label.setText("Unknown gesture.")  # Update status
                self.instruction_label.setText(f"'{word}' does not match any known gesture. Try again.")  # Update instruction

    def closeEvent(self, event):
        # Clean up any running transcription workers
        if self.transcription_worker and self.transcription_worker.isRunning():
            self.transcription_worker.terminate()  # Terminate worker
            self.transcription_worker.wait()  # Wait for termination
        event.accept()  # Accept the close event

# ………………#…………………##########

class ContinuousTranslateWindow(QWidget):
    def __init__(self):
        super().__init__()  # Initialize parent class
        self.setWindowTitle("Continuous Translate Gestures")  # Set window title
        self.setGeometry(30, 20, 700, 500)  # Set window size and position
        self.setFixedSize(600, 400)  # Fix the window size

        self.init_ui()  # Initialize UI

        self.hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)  # Initialize MediaPipe hands
        self.engine = pyttsx3.init()  # Initialize text-to-speech engine
        voices = self.engine.getProperty('voices')  # Get available voices
        arabic_keywords = ["Arabic", "Naayf", "Zehra", "Tagh"]  # Define Arabic voice keywords
        for voice in voices:
            if any(keyword.lower() in voice.name.lower() for keyword in arabic_keywords):
                self.engine.setProperty('voice', voice.id)  # Set Arabic voice
                break

        self.model = load_model('models/gesture_recognition_model.keras')  # Load trained model
        self.scaler = joblib.load('models/scaler.save')  # Load scaler
        self.label_encoder = joblib.load('models/label_encoder.save')  # Load label encoder

        self.picam2 = Picamera2()  # Initialize camera
        self.picam2.configure(self.picam2.create_preview_configuration())  # Configure camera
        self.picam2.start()  # Start camera

        self.mp_drawing = mp.solutions.drawing_utils  # Initialize drawing utils
        self.mp_hands = mp.solutions.hands  # Initialize hands module

        self.gesture_sequence = []  # List to store gesture sequence
        self.last_prediction = None  # Last predicted gesture
        self.no_hand_counter = 0  # Counter for no hands detected

        self.timer = QTimer(self)  # Create a timer
        self.timer.timeout.connect(self.update_frame)  # Connect timer to update_frame
        self.timer.start(30)  # Start the timer

    def init_ui(self):
        main_layout = QVBoxLayout()  # Create main vertical layout
        top_button_layout = QHBoxLayout()  # Create top horizontal layout

        self.back_button = QPushButton(" Back")  # Create back button
        self.exit_button = QPushButton(" Exit")  # Create exit button

        # Style Back and Exit buttons as needed
        self.back_button.setFont(QFont("Arial", 16, QFont.Bold))  # Set font
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: #F1C40F;
                color: white;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #D8AC1D;
            }
        """)  # Set button style
        self.exit_button.setFont(QFont("Arial", 16, QFont.Bold))  # Set font
        self.exit_button.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)  # Set button style

        top_button_layout.addWidget(self.back_button)  # Add back button
        top_button_layout.addStretch()  # Add stretch for spacing
        top_button_layout.addWidget(self.exit_button)  # Add exit button

        self.video_label = QLabel()  # Create video label
        self.sequence_label = QLabel("Detected Sequence: ")  # Create sequence label
        self.sequence_label.setAlignment(Qt.AlignCenter)  # Center the text
        self.sequence_label.setStyleSheet("font-size: 16px; color: #FFFFFF;")  # Set style

        # Clear Button to clear the sequence
        self.clear_button = QPushButton("Clear")  # Create clear button
        self.clear_button.setFont(QFont("Arial", 16, QFont.Bold))  # Set font
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #95A5A6;
                color: #FFFFFF;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #7F8C8D;
            }
        """)  # Set button style
        self.clear_button.clicked.connect(self.clear_sequence)  # Connect to clear_sequence

        # Add widgets to layout
        main_layout.addLayout(top_button_layout)  # Add top button layout
        main_layout.addWidget(self.video_label)  # Add video label
        main_layout.addWidget(self.sequence_label)  # Add sequence label
        main_layout.addWidget(self.clear_button)  # Add clear button

        self.setLayout(main_layout)  # Set the main layout

        self.back_button.clicked.connect(self.close)  # Connect back button
        self.exit_button.clicked.connect(QApplication.quit)  # Connect exit button

    def clear_sequence(self):
        """Clears the detected sequence text and resets the gesture sequence."""
        self.gesture_sequence.clear()  # Clear the gesture sequence list
        self.sequence_label.setText("Detected Sequence: ")  # Reset the sequence label

    def update_frame(self):
        frame = self.picam2.capture_array()  # Capture frame from camera
        if frame is None:
            return  # Exit if no frame
        frame = cv2.flip(frame, 1)  # Flip frame horizontally
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color for processing
        results = self.hands.process(frame_rgb)  # Process frame with MediaPipe
        if results.multi_hand_landmarks:  # If hands are detected
            self.no_hand_counter = 0  # Reset no hand counter
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)  # Draw landmarks
                landmarks = [lm for point in hand_landmarks.landmark for lm in (point.x, point.y, point.z)]  # Extract landmarks
                X = self.scaler.transform(np.array(landmarks).reshape(1, -1))  # Scale features
                prediction = self.model.predict(X)  # Predict gesture
                class_id = np.argmax(prediction)  # Get class ID
                gesture = self.label_encoder.inverse_transform([class_id])[0]  # Decode gesture
                if gesture != self.last_prediction:
                    self.last_prediction = gesture  # Update last prediction
                    self.gesture_sequence.append(gesture)  # Append to sequence
                    self.sequence_label.setText("Detected Sequence: " + " ".join(self.gesture_sequence))  # Update label
                    speak_text_async(gesture)  # Speak the gesture
        else:
            self.no_hand_counter += 1  # Increment no hand counter
            if self.no_hand_counter > 30 and self.gesture_sequence:
                self.gesture_sequence = []  # Clear gesture sequence
                self.sequence_label.setText("Detected Sequence: ")  # Reset label
        frame_rgb = putTextArabic(frame_rgb, "Continuous Detection", (40, 40), font_size=32, color=(0,255,0))  # Add text
        qt_image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)  # Create QImage
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))  # Display the image

    def closeEvent(self, event):
        self.timer.stop()  # Stop timer
        self.picam2.stop()  # Stop camera
        self.hands.close()  # Close MediaPipe hands
        event.accept()  # Accept the close event

class MainMenu(QWidget):
    def __init__(self):
        super().__init__()  # Initialize parent class
        self.setWindowTitle("AI Smart Badge")  # Set window title
        # Set desired geometry (width reduced as needed)
        self.setGeometry(0, 0, 720, 1200)  # Set window size and position
        self.init_ui()  # Initialize UI

    def init_ui(self):
        # Create a main vertical layout
        main_layout = QVBoxLayout()  # Create main vertical layout

        # Horizontal layout for logos at the top
        logo_layout = QHBoxLayout()  # Create horizontal layout for logos
        self.left_logo_label = QLabel()  # Create left logo label
        if os.path.exists("qstss.png"):
            left_pixmap = QPixmap("qstss.png")  # Load left logo image
            # Adjust size as needed
            self.left_logo_label.setPixmap(left_pixmap.scaled(300, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation))  # Set pixmap
        self.left_logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # Align left logo

        self.right_logo_label = QLabel()  # Create right logo label
        if os.path.exists("hamdan.png"):
            right_pixmap = QPixmap("hamdan.png")  # Load right logo image
            # Adjust size as needed
            self.right_logo_label.setPixmap(right_pixmap.scaled(300, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation))  # Set pixmap
        self.right_logo_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # Align right logo

        # Add logos to the horizontal layout with stretches for spacing
        logo_layout.addWidget(self.left_logo_label)  # Add left logo
        logo_layout.addStretch()  # Add stretch for spacing
        logo_layout.addWidget(self.right_logo_label)  # Add right logo

        # Create the title labels for English and Arabic
        self.title_label_en = QLabel("Ai Smart Badge Platform")  # Create English title label
        self.title_label_en.setStyleSheet("""
            color: #d9f1ff; 
            font-size: 50px; 
            font-weight: bold; 
            padding-top: 20px;
        """)  # Set style
        self.title_label_en.setAlignment(Qt.AlignCenter)  # Center the text

        self.title_label_ar = QLabel("Ai Pioneers Team - Qatar")  # Create Arabic title label
        self.title_label_ar.setStyleSheet("""
            color: #e6b400; 
            font-size: 50px; 
            font-weight: bold; 
            padding-top: 10px;
        """)  # Set style
        self.title_label_ar.setAlignment(Qt.AlignCenter)  # Center the text
        self.title_label_ar.setLayoutDirection(Qt.RightToLeft)  # Set text direction

        # Add layouts and title labels to the main vertical layout
        main_layout.addLayout(logo_layout)  # Add logo layout
        main_layout.addWidget(self.title_label_en)  # Add English title
        main_layout.addWidget(self.title_label_ar)  # Add Arabic title
        self.setLayout(main_layout)  # Set the main layout

        # Continue with the rest of the UI setup (canvas, buttons, etc.)
        self.canvas = QFrame(self)  # Create a frame for buttons
        self.canvas.setObjectName("canvasFrame")  # Set object name for styling
        self.canvas.setFrameShape(QFrame.StyledPanel)  # Set frame shape
        self.canvas.setFrameShadow(QFrame.Raised)  # Set frame shadow

        canvas_layout = QVBoxLayout(self.canvas)  # Create vertical layout for canvas

        self.add_gesture_button = self.create_button(
            text=" Add / Manage Gestures",
            color="#9B59B6",
            hover_color="#8E44AD",
            action=self.open_add_gesture
        )  # Create Add/Manage Gestures button
        self.collect_data_button = self.create_button(
            text="  Collect Data",
            color="#3498DB",
            hover_color="#2980B9",
            action=self.open_collect_data
        )  # Create Collect Data button
        self.train_model_button = self.create_button(
            text=" Train Model",
            color="#2ECC71",
            hover_color="#27AE60",
            action=self.open_train_model
        )  # Create Train Model button
        self.IoTUpload_button = self.create_button(
            text="IoT Dataset Upload",
            color="#060270",        # Choose a color
            hover_color="#2B3C50",  # Choose a hover color
            action=self.open_IoTUpload
        )  # Create IoT Upload button
        self.translate_button = self.create_button(
            text=" Translate Gestures",
            color="#F39C12",
            hover_color="#E67E22",
            action=self.open_translate_window
        )  # Create Translate Gestures button
        self.reverse_translate_button = self.create_button(
            text="Reverse Translate",
            color="#8E84AD",
            hover_color="#7D3C98",
            action=self.open_reverse_translate
        )  # Create Reverse Translate button
        self.continuous_translate_button = self.create_button(
            text=" Continuous Translate",
            color="#1ABC9C",
            hover_color="#16A085",
            action=self.open_continuous_translate
        )  # Create Continuous Translate button
        self.voicetotext_button = self.create_button(
            text="User\\Guest Window",
            color="#34495E",        # Choose a color
            hover_color="#2C3E50",  # Choose a hover color
            action=self.open_voicetotext
        )  # Create User/Guest Window button
        self.exit_button = self.create_button(
            text=" Exit",
            color="#E74C3C",
            hover_color="#C0392B",
            action=self.confirm_exit
        )  # Create Exit button

        canvas_layout.addWidget(self.add_gesture_button)  # Add button to layout
        canvas_layout.addWidget(self.collect_data_button)  # Add button to layout
        canvas_layout.addWidget(self.train_model_button)  # Add button to layout
        canvas_layout.addWidget(self.IoTUpload_button)  # Add button to layout
        canvas_layout.addWidget(self.translate_button)  # Add button to layout
        canvas_layout.addWidget(self.reverse_translate_button)  # Add button to layout
        canvas_layout.addWidget(self.continuous_translate_button)  # Add button to layout
        canvas_layout.addWidget(self.voicetotext_button)  # Add button to layout
        canvas_layout.addWidget(self.exit_button)  # Add button to layout

        main_layout.addWidget(self.canvas)  # Add canvas to main layout

        self.setLayout(main_layout)  # Set the main layout

    def create_button(self, text, color, hover_color, action):
        btn = QPushButton(text)  # Create a QPushButton with text
        btn.setMinimumHeight(95)  # Set minimum height
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Set size policy
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                border-radius: 4px;
                padding: 2px 6px;
                color: #ECF0F1;
                font-size: 30px;
                font-weight: bold;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
        """)  # Set button style
        btn.clicked.connect(action)  # Connect button to action
        btn.setIconSize(btn.sizeHint())  # Set icon size
        return btn  # Return the button

    def confirm_exit(self):
        reply = QMessageBox.question(self, 'Exit', "Are you sure you want to exit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)  # Confirm exit
        if reply == QMessageBox.Yes:
            QApplication.quit()  # Quit application if yes

    def open_IoTUpload(self):
        # Run the roboflow.py script using Python interpreter
        subprocess.Popen(["python3", "roboflow.py"])  # Open IoT upload script

    def open_voicetotext(self):
        # Run the voicetotext.py script using Python interpreter
        subprocess.Popen(["python3", "user_guest.py"])  # Open User/Guest window script

    def open_add_gesture(self):
        self.add_window = AddGestureWindow()  # Create AddGestureWindow
        self.add_window.show()  # Show the window

    def open_collect_data(self):
        self.collect_window = CollectDataWindow()  # Create CollectDataWindow
        self.collect_window.show()  # Show the window

    def open_train_model(self):
        self.train_window = TrainWindow()  # Create TrainWindow
        self.train_window.show()  # Show the window

    def open_translate_window(self):
        self.translate_window = TranslateWindow()  # Create TranslateWindow
        self.translate_window.show()  # Show the window

    def open_reverse_translate(self):
        self.reverse_window = ReverseTranslateWindow()  # Create ReverseTranslateWindow
        self.reverse_window.show()  # Show the window

    def open_continuous_translate(self):
        self.continuous_window = ContinuousTranslateWindow()  # Create ContinuousTranslateWindow
        self.continuous_window.show()  # Show the window

if __name__ == "__main__":
    from PyQt5.QtCore import QCoreApplication  # Import QCoreApplication

    # Set library paths to include the directory containing Qt plugins
    QCoreApplication.setLibraryPaths(["/usr/lib/aarch64-linux-gnu/qt5/plugins"])  # Set Qt library paths

    app = QApplication(sys.argv)  # Create QApplication
    app.setStyleSheet("""
    QWidget {
        background-color: #0388FD;
        color: #FFFFFF;
        font-family: Arial;
        font-size: 14px;
    }
    QLabel {
        font-size: 20px;
        font-weight: bold;
        color: #FFFFFF;
    }
    QLineEdit {
        background-color: #40444B;
        border: 1px solid #5B6EAE;
        border-radius: 5px;
        padding: 5px;
        color: #FFFFFF;
    }
    QComboBox {
        background-color: #40444B;
        border: 1px solid #5B6EAE;
        border-radius: 5px;
        padding: 5px;
        color: #FFFFFF;
    }
    QComboBox QAbstractItemView {
        background-color: #40444B;
        selection-background-color: #5B6EAE;
        color: #FFFFFF;
    }
    QListWidget {
        background-color: #40444B;
        border: 1px solid #5B6EAE;
        border-radius: 5px;
        color: #FFFFFF;
    }
    QFrame#canvasFrame {
        background-color: #23272A;
        border: 2px solid #5B6EAE;
        border-radius: 10px;
        padding: 10px;
    }
    QFrame#collectDataFrame {
        background-color: #23272A;
        border: 2px solid #5B6EAE;
        border-radius: 10px;
        padding: 15px;
    }
    """)  # Set application-wide styles
    window = MainMenu()  # Create main menu window
    window.show()  # Show the window
    sys.exit(app.exec_())  # Execute the application and exit on close
