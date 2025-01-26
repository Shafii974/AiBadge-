import tkinter as tk  # Import Tkinter for GUI creation
from threading import Thread  # Import Thread for running tasks in the background
import speech_recognition as sr  # Import speech_recognition for handling voice input
from PIL import Image, ImageTk  # Import PIL for image processing
import arabic_reshaper  # Import arabic_reshaper to handle Arabic text
from bidi.algorithm import get_display  # Import get_display for bidirectional text support
import lgpio  # Import lgpio for GPIO control

# New imports for gesture recognition, camera, and Arabic TTS
import cv2  # Import OpenCV for image and video processing
import mediapipe as mp  # Import MediaPipe for hand tracking
import numpy as np  # Import NumPy for numerical operations
from picamera2 import Picamera2  # Import Picamera2 for camera access
import joblib  # Import joblib for saving and loading models
from tensorflow.keras.models import load_model  # Import load_model to load trained models
import pyttsx3  # Import pyttsx3 for text-to-speech
from gtts import gTTS  # Import gTTS for Google Text-to-Speech
import subprocess, os  # Import subprocess and os for running external processes and OS operations

# GPIO setup
green_led = 21  # GPIO pin for green LED
red_led = 13  # GPIO pin for red LED
gpio_handle = lgpio.gpiochip_open(0)  # Open GPIO chip 0
lgpio.gpio_claim_output(gpio_handle, green_led)  # Claim green LED pin as output
lgpio.gpio_claim_output(gpio_handle, red_led)  # Claim red LED pin as output

def set_led_state(green_on, red_on):
    # Function to set the state of the green and red LEDs
    lgpio.gpio_write(gpio_handle, green_led, 1 if green_on else 0)  # Turn green LED on or off
    lgpio.gpio_write(gpio_handle, red_led, 1 if red_on else 0)  # Turn red LED on or off

# Desired dimensions and settings
LOGO_WIDTH, LOGO_HEIGHT = 200, 75  # Dimensions for logo images
MIC_WIDTH, MIC_HEIGHT = 100, 100  # Dimensions for microphone image
MAX_LINES = 100  # Maximum number of lines in text widgets

global_last_gesture = None  # Variable to store the last recognized gesture

root = tk.Tk()  # Create the main Tkinter window
root.title("User Window")  # Set the window title
root.geometry("720x600+0+0")  # Set the window size and position
root.configure(bg="#50ABE7")  # Set the background color

mic_logo_image = None  # Initialize microphone image variable
try:
    mic_img = Image.open("mic.png")  # Open the microphone image
    mic_img = mic_img.resize((MIC_WIDTH, MIC_HEIGHT), Image.LANCZOS)  # Resize the image
    mic_logo_image = ImageTk.PhotoImage(mic_img)  # Convert to Tkinter-compatible image
except Exception as e:
    print("Error loading mic image:", e)  # Print error if image loading fails

def load_logos():
    # Function to load left and right logo images
    try:
        left_img = Image.open("qstss.png")  # Open left logo image
        right_img = Image.open("hamdan.png")  # Open right logo image
        left_img = left_img.resize((LOGO_WIDTH, LOGO_HEIGHT), Image.LANCZOS)  # Resize left logo
        right_img = right_img.resize((LOGO_WIDTH, LOGO_HEIGHT), Image.LANCZOS)  # Resize right logo
        return ImageTk.PhotoImage(left_img), ImageTk.PhotoImage(right_img)  # Return Tkinter-compatible images
    except Exception as e:
        print("Error loading images:", e)  # Print error if image loading fails
        return None, None  # Return None if loading fails

left_logo_image, right_logo_image = load_logos()  # Load the logo images

top_frame = tk.Frame(root, bg="#50ABE7")  # Create a frame for the top section
top_frame.pack(fill='x', pady=10)  # Pack the frame horizontally with padding

if left_logo_image:
    left_logo_label = tk.Label(top_frame, image=left_logo_image, bg="#50ABE7")  # Create label with left logo
    left_logo_label.image = left_logo_image  # Keep a reference to the image
    left_logo_label.pack(side="left", padx=10)  # Pack the label to the left
else:
    left_logo_label = tk.Label(top_frame, text="Left Logo", font=("Helvetica", 18, "bold"),
                               fg="white", bg="#50ABE7")  # Create text label if image not found
    left_logo_label.pack(side="left", padx=10)  # Pack the label to the left

title_label = tk.Label(top_frame, text="Ai Smart Badge",
                       font=("Helvetica", 18, "bold"), fg="white", bg="#50ABE7")  # Create title label
title_label.pack(side="left", expand=True)  # Pack the title label in the center

if right_logo_image:
    right_logo_label = tk.Label(top_frame, image=right_logo_image, bg="#50ABE7")  # Create label with right logo
    right_logo_label.image = right_logo_image  # Keep a reference to the image
    right_logo_label.pack(side="right", padx=10)  # Pack the label to the right
else:
    right_logo_label = tk.Label(top_frame, text="Right Logo", font=("Helvetica", 18, "bold"),
                                fg="white", bg="#50ABE7")  # Create text label if image not found
    right_logo_label.pack(side="right", padx=10)  # Pack the label to the right

listening_frame = tk.Frame(root, bg="#50ABE7")  # Create a frame for the listening section
listening_frame.pack(pady=20)  # Pack the frame with vertical padding

if mic_logo_image:
    mic_label = tk.Label(listening_frame, image=mic_logo_image, bg="#50ABE7")  # Create label with mic image
    mic_label.image = mic_logo_image  # Keep a reference to the image
    mic_label.pack(side="left", padx=5)  # Pack the label to the left
else:
    mic_label = tk.Label(listening_frame, text="(mic)", font=("Helvetica", 24, "bold"),
                         fg="black", bg="#50ABE7")  # Create text label if mic image not found
    mic_label.pack(side="left", padx=5)  # Pack the label to the left

listening_label = tk.Label(listening_frame, text="Listening...",
                           font=("Helvetica", 24, "bold"), fg="white", bg="#23BAC4")  # Create listening label
listening_label.pack(side="left", padx=5)  # Pack the label to the left

speaker_button_frame = tk.Frame(root, bg="#50ABE7")  # Create a frame for speaker buttons
speaker_button_frame.pack(side="bottom", pady=10)  # Pack the frame at the bottom with padding

exit_button_speaker = tk.Button(speaker_button_frame, text="Exit", command=root.destroy,
                                font=("Helvetica", 18, "bold"), fg="white", bg="#DC143C",
                                relief=tk.RAISED, bd=5, padx=20, pady=5)  # Create Exit button
exit_button_speaker.pack(side="right", padx=10)  # Pack the Exit button to the right

lang_frame = tk.LabelFrame(root, text="Select Language", padx=10, pady=10,
                           font=("Helvetica", 18, "bold"), fg="white", bg="#50ABE7",
                           relief=tk.RIDGE, bd=3)  # Create a frame for language selection
lang_frame.pack(pady=10)  # Pack the frame with padding

selected_language = tk.StringVar(value='en-US')  # Variable to store selected language

eng_radio = tk.Radiobutton(lang_frame, text="English", variable=selected_language,
                           value='en-US', font=("Helvetica", 18, "bold"),
                           fg="black", bg="#50ABE7",
                           selectcolor="#50ABE7", activebackground="#50ABE7",
                           activeforeground="white")  # Create English radio button
eng_radio.pack(anchor='w', pady=5)  # Pack the radio button to the west with padding

ara_radio = tk.Radiobutton(lang_frame, text="Arabic", variable=selected_language,
                           value='ar-EG', font=("Helvetica", 18, "bold"),
                           fg="white", bg="#50ABE7",
                           selectcolor="#50ABE7", activebackground="#50ABE7",
                           activeforeground="white")  # Create Arabic radio button
ara_radio.pack(anchor='w', pady=5)  # Pack the radio button to the west with padding

# Dark blue gesture label placed below the language selection box
gesture_label = tk.Label(root, text="", font=("Helvetica", 18, "bold"),
                         bg="#00008B", fg="white", width=30, height=2)  # Create gesture display label
gesture_label.pack(pady=10)  # Pack the label with padding

# Adding a Text Widget to the User Window to display voice-to-text
user_text_widget = tk.Text(root, wrap='word', font=("Helvetica", 18, "bold"),
                           fg="white", bg="#50ABE7")  # Create text widget for user
user_text_widget.pack(expand=True, fill='both', padx=10, pady=(20, 50))  # Pack the text widget

text_window = tk.Toplevel(root)  # Create a new top-level window for guest
text_window.title("Guest Window")  # Set window title
text_window.geometry("720x800+0+600")  # Set window size and position
text_window.configure(bg="#50ABE7")  # Set background color

text_top_frame = tk.Frame(text_window, bg="#50ABE7")  # Create a frame in the guest window
text_top_frame.pack(fill='x', pady=10)  # Pack the frame horizontally with padding

if left_logo_image:
    text_left_logo = tk.Label(text_top_frame, image=left_logo_image, bg="#50ABE7")  # Create label with left logo
    text_left_logo.image = left_logo_image  # Keep a reference to the image
    text_left_logo.pack(side="left", padx=10)  # Pack the label to the left
else:
    text_left_logo = tk.Label(text_top_frame, text="Left Logo", font=("Helvetica", 18, "bold"),
                              fg="white", bg="#50ABE7")  # Create text label if image not found
    text_left_logo.pack(side="left", padx=10)  # Pack the label to the left

if right_logo_image:
    text_right_logo = tk.Label(text_top_frame, image=right_logo_image, bg="#50ABE7")  # Create label with right logo
    text_right_logo.image = right_logo_image  # Keep a reference to the image
    text_right_logo.pack(side="right", padx=10)  # Pack the label to the right
else:
    text_right_logo = tk.Label(text_top_frame, text="Right Logo", font=("Helvetica", 18, "bold"),
                               fg="white", bg="#50ABE7")  # Create text label if image not found
    text_right_logo.pack(side="right", padx=10)  # Pack the label to the right

guest_camera_label = tk.Label(text_window, bg="#000000")  # Create label to display camera feed
guest_camera_label.pack(pady=10)  # Pack the label with padding

button_frame = tk.Frame(text_window, bg="#50ABE7")  # Create a frame for buttons in guest window
button_frame.pack(side="bottom", fill='x', pady=10)  # Pack the frame at the bottom with padding

def clear_text():
    # Function to clear text in both text widgets
    text_widget.delete("1.0", tk.END)  # Clear guest text widget
    user_text_widget.delete("1.0", tk.END)  # Clear user text widget

def exit_app():
    root.destroy()  # Function to exit the application

clear_button = tk.Button(button_frame, text="Clear", command=clear_text,
                         font=("Helvetica", 18, "bold"), fg="white", bg="#32CD32",
                         relief=tk.RAISED, bd=5, padx=20, pady=5)  # Create Clear button
clear_button.pack(side="left", padx=10, pady=5)  # Pack the Clear button to the left

exit_button = tk.Button(button_frame, text="Exit", command=exit_app,
                        font=("Helvetica", 18, "bold"), fg="white", bg="#DC143C",
                        relief=tk.RAISED, bd=5, padx=20, pady=5)  # Create Exit button
exit_button.pack(side="left", padx=10, pady=5)  # Pack the Exit button to the left

text_widget = tk.Text(text_window, wrap='word', font=("Helvetica", 18, "bold"),
                      fg="white", bg="#50ABE7")  # Create text widget for guest
text_widget.pack(expand=True, fill='both', padx=10, pady=(20, 50))  # Pack the text widget

def safe_insert_text(text):
    # Function to safely insert text into text widgets
    if selected_language.get() == 'ar-EG' and not text.startswith("Gesture:"):
        reshaped_text = arabic_reshaper.reshape(text)  # Reshape Arabic text
        text = get_display(reshaped_text)  # Apply bidirectional algorithm
    lines = float(text_widget.index('end-1c').split('.')[0])  # Get current number of lines
    if lines > MAX_LINES:
        clear_text()  # Clear text if maximum lines exceeded
    # Insert into Guest Window
    text_widget.insert(tk.END, text)  # Insert text into guest text widget
    text_widget.see(tk.END)  # Scroll to the end
    # Insert into User Window
    user_text_widget.insert(tk.END, text)  # Insert text into user text widget
    user_text_widget.see(tk.END)  # Scroll to the end

def recognize_audio():
    # Function to continuously recognize audio in a separate thread
    recognizer = sr.Recognizer()  # Initialize recognizer
    microphone = sr.Microphone()  # Initialize microphone
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
    while True:
        try:
            set_led_state(True, False)  # Turn green LED on, red off
            with microphone as source:
                audio_data = recognizer.listen(source, timeout=2, phrase_time_limit=3)  # Listen for audio
            language_code = selected_language.get()  # Get selected language
            result = recognizer.recognize_google(audio_data, language=language_code)  # Recognize speech
            text_window.after(0, lambda: safe_insert_text(result + "\n"))  # Insert recognized text
        except sr.WaitTimeoutError:
            set_led_state(False, True)  # Turn red LED on, green off
            continue  # Continue listening
        except sr.UnknownValueError:
            set_led_state(False, True)  # Turn red LED on, green off
            text_window.after(0, lambda: [safe_insert_text("> Could not understand\n"),
                                          set_led_state(False, True)])  # Inform user of misunderstanding
        except sr.RequestError as e:
            set_led_state(False, True)  # Turn red LED on, green off
            text_window.after(0, lambda: safe_insert_text(f"API error: {e}\n"))  # Inform user of API error

recognition_thread = Thread(target=recognize_audio, daemon=True)  # Create a daemon thread for audio recognition
recognition_thread.start()  # Start the recognition thread

mp_hands = mp.solutions.hands  # Alias for MediaPipe hands
mp_drawing = mp.solutions.drawing_utils  # Alias for MediaPipe drawing utilities
model = load_model('models/gesture_recognition_model.keras')  # Load the trained gesture recognition model
scaler = joblib.load('models/scaler.save')  # Load the scaler
label_encoder = joblib.load('models/label_encoder.save')  # Load the label encoder
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)  # Initialize MediaPipe hands

picam2 = Picamera2()  # Initialize the camera
cam_config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (300, 200)})  # Configure camera settings
picam2.configure(cam_config)  # Apply camera configuration
picam2.start()  # Start the camera

engine = pyttsx3.init()  # Initialize text-to-speech engine
voices = engine.getProperty('voices')  # Get available voices
if voices:
    engine.setProperty('voice', voices[0].id)  # Set the first available voice

def speak_text(text):
    # Function to speak text based on selected language
    if selected_language.get() == 'ar-EG':  # Check if Arabic is selected
        try:
            tts = gTTS(text=text, lang='ar')  # Create gTTS object for Arabic
            filename = "temp_arabic.mp3"  # Temporary filename
            tts.save(filename)  # Save the audio file
            subprocess.run(['mpg123', filename])  # Play the audio file
            os.remove(filename)  # Remove the temporary file
        except Exception as e:
            print(f"Arabic TTS error: {e}")  # Print error if TTS fails
    else:
        engine.say(text)  # Use pyttsx3 to speak text
        engine.runAndWait()  # Wait until speaking is done

def update_gesture():
    # Function to update gesture recognition
    global global_last_gesture  # Reference the global last gesture variable
    frame = picam2.capture_array()  # Capture frame from camera
    if frame is not None:
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        results = hands.process(frame_rgb)  # Process the frame with MediaPipe
        gesture = None  # Initialize gesture variable

        if results.multi_hand_landmarks:  # If hands are detected
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Draw landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])  # Extract landmarks
                if landmarks:
                    X = scaler.transform(np.array(landmarks).reshape(1, -1))  # Scale the landmarks
                    prediction = model.predict(X)  # Predict the gesture
                    class_id = np.argmax(prediction)  # Get the class ID
                    gesture = label_encoder.inverse_transform([class_id])[0]  # Decode the gesture

        if frame_rgb is not None:
            im_pil = Image.fromarray(frame_rgb)  # Convert frame to PIL image
            imgtk = ImageTk.PhotoImage(image=im_pil)  # Convert PIL image to Tkinter-compatible image
            guest_camera_label.imgtk = imgtk  # Keep a reference to the image
            guest_camera_label.configure(image=imgtk)  # Display the image

        if gesture and gesture != global_last_gesture:  # If a new gesture is detected
            global_last_gesture = gesture  # Update the last gesture
            display_gesture = gesture  # Prepare gesture text for display
            if selected_language.get() == 'ar-EG':
                reshaped = arabic_reshaper.reshape(gesture)  # Reshape Arabic text
                display_gesture = get_display(reshaped)  # Apply bidirectional algorithm
            # Insert gesture text without bidi for Arabic to keep correct order
            if selected_language.get() == 'ar-EG':
                safe_insert_text(f"Gesture: {display_gesture}\n")  # Insert gesture text for Arabic
            else:
                safe_insert_text(f"Gesture: {display_gesture}\n")  # Insert gesture text for English
            gesture_label.config(text=display_gesture)  # Update gesture label
            root.after(5000, lambda: gesture_label.config(text=""))  # Clear label after 5 seconds
            Thread(target=speak_text, args=(gesture,), daemon=True).start()  # Speak the gesture in a new thread

    root.after(30, update_gesture)  # Schedule the next gesture update

update_gesture()  # Start the gesture update loop

def cleanup_gpio():
    # Function to clean up GPIO and other resources on exit
    set_led_state(False, False)  # Turn off both LEDs
    lgpio.gpiochip_close(gpio_handle)  # Close the GPIO chip
    picam2.stop()  # Stop the camera
    hands.close()  # Close MediaPipe hands
    root.destroy()  # Destroy the main window

root.protocol("WM_DELETE_WINDOW", cleanup_gpio)  # Bind the cleanup function to window close event
root.mainloop()  # Start the Tkinter main loop
