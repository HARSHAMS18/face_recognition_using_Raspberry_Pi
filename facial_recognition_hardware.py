import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle
import RPi.GPIO as GPIO

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)}))
picam2.start()

# GPIO setup for the stepper motor
step_pins = [17, 18, 27, 22]  # Replace with your GPIO pins
for pin in step_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, False)

# Define stepper motor sequence for 28BYJ-48 stepper motor
step_sequence = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
]

step_delay = 0.001  # Delay between steps

# Initialize variables
cv_scaler = 4
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0
authorized_names = ["Shivu", "Harsha"]  # Replace with names you wish to authorize
motor_position = 0  # Current motor position (0 or 180 degrees)

def rotate_stepper(steps, direction):
    """
    Rotate the stepper motor a given number of steps in the specified direction.
    Positive steps for clockwise, negative steps for counter-clockwise.
    """
    global motor_position
    step_count = len(step_sequence)
    for _ in range(abs(steps)):
        for step in step_sequence[::direction]:
            for pin, output in zip(step_pins, step):
                GPIO.output(pin, output)
            time.sleep(step_delay)
    motor_position += (steps if direction == 1 else -steps)

def process_frame(frame):
    global face_locations, face_encodings, face_names, motor_position
    
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
    
    face_names = []
    authorized_face_detected = False
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            if name in authorized_names:
                authorized_face_detected = True
        face_names.append(name)
    
    if authorized_face_detected and motor_position == 0:
        rotate_stepper(512, 1)  # 512 steps for 180-degree rotation
    elif not authorized_face_detected and motor_position == 180:
        rotate_stepper(512, -1)  # 512 steps back to 0 degrees
    
    return frame

def draw_results(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        cv2.rectangle(frame, (left -3, top - 35), (right+3, top), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
        if name in authorized_names:
            cv2.putText(frame, "Authorized", (left + 6, bottom + 23), font, 0.6, (0, 255, 0), 1)
    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

try:
    while True:
        frame = picam2.capture_array()
        processed_frame = process_frame(frame)
        display_frame = draw_results(processed_frame)
        current_fps = calculate_fps()
        cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video', display_frame)
        if cv2.waitKey(1) == ord("q"):
            break
finally:
    GPIO.cleanup()
    cv2.destroyAllWindows()
    picam2.stop()
