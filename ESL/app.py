from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import copy
import itertools
import numpy as np
import pandas as pd
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFont
import io
import os

app = Flask(__name__)
model = keras.models.load_model("model.h5")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Detected character global variable
detected_character = ''

# Gujarati alphabet and corresponding English representation
gujarati_alphabet = [
    'અ', 'બ', 'ક', 'ડ', 'એ', 'ફ', 'ગ', 'હ', 'ઇ', 
    'જ', 'ક', 'લ', 'મ', 'ન', 'ઓ', 'પ', 'ક્', 'ર', 
    'સ', 'ત', 'ઉ', 'વ', 'વ', 'ક્', 'ય', 'ઝ'
]

english_representation = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

def load_font():
    font_path = "english.ttf"  # Update this path
    if not os.path.isfile(font_path):
        print(f"Font file not found: {font_path}")
        return ImageFont.load_default()

    try:
        return ImageFont.truetype(font_path, 32)
    except OSError as e:
        print(f"Error loading font: {e}")
        return ImageFont.load_default()

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value if max_value != 0 else 0

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def generate_frames():
    global detected_character
    cap = cv2.VideoCapture(0)
    font = load_font()

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            debug_image = copy.deepcopy(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    df = pd.DataFrame(pre_processed_landmark_list).transpose()

                    predictions = model.predict(df, verbose=0)
                    predicted_classes = np.argmax(predictions, axis=1)

                    if predicted_classes.size > 0:  # Check if there are predictions
                        detected_character = gujarati_alphabet[predicted_classes[0]]
                        detected_character_english = english_representation[predicted_classes[0]]
                    else:
                        detected_character = ''  # Default value if no predictions
                        detected_character_english = ''  # Default value if no predictions

            # Continue with drawing landmarks and streaming frames
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_character')
def get_character():
    # Return the English representation instead of Gujarati
    detected_character_english = english_representation[gujarati_alphabet.index(detected_character)] if detected_character in gujarati_alphabet else ''
    return jsonify({'character': detected_character_english})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
