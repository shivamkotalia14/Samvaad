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

gujarati_alphabet = [
    'અ', 'આ', 'ઇ', 'ઈ', 'ઉ', 'ઊ', 'ઋ', 'એ', 'ઐ', 'ઓ', 'ઔ',
    'ક', 'ખ', 'ગ', 'ઘ', 'ઙ', 'ચ', 'છ', 'જ', 'ઝ', 'ઞ',
    'ટ', 'ઠ', 'ડ', 'ઢ', 'ણ', 'ત', 'થ', 'દ', 'ધ', 'ન',
    'પ', 'ફ', 'બ', 'ભ', 'મ', 'ય', 'ર', 'લ', 'વ', 'શ',
    'ષ', 'સ', 'હ', 'ળ', 'ક્', 'ખ્', 'ગ્', 'ઘ્', 'ઙ્',
    'ચ્', 'છ્', 'જ્', 'ઝ્', 'ઞ્', 'ટ્', 'ઠ્', 'ડ્', 'ઢ્', 'ણ્',
    'ત્', 'થ્', 'દ્', 'ધ્', 'ન્', 'પ્', 'ફ્', 'બ્', 'ભ્', 'મ્',
    'ય્', 'ર્', 'લ્', 'વ્', 'શ્', 'ષ્', 'સ્', 'હ્', 'ળ્'
]

def load_font():
    font_path = "gujarati_font.ttf"  # Update this path
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
                    detected_character = gujarati_alphabet[predicted_classes[0]]

                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    # draw = ImageDraw.Draw(pil_image)
                    # draw.text((50, 50), detected_character, font=font, fill=(255, 0, 0))
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

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
    return jsonify({'character': detected_character})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
