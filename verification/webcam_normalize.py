import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import sys
from pathlib import Path
path_scripts = Path(__file__).resolve().parents[1]
sys.path.append(str(path_scripts))
from model.normalize import normalize_landmarks

# Since we're going to normalize the training data, we should also normalize the input data the same way

def process_frame(frame, hands):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    landmarks = np.zeros(84, dtype=np.float32)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx >= 2: 
                break
            offset = idx * 42  
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y])
            landmarks[offset:offset + 42] = coords
        
        landmarks = normalize_landmarks(landmarks)

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    return landmarks, frame


def main():
    model_path = "model/hand_gesture_model_data_add.keras"  
    model = load_model(model_path)

    global mp_hands, mp_drawing, mp_drawing_styles
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    labels = [chr(i) for i in range(65, 65 + 26)]  # A-Z

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'Q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        landmarks, annotated_frame = process_frame(frame, hands)

        if not np.allclose(landmarks, 0):
            input_data = landmarks.reshape(-1, 42, 2, 1) 

            prediction = model.predict(input_data, verbose=0)
            predicted_label = labels[np.argmax(prediction)]

            cv2.putText(
                annotated_frame,
                f"Prediction: {predicted_label}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("Gesture Recognition", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
