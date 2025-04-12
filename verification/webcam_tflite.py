import cv2
import mediapipe as mp
import numpy as np
import sys
import tensorflow as tf 
from pathlib import Path
path_scripts = Path(__file__).resolve().parents[1]
sys.path.append(str(path_scripts))
from model_hybrid.normalize import normalize_landmarks
from model_hybrid.angles import calculate_adjacent_angles



def process_frame(frame, hands):
    """
    Process a single webcam frame to extract hand landmarks and visualize them.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    landmarks = np.zeros((84,), dtype=np.float32)
    angles = np.zeros((8,), dtype=np.float32)

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
        angles = calculate_adjacent_angles(landmarks)

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    return landmarks, angles, frame

def main():
    # Load the TensorFlow Lite model
    tflite_model_path = "model_hybrid/hand_gesture_hybrid_data_add.tflite"
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    print(input_details)


    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    global mp_hands, mp_drawing, mp_drawing_styles
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    labels = [chr(i) for i in range(65, 65+26)]  # ASCII A-Z

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

        landmarks, angle, annotated_frame = process_frame(frame, hands)

        if not np.allclose(landmarks, 0):
            # Prepare the input tensor
            input_landmarks = landmarks.reshape(-1, 42, 2, 1)
            input_angles = angle.reshape(1, 8)
            input_angles = np.array(input_angles, dtype=np.float32)
            input_landmarks = np.array(input_landmarks, dtype=np.float32)
            interpreter.set_tensor(input_details[1]['index'], input_landmarks)  
            interpreter.set_tensor(input_details[0]['index'], input_angles)            
            interpreter.invoke()

            # Retrieve the output
            prediction = interpreter.get_tensor(output_details[0]['index'])
            predicted_label = labels[np.argmax(prediction)]
            cv2.putText(annotated_frame, f"Prediction: {predicted_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Gesture Recognition", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
