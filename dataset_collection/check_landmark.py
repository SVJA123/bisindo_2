import cv2
import mediapipe as mp
import numpy as np
import os

def check_hand_landmarks(image_path):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Failed to read {image_path}")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        print("Hand landmarks detected.")
    else:
        print("No hand landmarks detected.")

    cv2.imshow('Hand Landmarks', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hands.close()

if __name__ == "__main__":
    image_path = 'data_add/M/M_seq0_frame47.jpg' 
    check_hand_landmarks(image_path)