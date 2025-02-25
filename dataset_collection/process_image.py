import cv2
import mediapipe as mp
import numpy as np
import os

#  This is a part of the proceesiong of the images to extract the landmarks

def process_frames_and_extract_landmarks(DATA_DIR, number_of_classes):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    
    data = []
    labels = []

    for i in range(number_of_classes):
        label = chr(65 + i)  # 'A' -> 0, 'B' -> 1, etc.
        class_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(class_dir):
            continue  

        frames = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))]
        frames.sort()  
        
        sequence_data = []

        for frame_file in frames:
            print(frame_file)
            frame_path = os.path.join(class_dir, frame_file)

            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Warning: Failed to read {frame_path}")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)

            frame_landmarks = np.zeros(84, dtype=np.float32)

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if hand_idx >= 2:  
                        break
                    offset = hand_idx * 42  
                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x, lm.y])
                    frame_landmarks[offset:offset+42] = coords

            sequence_data.append(frame_landmarks)
            labels.append(label)


        if sequence_data:
            data.append(sequence_data)

    hands.close()
    return data, labels

# Example usage
if __name__ == "__main__":
    DATA_DIR = "./frames"  
    number_of_classes = 26  

    data, labels = process_frames_and_extract_landmarks(DATA_DIR, number_of_classes)
    print(f"Extracted {len(data)} sequences with corresponding labels.")
    print(f"First sequence shape: {np.array(data[0]).shape}") 
