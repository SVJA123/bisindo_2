import numpy as np
from tensorflow.keras.utils import to_categorical
import sys
from pathlib import Path
path_scripts = Path(__file__).resolve().parents[1]
sys.path.append(str(path_scripts))
from model_hybrid.normalize import normalize_landmarks
from model_hybrid.angles import calculate_adjacent_angles


def preprocess_hand_landmark_data(data, labels, num_classes):
    processed_coords = []
    processed_angles = []

    for i, images in enumerate(data):
        for frames in images:
            if frames.shape == (84,):  
                landmarks = frames.reshape(-1, 2)
                normalized_landmarks = normalize_landmarks(landmarks)
                angles = calculate_adjacent_angles(landmarks)
                processed_coords.append(normalized_landmarks)
                processed_angles.append(angles)

    X_coords = np.array(processed_coords).reshape(-1, 42, 2, 1) 

    X_angles = np.array(processed_angles) 

    label_to_index = {chr(i): i - 65 for i in range(65, 91)}
    integer_labels = [label_to_index[label] for label in labels]
    y = to_categorical(integer_labels, num_classes=num_classes)

    return X_coords, X_angles, y
