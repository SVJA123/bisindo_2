import numpy as np
from tensorflow.keras.utils import to_categorical
from normalize import normalize_landmarks

def preprocess_hand_landmark_data(data, labels, num_classes):
    processed_data = []

    for i, images in enumerate(data):
        for frames in images:
            if frames.shape == (84,):
                landmarks = frames.reshape(-1, 2)
                normalized_landmarks = normalize_landmarks(landmarks)
                processed_data.append(normalized_landmarks)

    X = np.array(processed_data).reshape(-1, 42, 2, 1)  

    label_to_index = {chr(i): i - 65 for i in range(65, 91)} 
    integer_labels = [label_to_index[label] for label in labels]

    y = to_categorical(integer_labels, num_classes=num_classes)

    return X, y

