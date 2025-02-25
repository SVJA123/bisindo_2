import numpy as np
from sklearn.preprocessing import LabelBinarizer

def normalize_landmarks(landmarks):
    # Reshape into (42, 2) where first 21 keypoints = Hand 1, next 21 keypoints = Hand 2
    landmarks = landmarks.reshape(-1, 2)

    # Check if second hand exists
    second_hand_exists = not (np.all(landmarks[21:] == 0))

    # Compute min and max across BOTH hands
    min_x, max_x = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
    min_y, max_y = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])

    # Normalize x and y coordinates using global min/max
    if max_x - min_x != 0:
        landmarks[:, 0] = (landmarks[:, 0] - min_x) / (max_x - min_x)
    if max_y - min_y != 0:
        landmarks[:, 1] = (landmarks[:, 1] - min_y) / (max_y - min_y)

    # If second hand is missing, don't normalize those coordinates
    if not second_hand_exists:
        landmarks[21:] = 0  

    return landmarks.flatten() 


def normalize(data):
    """
    Preprocess the hand landmark data for static frames.

    Args:
    - data: List of sequences, where each sequence is a list of 126-d numpy arrays (landmarks).
    - labels: List of labels corresponding to each sequence.
    - num_classes: Number of classes (e.g., 26 for A-Z gestures).

    Returns:
    - X: Preprocessed data as a numpy array of shape (num_samples, 126).
    - y: One-hot encoded labels as a numpy array.
    """
    processed_data = []

    for i, labels in enumerate(data):
        for frames in labels:
            if frames.shape == (84,):
                normalized_landmarks = normalize_landmarks(frames)
                processed_data.append(normalized_landmarks)

    return np.array(processed_data)

if __name__:
    data = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], [[13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]]
    labels = ['A', 'B']
    data = np.array(data)
    print(normalize(data))