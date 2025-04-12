import numpy as np

def normalize_landmarks(landmarks):
    landmarks = landmarks.reshape(-1, 2)

    second_hand_exists = not (np.all(landmarks[21:] == 0))

    min_x, max_x = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
    min_y, max_y = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])

    if max_x - min_x != 0:
        landmarks[:, 0] = (landmarks[:, 0] - min_x) / (max_x - min_x)
    if max_y - min_y != 0:
        landmarks[:, 1] = (landmarks[:, 1] - min_y) / (max_y - min_y)

    if not second_hand_exists:
        landmarks[21:] = 0  

    return landmarks.flatten()  
