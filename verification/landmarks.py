import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
path_scripts = Path(__file__).resolve().parents[1]
sys.path.append(str(path_scripts))

from model.normalize import normalize_landmarks

def normalize(data):
    normalized_data = []
    for sequence in data:
        for frame in sequence:
            if frame.shape == (84,):
                normalized_frame = normalize_landmarks(frame)
                normalized_data.append(normalized_frame)
    return np.array(normalized_data)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm connections
]

def show_mean_landmarks_label(target_label, path_to_pickle_file):
    with open(path_to_pickle_file, 'rb') as file:
        data_dict = pickle.load(file)

    data, labels = data_dict['data'], data_dict['labels']

    unique_labels = list(dict.fromkeys(labels))
    
    target_data = [data[i] for i in range(len(data)) if unique_labels[i] == target_label]

    if not target_data:
        print(f"No data found for label {target_label}")
        return

    target_data = normalize(target_data)

    mean_coordinates = np.mean(target_data, axis=0)

    # Extract coordinates for both hands
    hand1_x = mean_coordinates[0:42:2]  
    hand1_y = mean_coordinates[1:42:2]  
    hand2_x = mean_coordinates[42:84:2] 
    hand2_y = mean_coordinates[43:84:2] 

    plt.figure(figsize=(6, 6))

    # Plot Hand 1 (Blue)
    plt.scatter(hand1_x, hand1_y, color='blue', label='Hand 1')
    for connection in HAND_CONNECTIONS:
        idx1, idx2 = connection
        plt.plot([hand1_x[idx1], hand1_x[idx2]], [hand1_y[idx1], hand1_y[idx2]], color='blue')

    # Plot Hand 2 (Red)
    plt.scatter(hand2_x, hand2_y, color='red', label='Hand 2')
    for connection in HAND_CONNECTIONS:
        idx1, idx2 = connection
        plt.plot([hand2_x[idx1], hand2_x[idx2]], [hand2_y[idx1], hand2_y[idx2]], color='red')

    plt.title(f'Mean Landmarks for Label {target_label}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()

    plt.show()

if __name__ == '__main__':
    # Show the mean landmarks for the label 'O'
    show_mean_landmarks_label('U', 'dataset_collection/landmark_data_84_data_add.pickle')
