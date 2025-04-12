import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
path_scripts = Path(__file__).resolve().parents[1]
sys.path.append(str(path_scripts))
from model.normalize import normalize_landmarks
from model_angle.angles import calculate_adjacent_angles
from matplotlib.patches import Arc
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

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

    angles = calculate_adjacent_angles(mean_coordinates.reshape(-1, 2))

    plt.figure(figsize=(8, 8))

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

    angle_labels = [
        "Thumb-Index", "Index-Middle", "Middle-Ring", "Ring-Pinky",  
        "Thumb-Index", "Index-Middle", "Middle-Ring", "Ring-Pinky"   
    ]

    finger_tip_indices = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips

    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=min(angles), vmax=max(angles))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for i, angle in enumerate(angles):
        if i < 4:  # Angles for Hand 1
            # Get the coordinates of the current finger tip and the next finger tip
            x1, y1 = hand1_x[finger_tip_indices[i]], hand1_y[finger_tip_indices[i]]
            x2, y2 = hand1_x[finger_tip_indices[i + 1]], hand1_y[finger_tip_indices[i + 1]]
            wrist_x, wrist_y = hand1_x[0], hand1_y[0]
        else:  # Angles for Hand 2
            # Get the coordinates of the current finger tip and the next finger tip
            x1, y1 = hand2_x[finger_tip_indices[i - 4]], hand2_y[finger_tip_indices[i - 4]]
            x2, y2 = hand2_x[finger_tip_indices[i - 3]], hand2_y[finger_tip_indices[i - 3]]
            wrist_x, wrist_y = hand2_x[0], hand2_y[0]

        plt.plot([wrist_x, x1], [wrist_y, y1], linestyle='dotted', color='green', alpha=0.5)
        plt.plot([wrist_x, x2], [wrist_y, y2], linestyle='dotted', color='green', alpha=0.5)

        angle1 = np.degrees(np.arctan2(y1 - wrist_y, x1 - wrist_x))
        angle2 = np.degrees(np.arctan2(y2 - wrist_y, x2 - wrist_x))

        if abs(angle2 - angle1) > 180:
            if angle1 < angle2:
                angle1 += 360
            else:
                angle2 += 360

        arc_color = cmap(norm(angle))
        arc = Arc((wrist_x, wrist_y), 2 * np.sqrt((x1 - wrist_x)**2 + (y1 - wrist_y)**2),
                  2 * np.sqrt((x2 - wrist_x)**2 + (y2 - wrist_y)**2),
                  theta1=min(angle1, angle2), theta2=max(angle1, angle2), color=arc_color, alpha=0.8, lw=2)
        plt.gca().add_patch(arc)

        mid_angle = (angle1 + angle2) / 2
        label_distance = 1.2 * np.sqrt((x1 - wrist_x)**2 + (y1 - wrist_y)**2)  # Increase the label distance
        label_x = wrist_x + label_distance * np.cos(np.radians(mid_angle))
        label_y = wrist_y + label_distance * np.sin(np.radians(mid_angle))

        plt.text(label_x, label_y, f"{angle_labels[i]}: {angle*180:.1f}°", fontsize=9, color=arc_color, ha='center', va='center')

    # cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', label='Angle (degrees)')
    # cbar.set_ticks(np.linspace(min(angles), max(angles), 5))
    # cbar.set_ticklabels([f"{x*180:.1f}°" for x in np.linspace(min(angles), max(angles), 5)])

    
    plt.title(f'Mean Landmarks for Label {target_label}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()

    plt.show()

if __name__ == '__main__':
    show_mean_landmarks_label('C', 'dataset_collection/landmark_data_84_data_add.pickle')
