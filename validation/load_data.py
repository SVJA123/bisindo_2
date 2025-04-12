import pickle
import numpy as np
import sys
from pathlib import Path
path_scripts = Path(__file__).resolve().parents[1]
sys.path.append(str(path_scripts))
from preprocess import preprocess_hand_landmark_data

def load_data(data_path, num_classes):
    with open(data_path, "rb") as f:
        data_dict = pickle.load(f)
    data, labels = data_dict['data'], data_dict['labels']
    data = np.array([np.array(d) for d in data])
    X_coords, X_angles, y = preprocess_hand_landmark_data(data, labels, num_classes)
    return X_coords, X_angles, y