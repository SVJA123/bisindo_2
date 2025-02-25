import numpy as np
from tensorflow.keras.utils import to_categorical

def preprocess_hand_landmark_data(data, labels, num_classes):
    # Ensure data is correctly reshaped for CNN input
    X = data.reshape(-1, 42, 2, 1)  

    # Convert labels from characters to integers
    label_to_index = {chr(i): i - 65 for i in range(65, 91)} 
    integer_labels = [label_to_index[label] for label in labels]

    # Convert labels to one-hot encoding
    y = to_categorical(integer_labels, num_classes=num_classes)

    return X, y

