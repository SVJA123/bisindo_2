import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from preprocess import preprocess_hand_landmark_data
from train import train_model
from normalize import normalize

def main():
    with open("dataset_collection/landmark_data_84.pickle", "rb") as f:
        data_dict = pickle.load(f)
    data, labels = data_dict['data'], data_dict['labels']


    data = np.array([np.array(d) for d in data])

    data = normalize(data)

    # Preprocess the data
    num_classes = 26 
    X, y = preprocess_hand_landmark_data(data, labels, num_classes)

    print(X.shape)
    print(y.shape)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = X_train.shape[1:]  # Shape: (42, 2, 1)

    # Train the model
    model = train_model(X_train, y_train, X_val, y_val, input_shape, num_classes)

    # Save the trained model
    model.save("hand_gesture_model_additional_data.keras")
    print("Model trained and saved as 'hand_gesture_model_additional_data.keras'")

if __name__ == "__main__":
    main()