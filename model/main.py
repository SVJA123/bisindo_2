from matplotlib import pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from preprocess import preprocess_hand_landmark_data
from train import train_model

# model 2 is the correct one

def main():
    with open("dataset_collection/landmark_data_84_data_add.pickle", "rb") as f:
        data_dict = pickle.load(f)
    data, labels = data_dict['data'], data_dict['labels']


    data = np.array([np.array(d) for d in data])

    num_classes = 26 
    X, y = preprocess_hand_landmark_data(data, labels, num_classes)

    print(X.shape)
    print(y.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = X_train.shape[1:] 

    print(input_shape)

    model, history = train_model(X_train, y_train, X_val, y_val, input_shape, num_classes)

    model.save("hand_gesture_model_data_add.keras")
    print("Model trained and saved as 'hand_gesture_model_data_add.keras'")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

if __name__ == "__main__":
    main()