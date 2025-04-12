from keras.models import load_model
from load_data import load_data
from evaluate_model import evaluate_model

def main():
    data_path = "validation/landmark_data_test.pickle" 
    model_path = "model_hybrid/hand_gesture_model_hybrid_data_add.keras" 
    num_classes = 26  
    labels = [chr(i) for i in range(65, 65 + num_classes)] 

    X_coords, X_angles, y_true = load_data(data_path, num_classes)

    model = load_model(model_path)

    evaluate_model(model, X_coords, X_angles, y_true, labels)

if __name__ == "__main__":
    main()