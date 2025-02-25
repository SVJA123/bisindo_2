import pickle

def save_landmark_data(output_file, data, labels):
    with open(output_file, "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)
    print(f"Landmark data saved to {output_file}")
