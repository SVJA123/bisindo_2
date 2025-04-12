import os
import numpy as np
import cv2

# This was a helper function to convert the .npy files to frames

def process_npy_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through label subdirectories
    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir):
            continue

        label_output_dir = os.path.join(output_dir, label)
        os.makedirs(label_output_dir, exist_ok=True)

        # Process each .npy file in the label directory
        for npy_file in os.listdir(label_dir):
            if npy_file.endswith('.npy'):
                npy_path = os.path.join(label_dir, npy_file)
                print(f"Processing: {npy_path}")

                # load the .npy file
                sequence = np.load(npy_path, allow_pickle=True)  
                print(f"Loaded sequence of shape: {sequence.shape}")

                for i, frame in enumerate(sequence):
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    frame_filename = f"{npy_file[:-4]}_frame_{i+1:03d}.jpg"
                    frame_path = os.path.join(label_output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    print(f"Saved frame: {frame_path}")


if __name__ == "__main__":
    input_directory = "data"  
    output_directory = "data"  
    process_npy_files(input_directory, output_directory)
