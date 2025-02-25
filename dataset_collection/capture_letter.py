import cv2
import os
import numpy as np
import string

def capture_single_letter_sequences(DATA_DIR, sequence_length, dataset_size):

    chosen_letter = input("Which letter do you want to capture? [A-Z]: ").upper()

    if chosen_letter not in string.ascii_uppercase:
        print("Invalid letter. Please choose a single letter from A to Z.")
        return

    class_dir = os.path.join(DATA_DIR, chosen_letter)
    os.makedirs(class_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    print(f"\n--- Collecting sequences for class '{chosen_letter}' ---")

    for seq_num in range(dataset_size):
        print(f"\nSequence {seq_num+1} of {dataset_size} for class '{chosen_letter}'")
        print("Position your hands, then press 'S' to start capturing the sequence.")
        print("Press 'Q' at any time to quit.")
        
        # Wait for user input to start capturing
        capturing = False
        while not capturing:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from camera.")
                continue
            
            cv2.putText(
                frame,
                f"Class {chosen_letter} Seq {seq_num}: Press 'S' to start or 'Q' to quit",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                capturing = True
                print(f"Capturing {sequence_length} frames...")
            elif key == ord('q'):
                print("Exiting capture early.")
                cap.release()
                cv2.destroyAllWindows()
                return

        # Now capture the actual sequence of frames
        sequence_frames = []
        for _ in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                continue
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Capture interrupted by user.")
                cap.release()
                cv2.destroyAllWindows()
                return
            
            sequence_frames.append(frame)

        # Save the sequence as a single .npy file
        sequence_array = np.array(sequence_frames, dtype=object)  # dtype=object to store frames of varying shapes
        save_path = os.path.join(class_dir, f'sequence_{seq_num}.npy')
        np.save(save_path, sequence_array)
        print(f"Sequence {seq_num} saved for class '{chosen_letter}'.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nAll {dataset_size} sequences for letter '{chosen_letter}' have been captured.")
