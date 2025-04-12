import cv2
import os
import numpy as np
import string
import time

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
                print("Starting capture in 2 seconds...")
                time.sleep(2)  # Add a 2-second delay before starting capture
                print(f"Capturing {sequence_length} frames...")
            elif key == ord('q'):
                print("Exiting capture early.")
                cap.release()
                cv2.destroyAllWindows()
                return

        for frame_num in range(sequence_length):
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
            
            frame_filename = f"{chosen_letter}_seq{seq_num}_frame{frame_num}.jpg"
            frame_path = os.path.join(class_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            print(f"Saved frame: {frame_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nAll {dataset_size} sequences for letter '{chosen_letter}' have been captured.")