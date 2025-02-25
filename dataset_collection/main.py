from initialize import initialize_directories
from capture_letter import capture_single_letter_sequences
from process_image import process_frames_and_extract_landmarks
from save_landmark import save_landmark_data
from augment import duplicate_sequences_with_augmentation
from convert import process_npy_files

DATA_DIR = './frames'
number_of_classes = 26
sequence_length = 100
dataset_size = 2
num_augment_sequences = 2

"""
Step 1 to 3b is now mainly not used as we have the images already
step 4 to 5 needs to be done again once images from a different source are used
"""

def main():
    # # Step 1: Initialize directories for each class (A-Z).
    # initialize_directories(DATA_DIR, number_of_classes)
    
    # # Step 2: Capture sequences with user-controlled starts.
    # capture_single_letter_sequences(DATA_DIR, sequence_length, dataset_size)
    
    # # Step 3: augmenting currently optional due to constraints of storage space
    # duplicate_sequences_with_augmentation(DATA_DIR, number_of_classes, num_new_sequences=num_augment_sequences)

    # # Step 3b: This was done to convert the npy files to jpg files
    # process_npy_files(DATA_DIR, number_of_classes)
    
    # Step 4: Process all sequences (original + augmented) to extract landmarks.
    data, labels = process_frames_and_extract_landmarks(DATA_DIR, number_of_classes)
    
    # Step 5: Save the final data.
    save_landmark_data("landmark_data_84_additional_data.pickle", data, labels)


if __name__ == "__main__":
    main()
