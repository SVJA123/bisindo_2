from initialize import initialize_directories
from capture_letter import capture_single_letter_sequences
from process_image import process_frames_and_extract_landmarks
from save_landmark import save_landmark_data
from augment import duplicate_sequences_with_augmentation, duplicate_single_subfolder
from convert import process_npy_files

DATA_DIR = './data' # training data directory
# DATA_DIR = './data_add' # test data directory
number_of_classes = 26
section_length = 100
num_sections = 2
num_augment_sections = 2

"""
Step 1 to 3b is now mainly not used as we have the images already
step 4 to 5 needs to be done again once images from a different source are used
"""

def main():
    # # Step 1: Initialize directories for each class (A-Z).
    # initialize_directories(DATA_DIR, number_of_classes)
    
    # Step 2: Capture sequences with user-controlled starts.
    capture_single_letter_sequences(DATA_DIR, section_length, num_sections)
    
    # Step 3: augmenting currently optional due to constraints of storage space
    duplicate_sequences_with_augmentation(DATA_DIR, number_of_classes, num_new_sequences=num_augment_sections)
    # for individual letter augmentation
    # duplicate_single_subfolder(DATA_DIR, 'U', num_new_sequences=num_augment_sections)


    # Step 3b: This was done to convert the npy files to jpg files
    # process_npy_files(DATA_DIR, './temp2')
    
    # Step 4: Process all sequences (original + augmented) to extract landmarks.
    data, labels = process_frames_and_extract_landmarks(DATA_DIR, number_of_classes)
    
    # Step 5: Save the final data.
    save_landmark_data("test/landmark_data_test.pickle", data, labels)


if __name__ == "__main__":
    main()
