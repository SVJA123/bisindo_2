import os

# This was a helper function to initialize the directories

def initialize_directories(DATA_DIR, number_of_classes):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    for i in range(number_of_classes):
        class_dir = os.path.join(DATA_DIR, chr(65 + i))  # A-Z
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
